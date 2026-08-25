//! Reading the binary format without copying it.
//!
//! [`BinaryIndex`] borrows the byte slice it was opened over and decodes on
//! demand. Nothing is allocated at open time beyond the small metadata vectors —
//! the term dictionary, postings and positions stay in the caller's buffer,
//! which may be a memory-mapped file or a `Uint8Array` handed over from
//! JavaScript.
//!
//! # What is validated when
//!
//! [`BinaryIndex::open`] checks the parts that every later read depends on: the
//! magic, the version, and that the section offsets ascend and lie inside the
//! buffer. It deliberately does *not* walk the postings, because that would cost
//! time proportional to the whole file and defeat the point of a lazy format.
//!
//! Everything decoded later is therefore treated as untrusted and bounds
//! checked at the point of use, returning [`FormatError`]. A corrupt index
//! yields an error from the accessor that touched the corrupt bytes; it never
//! panics and never reads outside its section.

use super::varint::Cursor;
use super::{
    read_f64, read_u32, FormatError, Header, FORMAT_VERSION, HEADER_LEN, MAGIC, TERMS_PER_BLOCK,
};

/// One document's entry in a term's posting list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PostingEntry {
    /// Interned document id.
    pub doc_id: u32,
    /// Occurrences of the term in this document field.
    pub term_frequency: u32,
    /// Offset into the positions section, relative to its base.
    positions_offset: u32,
    /// Number of positions stored for this entry.
    position_count: u32,
}

/// A term's postings within one field.
#[derive(Debug, Clone)]
pub struct FieldPostings {
    /// Field id, indexing [`BinaryIndex::field_name`].
    pub field_id: u32,
    /// Documents containing the term in this field, ascending by id.
    pub entries: Vec<PostingEntry>,
}

/// A decoded posting list for one term.
#[derive(Debug, Clone)]
pub struct TermPostings {
    /// Distinct documents containing the term across all fields — BM25's `df`.
    pub document_frequency: u32,
    /// Per-field postings, ascending by field id.
    pub fields: Vec<FieldPostings>,
}

/// A zero-copy view over a binary index.
#[derive(Debug, Clone)]
pub struct BinaryIndex<'a> {
    bytes: &'a [u8],
    header: Header,
    language: String,
    fields: Vec<String>,
    field_boosts: Vec<f64>,
    pipeline: Vec<String>,
    /// Start of the document-reference heap, relative to the docs section.
    doc_heap_base: usize,
    /// Start of the front-coded dictionary, relative to the terms section.
    dictionary_base: usize,
}

impl<'a> BinaryIndex<'a> {
    /// Validate the header and metadata of `bytes`.
    ///
    /// Runs in time proportional to the number of fields and pipeline stages,
    /// not to the size of the index.
    pub fn open(bytes: &'a [u8]) -> Result<Self, FormatError> {
        if bytes.len() < HEADER_LEN {
            return Err(FormatError::TooShort {
                needed: HEADER_LEN,
                got: bytes.len(),
            });
        }
        let magic: [u8; 4] = [bytes[0], bytes[1], bytes[2], bytes[3]];
        if magic != MAGIC {
            return Err(FormatError::BadMagic(magic));
        }
        let format_version = u16::from_le_bytes([bytes[4], bytes[5]]);
        if format_version != FORMAT_VERSION {
            return Err(FormatError::UnsupportedVersion {
                found: format_version,
                supported: FORMAT_VERSION,
            });
        }

        let header = Header {
            format_version,
            flags: u16::from_le_bytes([bytes[6], bytes[7]]),
            doc_count: read_u32(bytes, 8, "header")?,
            document_count: read_u32(bytes, 12, "header")?,
            field_count: read_u32(bytes, 16, "header")?,
            term_count: read_u32(bytes, 20, "header")?,
            k1: read_f64(bytes, 24, "header")?,
            b: read_f64(bytes, 32, "header")?,
            meta_offset: read_u32(bytes, 40, "header")?,
            docs_offset: read_u32(bytes, 44, "header")?,
            terms_offset: read_u32(bytes, 48, "header")?,
            postings_offset: read_u32(bytes, 52, "header")?,
            positions_offset: read_u32(bytes, 56, "header")?,
            end_offset: read_u32(bytes, 60, "header")?,
        };

        // Establishing this once is what lets every later accessor slice its own
        // section without re-checking that the section itself is sane.
        let sections = [
            ("meta", header.meta_offset),
            ("docs", header.docs_offset),
            ("terms", header.terms_offset),
            ("postings", header.postings_offset),
            ("positions", header.positions_offset),
            ("end", header.end_offset),
        ];
        if (header.end_offset as usize) > bytes.len() {
            return Err(FormatError::SectionOutOfBounds {
                section: "end",
                offset: header.end_offset as usize,
                end: bytes.len(),
            });
        }
        for window in sections.windows(2) {
            let (first, first_offset) = window[0];
            let (second, second_offset) = window[1];
            if second_offset < first_offset {
                return Err(FormatError::SectionsNotOrdered { first, second });
            }
            if (second_offset as usize) > bytes.len() {
                return Err(FormatError::SectionOutOfBounds {
                    section: second,
                    offset: second_offset as usize,
                    end: bytes.len(),
                });
            }
        }

        let mut index = Self {
            bytes,
            header,
            language: String::new(),
            fields: Vec::new(),
            field_boosts: Vec::new(),
            pipeline: Vec::new(),
            doc_heap_base: 0,
            dictionary_base: 0,
        };
        index.read_meta()?;
        index.locate_variable_sections()?;
        Ok(index)
    }

    fn read_meta(&mut self) -> Result<(), FormatError> {
        let meta = self.section(self.header.meta_offset, self.header.docs_offset, "meta")?;
        let mut cursor = Cursor::new(meta, "meta");
        self.language = cursor.read_str()?.to_string();
        for _ in 0..self.header.field_count {
            self.fields.push(cursor.read_str()?.to_string());
        }
        let boosts_start = cursor.position();
        for i in 0..self.header.field_count as usize {
            self.field_boosts
                .push(read_f64(meta, boosts_start + i * 8, "meta")?);
        }
        cursor.skip(self.header.field_count as usize * 8)?;
        let stages = cursor.read_usize()?;
        for _ in 0..stages {
            self.pipeline.push(cursor.read_str()?.to_string());
        }
        Ok(())
    }

    /// Compute where the docs section's heap and the term dictionary begin.
    ///
    /// Both follow a run of fixed-width tables whose size is a function of the
    /// header counts, so this is arithmetic — but it is checked once here rather
    /// than recomputed and re-checked on every lookup.
    fn locate_variable_sections(&mut self) -> Result<(), FormatError> {
        let doc_count = self.header.doc_count as usize;
        let field_count = self.header.field_count as usize;
        // (doc_count + 1) offsets, doc_count boosts, doc_count * field_count lengths.
        self.doc_heap_base = (doc_count + 1) * 4 + doc_count * 8 + doc_count * field_count * 4;
        let docs_len = (self.header.terms_offset - self.header.docs_offset) as usize;
        if self.doc_heap_base > docs_len {
            return Err(FormatError::Truncated { section: "docs" });
        }

        let term_count = self.header.term_count as usize;
        // (block_count + 1) block offsets, (term_count + 1) postings offsets.
        self.dictionary_base = (self.header.term_block_count() + 1) * 4 + (term_count + 1) * 4;
        let terms_len = (self.header.postings_offset - self.header.terms_offset) as usize;
        if self.dictionary_base > terms_len {
            return Err(FormatError::Truncated { section: "terms" });
        }
        Ok(())
    }

    /// Slice a section, given its start and the start of the next one.
    fn section(&self, start: u32, end: u32, name: &'static str) -> Result<&'a [u8], FormatError> {
        self.bytes
            .get(start as usize..end as usize)
            .ok_or(FormatError::SectionOutOfBounds {
                section: name,
                offset: start as usize,
                end: self.bytes.len(),
            })
    }

    fn docs_section(&self) -> Result<&'a [u8], FormatError> {
        self.section(self.header.docs_offset, self.header.terms_offset, "docs")
    }

    fn terms_section(&self) -> Result<&'a [u8], FormatError> {
        self.section(
            self.header.terms_offset,
            self.header.postings_offset,
            "terms",
        )
    }

    fn postings_section(&self) -> Result<&'a [u8], FormatError> {
        self.section(
            self.header.postings_offset,
            self.header.positions_offset,
            "postings",
        )
    }

    fn positions_section(&self) -> Result<&'a [u8], FormatError> {
        self.section(
            self.header.positions_offset,
            self.header.end_offset,
            "positions",
        )
    }

    /// The parsed header.
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// Language code the index was built with.
    pub fn language(&self) -> &str {
        &self.language
    }

    /// Indexed field names, in id order.
    pub fn fields(&self) -> &[String] {
        &self.fields
    }

    /// Name of the field with id `field_id`.
    pub fn field_name(&self, field_id: u32) -> Result<&str, FormatError> {
        self.fields
            .get(field_id as usize)
            .map(String::as_str)
            .ok_or(FormatError::InvalidFieldId(field_id))
    }

    /// Boost configured for `field_id`.
    pub fn field_boost(&self, field_id: u32) -> Result<f64, FormatError> {
        self.field_boosts
            .get(field_id as usize)
            .copied()
            .ok_or(FormatError::InvalidFieldId(field_id))
    }

    /// Pipeline stage labels recorded at build time.
    pub fn pipeline(&self) -> &[String] {
        &self.pipeline
    }

    /// BM25's `N`.
    pub fn document_count(&self) -> usize {
        self.header.document_count as usize
    }

    /// Number of distinct interned document references.
    pub fn doc_count(&self) -> usize {
        self.header.doc_count as usize
    }

    /// Number of distinct terms.
    pub fn term_count(&self) -> usize {
        self.header.term_count as usize
    }

    /// BM25 `k1`.
    pub fn k1(&self) -> f64 {
        self.header.k1
    }

    /// BM25 `b`.
    pub fn b(&self) -> f64 {
        self.header.b
    }

    /// The document reference for `doc_id`, borrowed from the buffer.
    pub fn doc_ref(&self, doc_id: u32) -> Result<&'a str, FormatError> {
        if doc_id >= self.header.doc_count {
            return Err(FormatError::InvalidDocId(doc_id));
        }
        let docs = self.docs_section()?;
        let index = doc_id as usize * 4;
        let start = read_u32(docs, index, "docs")? as usize;
        let end = read_u32(docs, index + 4, "docs")? as usize;
        if end < start {
            return Err(FormatError::Truncated { section: "docs" });
        }
        let heap_start = self.doc_heap_base + start;
        let heap_end = self.doc_heap_base + end;
        let slice = docs
            .get(heap_start..heap_end)
            .ok_or(FormatError::Truncated { section: "docs" })?;
        std::str::from_utf8(slice).map_err(|_| FormatError::InvalidUtf8 { section: "docs" })
    }

    /// The id of `doc_ref`, or `None` if the index does not contain it.
    ///
    /// References are stored sorted, so this is a binary search over the heap
    /// rather than a scan.
    pub fn doc_id(&self, doc_ref: &str) -> Result<Option<u32>, FormatError> {
        let mut low = 0u32;
        let mut high = self.header.doc_count;
        while low < high {
            let mid = low + (high - low) / 2;
            match self.doc_ref(mid)?.cmp(doc_ref) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Greater => high = mid,
                std::cmp::Ordering::Equal => return Ok(Some(mid)),
            }
        }
        Ok(None)
    }

    /// Build-time boost for `doc_id`.
    pub fn doc_boost(&self, doc_id: u32) -> Result<f64, FormatError> {
        if doc_id >= self.header.doc_count {
            return Err(FormatError::InvalidDocId(doc_id));
        }
        let docs = self.docs_section()?;
        let base = (self.header.doc_count as usize + 1) * 4;
        read_f64(docs, base + doc_id as usize * 8, "docs")
    }

    /// Token count of one document field.
    pub fn field_length(&self, doc_id: u32, field_id: u32) -> Result<u32, FormatError> {
        if doc_id >= self.header.doc_count {
            return Err(FormatError::InvalidDocId(doc_id));
        }
        if field_id >= self.header.field_count {
            return Err(FormatError::InvalidFieldId(field_id));
        }
        let docs = self.docs_section()?;
        let doc_count = self.header.doc_count as usize;
        let field_count = self.header.field_count as usize;
        let base = (doc_count + 1) * 4 + doc_count * 8;
        let cell = doc_id as usize * field_count + field_id as usize;
        read_u32(docs, base + cell * 4, "docs")
    }

    /// Mean length of `field_id` over documents that have it non-empty.
    ///
    /// Derived rather than stored, for the same reason the JSON format derives
    /// it: it is a pure function of the field-length matrix, so storing it would
    /// create a second source of truth that can drift.
    ///
    /// This walks the whole matrix, so a searcher should compute it once at load
    /// rather than per query.
    pub fn average_field_length(&self, field_id: u32) -> Result<f64, FormatError> {
        if field_id >= self.header.field_count {
            return Err(FormatError::InvalidFieldId(field_id));
        }
        let mut total = 0u64;
        let mut count = 0u64;
        for doc_id in 0..self.header.doc_count {
            let length = self.field_length(doc_id, field_id)?;
            total += u64::from(length);
            count += 1;
        }
        if count == 0 {
            return Ok(0.0);
        }
        Ok(total as f64 / count as f64)
    }

    /// The term with id `term_id`, reconstructed from its front-coded block.
    ///
    /// Costs up to [`TERMS_PER_BLOCK`] prefix reconstructions, and allocates
    /// because a front-coded term does not exist contiguously in the buffer.
    pub fn term(&self, term_id: u32) -> Result<String, FormatError> {
        if term_id >= self.header.term_count {
            return Err(FormatError::InvalidTermId(term_id));
        }
        let block = term_id as usize / TERMS_PER_BLOCK;
        let within = term_id as usize % TERMS_PER_BLOCK;
        let mut cursor = self.block_cursor(block)?;
        let mut term = cursor.read_str()?.to_string();
        for _ in 0..within {
            term = self.next_in_block(&mut cursor, &term)?;
        }
        Ok(term)
    }

    /// A cursor positioned at the start of dictionary block `block`.
    fn block_cursor(&self, block: usize) -> Result<Cursor<'a>, FormatError> {
        if block >= self.header.term_block_count() {
            return Err(FormatError::Truncated { section: "terms" });
        }
        let terms = self.terms_section()?;
        let start = read_u32(terms, block * 4, "terms")? as usize;
        let end = read_u32(terms, (block + 1) * 4, "terms")? as usize;
        if end < start {
            return Err(FormatError::Truncated { section: "terms" });
        }
        let slice = terms
            .get(self.dictionary_base + start..self.dictionary_base + end)
            .ok_or(FormatError::Truncated { section: "terms" })?;
        Ok(Cursor::new(slice, "terms"))
    }

    /// Decode the next front-coded term in a block, given the previous one.
    fn next_in_block(
        &self,
        cursor: &mut Cursor<'a>,
        previous: &str,
    ) -> Result<String, FormatError> {
        let shared = cursor.read_usize()?;
        // A shared prefix past the end of the previous term, or one that lands
        // inside a multi-byte character, would slice a `String` and panic.
        if shared > previous.len() || !previous.is_char_boundary(shared) {
            return Err(FormatError::InvalidSharedPrefix {
                shared,
                available: previous.len(),
            });
        }
        let suffix = cursor.read_str()?;
        let mut term = String::with_capacity(shared + suffix.len());
        term.push_str(&previous[..shared]);
        term.push_str(suffix);
        Ok(term)
    }

    /// The id of `term`, or `None` if it is not in the dictionary.
    ///
    /// Binary searches the block index on each block's first term — which is
    /// stored whole — then scans within the one block that can contain `term`.
    pub fn term_id(&self, term: &str) -> Result<Option<u32>, FormatError> {
        let block_count = self.header.term_block_count();
        if block_count == 0 {
            return Ok(None);
        }

        // Find the last block whose first term is <= `term`. That is the only
        // block that can contain it, since the dictionary is sorted.
        let mut low = 0usize;
        let mut high = block_count;
        while low < high {
            let mid = low + (high - low) / 2;
            let first = self.block_cursor(mid)?.read_str()?;
            if first <= term {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        if low == 0 {
            return Ok(None);
        }
        let block = low - 1;

        let mut cursor = self.block_cursor(block)?;
        let mut current = cursor.read_str()?.to_string();
        let block_start = block * TERMS_PER_BLOCK;
        if current == term {
            return Ok(Some(block_start as u32));
        }
        let block_len = (self.header.term_count as usize - block_start).min(TERMS_PER_BLOCK);
        for within in 1..block_len {
            current = self.next_in_block(&mut cursor, &current)?;
            match current.as_str().cmp(term) {
                std::cmp::Ordering::Less => continue,
                std::cmp::Ordering::Equal => return Ok(Some((block_start + within) as u32)),
                // Sorted order means nothing later in the block can match.
                std::cmp::Ordering::Greater => return Ok(None),
            }
        }
        Ok(None)
    }

    /// Every term in the dictionary, in sorted order.
    ///
    /// Decodes the whole dictionary, so this is for building an in-memory
    /// structure (a [`crate::token_set::TokenSet`] for wildcard expansion) or
    /// for diagnostics — not for serving a query.
    pub fn terms(&self) -> Result<Vec<String>, FormatError> {
        let mut out = Vec::with_capacity(self.header.term_count as usize);
        for block in 0..self.header.term_block_count() {
            let mut cursor = self.block_cursor(block)?;
            let mut current = cursor.read_str()?.to_string();
            let block_start = block * TERMS_PER_BLOCK;
            let block_len = (self.header.term_count as usize - block_start).min(TERMS_PER_BLOCK);
            out.push(current.clone());
            for _ in 1..block_len {
                current = self.next_in_block(&mut cursor, &current)?;
                out.push(current.clone());
            }
        }
        Ok(out)
    }

    /// Decode the posting list for `term_id`.
    pub fn postings(&self, term_id: u32) -> Result<TermPostings, FormatError> {
        if term_id >= self.header.term_count {
            return Err(FormatError::InvalidTermId(term_id));
        }
        let terms = self.terms_section()?;
        let table = (self.header.term_block_count() + 1) * 4;
        let start = read_u32(terms, table + term_id as usize * 4, "terms")? as usize;
        let end = read_u32(terms, table + (term_id as usize + 1) * 4, "terms")? as usize;
        if end < start {
            return Err(FormatError::Truncated {
                section: "postings",
            });
        }

        let postings = self.postings_section()?;
        let slice = postings.get(start..end).ok_or(FormatError::Truncated {
            section: "postings",
        })?;
        let mut cursor = Cursor::new(slice, "postings");

        let positions_base = cursor.read_u32()?;
        let document_frequency = cursor.read_u32()?;
        let field_count = cursor.read_usize()?;

        let mut fields = Vec::with_capacity(field_count);
        let mut positions_offset = positions_base;
        for _ in 0..field_count {
            let field_id = cursor.read_u32()?;
            if field_id >= self.header.field_count {
                return Err(FormatError::InvalidFieldId(field_id));
            }
            let entry_count = cursor.read_usize()?;
            let mut entries = Vec::with_capacity(entry_count);
            let mut doc_id = 0u32;
            for _ in 0..entry_count {
                doc_id = doc_id
                    .checked_add(cursor.read_u32()?)
                    .ok_or(FormatError::Truncated {
                        section: "postings",
                    })?;
                if doc_id >= self.header.doc_count {
                    return Err(FormatError::InvalidDocId(doc_id));
                }
                let term_frequency = cursor.read_u32()?;
                let position_count = cursor.read_u32()?;
                entries.push(PostingEntry {
                    doc_id,
                    term_frequency,
                    positions_offset,
                    position_count,
                });
                // Position blocks are laid out in the same order they are
                // referenced, so each entry's block starts where the previous
                // one ended. Advancing here avoids storing an offset per entry.
                positions_offset = positions_offset
                    .checked_add(self.position_block_len(positions_offset, position_count)?)
                    .ok_or(FormatError::Truncated {
                        section: "positions",
                    })?;
            }
            fields.push(FieldPostings { field_id, entries });
        }
        Ok(TermPostings {
            document_frequency,
            fields,
        })
    }

    /// Byte length of the position block at `offset` holding `count` positions.
    fn position_block_len(&self, offset: u32, count: u32) -> Result<u32, FormatError> {
        if count == 0 {
            return Ok(0);
        }
        let positions = self.positions_section()?;
        let slice = positions
            .get(offset as usize..)
            .ok_or(FormatError::Truncated {
                section: "positions",
            })?;
        let mut cursor = Cursor::new(slice, "positions");
        let uniform_length = cursor.read_usize()?;
        let per_position = if uniform_length == 0 { 2 } else { 1 };
        for _ in 0..count as usize * per_position {
            cursor.read_varint()?;
        }
        u32::try_from(cursor.position()).map_err(|_| FormatError::Truncated {
            section: "positions",
        })
    }

    /// Decode the positions of one posting entry, as `(start, length)` pairs.
    ///
    /// Returns an empty vector when the index was built without positions.
    pub fn positions(&self, entry: &PostingEntry) -> Result<Vec<(usize, usize)>, FormatError> {
        if entry.position_count == 0 {
            return Ok(Vec::new());
        }
        if !self.header.has_positions() {
            return Err(FormatError::PositionsUnavailable);
        }
        let positions = self.positions_section()?;
        let slice =
            positions
                .get(entry.positions_offset as usize..)
                .ok_or(FormatError::Truncated {
                    section: "positions",
                })?;
        let mut cursor = Cursor::new(slice, "positions");
        let uniform_length = cursor.read_usize()?;

        let mut out = Vec::with_capacity(entry.position_count as usize);
        let mut start = 0usize;
        for _ in 0..entry.position_count {
            start = start
                .checked_add(cursor.read_usize()?)
                .ok_or(FormatError::Truncated {
                    section: "positions",
                })?;
            let length = if uniform_length == 0 {
                cursor.read_usize()?
            } else {
                uniform_length
            };
            out.push((start, length));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::super::writer::{write_index, IndexSnapshot};
    use super::*;
    use crate::index::{FieldRef, Posting, PostingDoc};
    use std::collections::{BTreeMap, HashMap};

    struct Fixture {
        fields: Vec<String>,
        field_boosts: HashMap<String, f64>,
        doc_boosts: HashMap<String, f64>,
        field_lengths: HashMap<FieldRef, usize>,
        inverted_index: BTreeMap<String, Posting>,
    }

    impl Fixture {
        fn snapshot(&self) -> IndexSnapshot<'_> {
            IndexSnapshot {
                language: "ja",
                fields: &self.fields,
                field_boosts: &self.field_boosts,
                pipeline: vec!["trimmer".to_string()],
                document_count: 3,
                doc_boosts: &self.doc_boosts,
                field_lengths: &self.field_lengths,
                inverted_index: &self.inverted_index,
                k1: 1.2,
                b: 0.75,
                include_positions: true,
            }
        }
    }

    /// A fixture with enough terms to span several dictionary blocks, a docref
    /// containing slashes, and multi-byte terms sharing prefixes.
    fn fixture() -> Fixture {
        let fields = vec!["title".to_string(), "body".to_string()];
        let field_boosts = [("title".to_string(), 10.0), ("body".to_string(), 1.0)]
            .into_iter()
            .collect();
        let docs = ["guide/a/index.html", "guide/b/index.html", "zzz-last"];
        let doc_boosts = docs
            .iter()
            .enumerate()
            .map(|(i, d)| (d.to_string(), 1.0 + i as f64))
            .collect();
        let mut field_lengths = HashMap::new();
        for (i, doc) in docs.iter().enumerate() {
            field_lengths.insert(FieldRef::new(*doc, "title"), 3 + i);
            field_lengths.insert(FieldRef::new(*doc, "body"), 40 + i * 7);
        }

        // 40 terms forces three blocks at TERMS_PER_BLOCK = 16.
        let mut inverted_index = BTreeMap::new();
        for i in 0..40u32 {
            let term = format!("検索{i:03}");
            let mut posting = Posting::default();
            let doc = docs[(i % 3) as usize];
            posting
                .fields
                .entry("body".to_string())
                .or_default()
                .insert(
                    doc.to_string(),
                    PostingDoc {
                        term_frequency: (i % 4) + 1,
                        positions: (0..=(i % 4)).map(|p| (p as usize * 3, 2)).collect(),
                    },
                );
            if i % 5 == 0 {
                posting
                    .fields
                    .entry("title".to_string())
                    .or_default()
                    .insert(
                        doc.to_string(),
                        PostingDoc {
                            term_frequency: 1,
                            positions: vec![(0, 2)],
                        },
                    );
            }
            inverted_index.insert(term, posting);
        }
        Fixture {
            fields,
            field_boosts,
            doc_boosts,
            field_lengths,
            inverted_index,
        }
    }

    #[test]
    fn header_and_metadata_survive_a_roundtrip() {
        let fixture = fixture();
        let bytes = write_index(&fixture.snapshot());
        let index = BinaryIndex::open(&bytes).unwrap();

        assert_eq!(index.language(), "ja");
        assert_eq!(index.fields(), ["title", "body"]);
        assert_eq!(index.field_boost(0), Ok(10.0));
        assert_eq!(index.field_boost(1), Ok(1.0));
        assert_eq!(index.pipeline(), ["trimmer"]);
        assert_eq!(index.document_count(), 3);
        assert_eq!(index.doc_count(), 3);
        assert_eq!(index.term_count(), 40);
        assert_eq!(index.k1(), 1.2);
        assert_eq!(index.b(), 0.75);
        assert!(index.header().has_positions());
    }

    #[test]
    fn document_references_roundtrip_including_slashes() {
        let fixture = fixture();
        let bytes = write_index(&fixture.snapshot());
        let index = BinaryIndex::open(&bytes).unwrap();

        // Sorted order, and a docref full of slashes must come back intact —
        // this is exactly what the JSON FieldRef parsing got wrong.
        assert_eq!(index.doc_ref(0), Ok("guide/a/index.html"));
        assert_eq!(index.doc_ref(1), Ok("guide/b/index.html"));
        assert_eq!(index.doc_ref(2), Ok("zzz-last"));
        assert_eq!(index.doc_ref(3), Err(FormatError::InvalidDocId(3)));

        for id in 0..3 {
            let reference = index.doc_ref(id).unwrap();
            assert_eq!(index.doc_id(reference), Ok(Some(id)));
        }
        assert_eq!(index.doc_id("missing"), Ok(None));
        // Before the first and after the last, to exercise both search bounds.
        assert_eq!(index.doc_id("aaa"), Ok(None));
        assert_eq!(index.doc_id("zzzzzz"), Ok(None));
    }

    #[test]
    fn boosts_and_field_lengths_roundtrip() {
        let fixture = fixture();
        let bytes = write_index(&fixture.snapshot());
        let index = BinaryIndex::open(&bytes).unwrap();

        for id in 0..3u32 {
            assert_eq!(index.doc_boost(id), Ok(1.0 + f64::from(id)));
            assert_eq!(index.field_length(id, 0), Ok(3 + id));
            assert_eq!(index.field_length(id, 1), Ok(40 + id * 7));
        }
        assert_eq!(
            index.field_length(0, 2),
            Err(FormatError::InvalidFieldId(2))
        );
        assert_eq!(index.field_length(9, 0), Err(FormatError::InvalidDocId(9)));

        // (40 + 47 + 54) / 3
        assert_eq!(index.average_field_length(1), Ok(47.0));
    }

    #[test]
    fn every_term_roundtrips_across_block_boundaries() {
        let fixture = fixture();
        let bytes = write_index(&fixture.snapshot());
        let index = BinaryIndex::open(&bytes).unwrap();

        let expected: Vec<String> = fixture.inverted_index.keys().cloned().collect();
        assert_eq!(index.terms().unwrap(), expected);

        // Individually too, which exercises the per-block prefix walk rather
        // than the sequential decode.
        for (id, term) in expected.iter().enumerate() {
            assert_eq!(&index.term(id as u32).unwrap(), term, "term id {id}");
            assert_eq!(index.term_id(term), Ok(Some(id as u32)), "lookup {term}");
        }
        assert_eq!(index.term(40), Err(FormatError::InvalidTermId(40)));
    }

    #[test]
    fn term_lookup_rejects_absent_terms() {
        let fixture = fixture();
        let bytes = write_index(&fixture.snapshot());
        let index = BinaryIndex::open(&bytes).unwrap();

        // Before the first term, after the last, and in a gap between two.
        assert_eq!(index.term_id("あ"), Ok(None));
        assert_eq!(index.term_id("検索999"), Ok(None));
        assert_eq!(index.term_id("検索0005"), Ok(None));
        assert_eq!(index.term_id(""), Ok(None));
    }

    #[test]
    fn postings_and_positions_roundtrip() {
        let fixture = fixture();
        let bytes = write_index(&fixture.snapshot());
        let index = BinaryIndex::open(&bytes).unwrap();

        for (term_id, (term, posting)) in fixture.inverted_index.iter().enumerate() {
            let decoded = index.postings(term_id as u32).unwrap();
            assert_eq!(
                decoded.document_frequency as usize,
                posting.document_frequency(),
                "df for {term}"
            );

            for field_postings in &decoded.fields {
                let field_name = index.field_name(field_postings.field_id).unwrap();
                let expected = &posting.fields[field_name];
                assert_eq!(field_postings.entries.len(), expected.len());
                for entry in &field_postings.entries {
                    let doc_ref = index.doc_ref(entry.doc_id).unwrap();
                    let expected_doc = &expected[doc_ref];
                    assert_eq!(entry.term_frequency, expected_doc.term_frequency);
                    assert_eq!(
                        index.positions(entry).unwrap(),
                        expected_doc.positions,
                        "positions for {term} / {field_name} / {doc_ref}"
                    );
                }
            }
        }
    }

    #[test]
    fn posting_entries_ascend_by_document_id() {
        // Delta decoding depends on it, and a searcher can binary search only if
        // it holds.
        let fixture = fixture();
        let bytes = write_index(&fixture.snapshot());
        let index = BinaryIndex::open(&bytes).unwrap();

        for term_id in 0..index.term_count() as u32 {
            let postings = index.postings(term_id).unwrap();
            let mut previous_field = None;
            for field_postings in &postings.fields {
                if let Some(previous) = previous_field {
                    assert!(field_postings.field_id > previous, "fields not ascending");
                }
                previous_field = Some(field_postings.field_id);
                for pair in field_postings.entries.windows(2) {
                    assert!(pair[0].doc_id < pair[1].doc_id, "docs not ascending");
                }
            }
        }
    }

    #[test]
    fn positions_are_unavailable_when_omitted() {
        let fixture = fixture();
        let mut snapshot = fixture.snapshot();
        snapshot.include_positions = false;
        let bytes = write_index(&snapshot);
        let index = BinaryIndex::open(&bytes).unwrap();

        assert!(!index.header().has_positions());
        // Postings must still decode and score; only positions are gone.
        let postings = index.postings(0).unwrap();
        let entry = &postings.fields[0].entries[0];
        assert!(entry.term_frequency > 0);
        assert_eq!(index.positions(entry), Ok(Vec::new()));
    }

    #[test]
    fn empty_index_opens_and_reports_nothing() {
        let fields: Vec<String> = vec!["body".to_string()];
        let field_boosts = HashMap::new();
        let doc_boosts = HashMap::new();
        let field_lengths = HashMap::new();
        let inverted_index = BTreeMap::new();
        let snapshot = IndexSnapshot {
            language: "en",
            fields: &fields,
            field_boosts: &field_boosts,
            pipeline: Vec::new(),
            document_count: 0,
            doc_boosts: &doc_boosts,
            field_lengths: &field_lengths,
            inverted_index: &inverted_index,
            k1: 1.2,
            b: 0.75,
            include_positions: true,
        };
        let bytes = write_index(&snapshot);
        let index = BinaryIndex::open(&bytes).unwrap();

        assert_eq!(index.term_count(), 0);
        assert_eq!(index.doc_count(), 0);
        assert_eq!(index.term_id("anything"), Ok(None));
        assert_eq!(index.terms(), Ok(Vec::new()));
        assert_eq!(index.average_field_length(0), Ok(0.0));
    }

    #[test]
    fn open_rejects_a_short_buffer() {
        assert_eq!(
            BinaryIndex::open(&[]).unwrap_err(),
            FormatError::TooShort {
                needed: HEADER_LEN,
                got: 0
            }
        );
        assert!(BinaryIndex::open(&[0u8; 63]).is_err());
    }

    #[test]
    fn open_rejects_bad_magic_and_version() {
        let fixture = fixture();
        let bytes = write_index(&fixture.snapshot());

        let mut wrong_magic = bytes.clone();
        wrong_magic[0] = b'J';
        assert!(matches!(
            BinaryIndex::open(&wrong_magic),
            Err(FormatError::BadMagic(_))
        ));

        let mut wrong_version = bytes.clone();
        wrong_version[4] = 99;
        assert!(matches!(
            BinaryIndex::open(&wrong_version),
            Err(FormatError::UnsupportedVersion { .. })
        ));
    }

    #[test]
    fn open_rejects_offsets_outside_the_buffer() {
        let fixture = fixture();
        let mut bytes = write_index(&fixture.snapshot());
        // Push end_offset past the real length.
        let bogus = (bytes.len() as u32 + 4096).to_le_bytes();
        bytes[60..64].copy_from_slice(&bogus);
        assert!(matches!(
            BinaryIndex::open(&bytes),
            Err(FormatError::SectionOutOfBounds { .. })
        ));
    }

    #[test]
    fn open_rejects_sections_out_of_order() {
        let fixture = fixture();
        let mut bytes = write_index(&fixture.snapshot());
        // Point docs_offset before meta_offset.
        bytes[44..48].copy_from_slice(&0u32.to_le_bytes());
        assert!(matches!(
            BinaryIndex::open(&bytes),
            Err(FormatError::SectionsNotOrdered { .. })
        ));
    }

    #[test]
    fn truncating_the_buffer_never_panics() {
        // The property that matters for reading a partial download: every prefix
        // either opens and reads, or errors. Nothing panics, nothing reads out
        // of bounds.
        let fixture = fixture();
        let bytes = write_index(&fixture.snapshot());

        for length in 0..bytes.len() {
            let truncated = &bytes[..length];
            let Ok(index) = BinaryIndex::open(truncated) else {
                continue;
            };
            // If it opened, exercise every accessor and require Ok or Err.
            for id in 0..index.doc_count() as u32 + 1 {
                let _ = index.doc_ref(id);
                let _ = index.doc_boost(id);
                let _ = index.field_length(id, 0);
            }
            let _ = index.terms();
            for id in 0..index.term_count() as u32 + 1 {
                let _ = index.term(id);
                if let Ok(postings) = index.postings(id) {
                    for field_postings in &postings.fields {
                        for entry in &field_postings.entries {
                            let _ = index.positions(entry);
                        }
                    }
                }
            }
            let _ = index.term_id("検索000");
            let _ = index.average_field_length(0);
        }
    }

    #[test]
    fn corrupting_any_single_byte_never_panics() {
        // Bit-rot and hostile input both look like this. Every accessor must
        // return a result rather than unwinding.
        let fixture = fixture();
        let bytes = write_index(&fixture.snapshot());

        // Every 7th byte, so the test stays fast while still hitting each
        // section including the header.
        for i in (0..bytes.len()).step_by(7) {
            for xor in [0x01u8, 0x80, 0xff] {
                let mut corrupt = bytes.clone();
                corrupt[i] ^= xor;
                let Ok(index) = BinaryIndex::open(&corrupt) else {
                    continue;
                };
                for id in 0..index.doc_count().min(64) as u32 {
                    let _ = index.doc_ref(id);
                    let _ = index.doc_boost(id);
                    let _ = index.field_length(id, 0);
                }
                let _ = index.terms();
                for id in 0..index.term_count().min(64) as u32 {
                    let _ = index.term(id);
                    if let Ok(postings) = index.postings(id) {
                        for field_postings in &postings.fields {
                            for entry in &field_postings.entries {
                                let _ = index.positions(entry);
                            }
                        }
                    }
                }
                let _ = index.term_id("検索000");
                let _ = index.doc_id("guide/a/index.html");
            }
        }
    }
}
