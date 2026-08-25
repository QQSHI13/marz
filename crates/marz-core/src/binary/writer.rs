//! Serializing an index to the binary format.
//!
//! The writer builds each section into its own buffer, then concatenates them
//! and stamps the header with the resulting offsets. Two things make that
//! possible in a single pass:
//!
//! * Every intra-file pointer is **relative to its own section**, so the
//!   postings section can reference the positions section before either one's
//!   final absolute offset is known. Absolute offsets would be circular —
//!   positions come after postings, so the positions base depends on how long
//!   the postings are, which depends on the offsets being written into them.
//! * Sections are ordered coarse-to-fine, so nothing earlier depends on
//!   anything later.
//!
//! See [`super`] for the layout this produces.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::varint::{write_str, write_varint};
use super::{FLAG_HAS_POSITIONS, FORMAT_VERSION, HEADER_LEN, MAGIC, TERMS_PER_BLOCK};
use crate::index::{FieldRef, Posting};

/// Everything the writer needs from a built index.
///
/// A borrowed snapshot rather than `&Index` so the writer does not need access
/// to `Index`'s private fields, and so it can be driven directly from a test
/// fixture.
pub struct IndexSnapshot<'a> {
    /// Language code, e.g. `"ja"`.
    pub language: &'a str,
    /// Indexed field names, in configured order. A field's index here is its id.
    pub fields: &'a [String],
    /// Per-field boosts. Missing entries default to `1.0`.
    pub field_boosts: &'a HashMap<String, f64>,
    /// Pipeline stage labels, recorded for diagnostics.
    pub pipeline: Vec<String>,
    /// BM25's `N`: the number of documents added to the index.
    pub document_count: usize,
    /// Per-document boosts. Missing entries default to `1.0`.
    pub doc_boosts: &'a HashMap<String, f64>,
    /// Token count per document field.
    pub field_lengths: &'a HashMap<FieldRef, usize>,
    /// The inverted index, keyed by term in sorted order.
    pub inverted_index: &'a BTreeMap<String, Posting>,
    /// BM25 term-frequency saturation parameter.
    pub k1: f64,
    /// BM25 length-normalization parameter.
    pub b: f64,
    /// Whether to emit the positions section.
    ///
    /// Dropping it saves roughly a tenth of the file at the cost of
    /// highlighting and CJK phrase verification.
    pub include_positions: bool,
}

/// Serialize an index to the binary format.
pub fn write_index(snapshot: &IndexSnapshot<'_>) -> Vec<u8> {
    let doc_refs = collect_doc_refs(snapshot);
    let doc_ids: HashMap<&str, u32> = doc_refs
        .iter()
        .enumerate()
        .map(|(i, r)| (*r, i as u32))
        .collect();
    let field_ids: HashMap<&str, u32> = snapshot
        .fields
        .iter()
        .enumerate()
        .map(|(i, f)| (f.as_str(), i as u32))
        .collect();

    let terms: Vec<&String> = snapshot.inverted_index.keys().collect();

    let meta = build_meta(snapshot);
    let docs = build_docs(snapshot, &doc_refs, &field_ids);
    let (postings, positions, postings_offsets) =
        build_postings(snapshot, &doc_ids, &field_ids, &terms);
    let terms_section = build_terms(&terms, &postings_offsets);

    // Sections are laid out in the order the header declares them.
    let meta_offset = HEADER_LEN;
    let docs_offset = meta_offset + meta.len();
    let terms_offset = docs_offset + docs.len();
    let postings_offset = terms_offset + terms_section.len();
    let positions_offset = postings_offset + postings.len();
    let end_offset = positions_offset + positions.len();

    let mut out = Vec::with_capacity(end_offset);
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    let flags = if snapshot.include_positions {
        FLAG_HAS_POSITIONS
    } else {
        0
    };
    out.extend_from_slice(&flags.to_le_bytes());
    out.extend_from_slice(&(doc_refs.len() as u32).to_le_bytes());
    out.extend_from_slice(&(snapshot.document_count as u32).to_le_bytes());
    out.extend_from_slice(&(snapshot.fields.len() as u32).to_le_bytes());
    out.extend_from_slice(&(terms.len() as u32).to_le_bytes());
    out.extend_from_slice(&snapshot.k1.to_le_bytes());
    out.extend_from_slice(&snapshot.b.to_le_bytes());
    for offset in [
        meta_offset,
        docs_offset,
        terms_offset,
        postings_offset,
        positions_offset,
        end_offset,
    ] {
        out.extend_from_slice(&(offset as u32).to_le_bytes());
    }
    debug_assert_eq!(out.len(), HEADER_LEN, "header must be exactly HEADER_LEN");

    out.extend_from_slice(&meta);
    out.extend_from_slice(&docs);
    out.extend_from_slice(&terms_section);
    out.extend_from_slice(&postings);
    out.extend_from_slice(&positions);
    out
}

/// Every document reference in the index, sorted.
///
/// Sorting is what makes a posting list an ascending run of ids — which is what
/// delta coding needs — and lets a reader find a reference by binary search.
///
/// References are gathered from field lengths and boosts rather than from the
/// postings, so a document whose every field was empty still gets an id and can
/// still be returned by a purely negated query.
fn collect_doc_refs<'a>(snapshot: &IndexSnapshot<'a>) -> Vec<&'a str> {
    let mut refs: BTreeSet<&str> = BTreeSet::new();
    for field_ref in snapshot.field_lengths.keys() {
        refs.insert(field_ref.doc_ref());
    }
    for doc_ref in snapshot.doc_boosts.keys() {
        refs.insert(doc_ref.as_str());
    }
    refs.into_iter().collect()
}

/// Language, field names, field boosts and pipeline labels.
fn build_meta(snapshot: &IndexSnapshot<'_>) -> Vec<u8> {
    let mut out = Vec::new();
    write_str(&mut out, snapshot.language);
    for field in snapshot.fields {
        write_str(&mut out, field);
    }
    for field in snapshot.fields {
        let boost = snapshot.field_boosts.get(field).copied().unwrap_or(1.0);
        out.extend_from_slice(&boost.to_le_bytes());
    }
    write_varint(&mut out, snapshot.pipeline.len() as u64);
    for label in &snapshot.pipeline {
        write_str(&mut out, label);
    }
    out
}

/// Document reference heap, boosts, and the field-length matrix.
///
/// The fixed-width parts come first so a reader can address them arithmetically
/// from the section base; the variable-length heap goes last.
///
/// Boosts are a full `f64` per document even though they are almost always
/// `1.0`. A "uniform boosts" flag would save eight bytes per document, but that
/// is well under a percent of a real index and it would put a conditional in the
/// middle of the offset arithmetic — which is precisely where format bugs go
/// unnoticed.
fn build_docs(
    snapshot: &IndexSnapshot<'_>,
    doc_refs: &[&str],
    field_ids: &HashMap<&str, u32>,
) -> Vec<u8> {
    let field_count = snapshot.fields.len();
    let mut out = Vec::new();

    // Offset table: one entry per reference plus a terminator, so entry `i`'s
    // length is `table[i + 1] - table[i]` with no special case for the last.
    let mut heap = Vec::new();
    let mut table: Vec<u32> = Vec::with_capacity(doc_refs.len() + 1);
    for doc_ref in doc_refs {
        table.push(heap.len() as u32);
        heap.extend_from_slice(doc_ref.as_bytes());
    }
    table.push(heap.len() as u32);
    for offset in &table {
        out.extend_from_slice(&offset.to_le_bytes());
    }

    for doc_ref in doc_refs {
        let boost = snapshot.doc_boosts.get(*doc_ref).copied().unwrap_or(1.0);
        out.extend_from_slice(&boost.to_le_bytes());
    }

    // Row-major matrix indexed by `doc_id * field_count + field_id`. Scoring
    // needs random access to a single cell, so this stays fixed-width; varints
    // would be smaller but would force a scan from the start of the matrix.
    let mut lengths = vec![0u32; doc_refs.len() * field_count];
    for (field_ref, length) in snapshot.field_lengths {
        let Some(doc_id) = doc_refs.binary_search(&field_ref.doc_ref()).ok() else {
            continue;
        };
        let Some(field_id) = field_ids.get(field_ref.field_name()) else {
            continue;
        };
        lengths[doc_id * field_count + *field_id as usize] = *length as u32;
    }
    for length in &lengths {
        out.extend_from_slice(&length.to_le_bytes());
    }

    out.extend_from_slice(&heap);
    out
}

/// Postings and positions, plus each term's relative postings offset.
///
/// Returns `(postings, positions, offsets)` where `offsets` has `term_count + 1`
/// entries so a term's postings extent is `offsets[id]..offsets[id + 1]`.
fn build_postings(
    snapshot: &IndexSnapshot<'_>,
    doc_ids: &HashMap<&str, u32>,
    field_ids: &HashMap<&str, u32>,
    terms: &[&String],
) -> (Vec<u8>, Vec<u8>, Vec<u32>) {
    let mut postings = Vec::new();
    let mut positions = Vec::new();
    let mut offsets: Vec<u32> = Vec::with_capacity(terms.len() + 1);

    for term in terms {
        offsets.push(postings.len() as u32);
        let posting = &snapshot.inverted_index[*term];

        // Order the term's fields by id, and each field's documents by id, so
        // both are ascending runs that delta-code well and can be binary
        // searched by a reader that wants one document.
        let mut by_field: Vec<(u32, Vec<(u32, &crate::index::PostingDoc)>)> = Vec::new();
        for (field_name, docs) in &posting.fields {
            let Some(&field_id) = field_ids.get(field_name.as_str()) else {
                continue;
            };
            let mut entries: Vec<(u32, &crate::index::PostingDoc)> = docs
                .iter()
                .filter_map(|(doc_ref, posting_doc)| {
                    doc_ids.get(doc_ref.as_str()).map(|id| (*id, posting_doc))
                })
                .collect();
            if entries.is_empty() {
                continue;
            }
            entries.sort_unstable_by_key(|(id, _)| *id);
            by_field.push((field_id, entries));
        }
        by_field.sort_unstable_by_key(|(id, _)| *id);

        // Document frequency counts distinct documents across all fields, so a
        // document holding the term in both title and body counts once. Stored
        // rather than derived: scoring needs it for every matched term, and
        // recomputing it would mean walking the whole posting list to build a
        // set on every query.
        let mut distinct: BTreeSet<u32> = BTreeSet::new();
        for (_, entries) in &by_field {
            distinct.extend(entries.iter().map(|(id, _)| *id));
        }

        // Relative to the positions section base, not to the file, so this can
        // be written before the section's absolute offset is known.
        write_varint(&mut postings, positions.len() as u64);
        write_varint(&mut postings, distinct.len() as u64);
        write_varint(&mut postings, by_field.len() as u64);

        for (field_id, entries) in &by_field {
            write_varint(&mut postings, u64::from(*field_id));
            write_varint(&mut postings, entries.len() as u64);
            let mut previous_doc = 0u32;
            for (doc_id, posting_doc) in entries {
                write_varint(&mut postings, u64::from(doc_id - previous_doc));
                previous_doc = *doc_id;
                write_varint(&mut postings, u64::from(posting_doc.term_frequency));

                if snapshot.include_positions {
                    write_varint(&mut postings, posting_doc.positions.len() as u64);
                    write_position_block(&mut positions, &posting_doc.positions);
                } else {
                    write_varint(&mut postings, 0);
                }
            }
        }
    }
    offsets.push(postings.len() as u32);

    (postings, positions, offsets)
}

/// Append one posting's positions.
///
/// The block is self-delimiting given the position count from the posting entry,
/// so it carries no length of its own.
///
/// Token lengths are usually all the same within a block — every CJK bigram is
/// two characters — so a uniform length is written once and only start offsets
/// follow. A length of zero is impossible for a real token, which makes it a
/// free sentinel for "lengths vary, they are interleaved". This roughly halves
/// the positions section on CJK text.
fn write_position_block(out: &mut Vec<u8>, positions: &[(usize, usize)]) {
    if positions.is_empty() {
        return;
    }
    let first_length = positions[0].1;
    let uniform = first_length != 0 && positions.iter().all(|(_, len)| *len == first_length);

    if uniform {
        write_varint(out, first_length as u64);
        let mut previous = 0usize;
        for (start, _) in positions {
            write_varint(out, start.saturating_sub(previous) as u64);
            previous = *start;
        }
    } else {
        write_varint(out, 0);
        let mut previous = 0usize;
        for (start, length) in positions {
            write_varint(out, start.saturating_sub(previous) as u64);
            write_varint(out, *length as u64);
            previous = *start;
        }
    }
}

/// Front-coded term dictionary, its block index, and the postings offset table.
fn build_terms(terms: &[&String], postings_offsets: &[u32]) -> Vec<u8> {
    let block_count = terms.len().div_ceil(TERMS_PER_BLOCK);

    let mut dictionary = Vec::new();
    let mut block_offsets: Vec<u32> = Vec::with_capacity(block_count + 1);
    for block in terms.chunks(TERMS_PER_BLOCK) {
        block_offsets.push(dictionary.len() as u32);
        // The first term of each block is stored whole. That is what lets a
        // lookup binary search to a block and start decoding there instead of
        // from the beginning of the dictionary.
        write_str(&mut dictionary, block[0]);
        let mut previous: &str = block[0];
        for term in &block[1..] {
            let shared = shared_prefix_len(previous, term);
            write_varint(&mut dictionary, shared as u64);
            write_str(&mut dictionary, &term[shared..]);
            previous = term;
        }
    }
    block_offsets.push(dictionary.len() as u32);

    let mut out =
        Vec::with_capacity((block_offsets.len() + postings_offsets.len()) * 4 + dictionary.len());
    for offset in &block_offsets {
        out.extend_from_slice(&offset.to_le_bytes());
    }
    for offset in postings_offsets {
        out.extend_from_slice(&offset.to_le_bytes());
    }
    out.extend_from_slice(&dictionary);
    out
}

/// Length of the longest shared prefix of `a` and `b` that ends on a UTF-8
/// character boundary.
///
/// Truncating mid-character would make the stored suffix invalid UTF-8, so the
/// comparison walks characters and accumulates their byte lengths rather than
/// comparing bytes directly.
fn shared_prefix_len(a: &str, b: &str) -> usize {
    let mut shared = 0;
    for (ca, cb) in a.chars().zip(b.chars()) {
        if ca != cb {
            break;
        }
        shared += ca.len_utf8();
    }
    shared
}

/// Bytes the meta section will occupy.
///
/// This duplicates [`build_meta`]'s arithmetic on purpose: a test asserts the
/// two agree, which catches a field added to one and not the other. Section
/// offsets themselves come from real buffer lengths, so a mismatch here can
/// never corrupt a written index.
#[cfg(test)]
fn meta_len(snapshot: &IndexSnapshot<'_>) -> usize {
    use super::varint::{str_len, varint_len};
    str_len(snapshot.language)
        + snapshot.fields.iter().map(|f| str_len(f)).sum::<usize>()
        + snapshot.fields.len() * 8
        + varint_len(snapshot.pipeline.len() as u64)
        + snapshot.pipeline.iter().map(|l| str_len(l)).sum::<usize>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::PostingDoc;

    /// One `(field, doc, term_frequency, positions)` row of a posting.
    type PostingRow<'a> = (&'a str, &'a str, u32, &'a [(usize, usize)]);

    fn posting(entries: &[PostingRow<'_>]) -> Posting {
        let mut posting = Posting::default();
        for (field, doc, tf, positions) in entries {
            posting.fields.entry(field.to_string()).or_default().insert(
                doc.to_string(),
                PostingDoc {
                    term_frequency: *tf,
                    positions: positions.to_vec(),
                },
            );
        }
        posting
    }

    /// Owner of the data an [`IndexSnapshot`] borrows.
    ///
    /// The snapshot holds references, so the parts must outlive it; keeping them
    /// in one struct avoids threading five separate locals through every test.
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
                language: "en",
                fields: &self.fields,
                field_boosts: &self.field_boosts,
                pipeline: vec!["trimmer".to_string(), "stemmer".to_string()],
                document_count: 2,
                doc_boosts: &self.doc_boosts,
                field_lengths: &self.field_lengths,
                inverted_index: &self.inverted_index,
                k1: 1.2,
                b: 0.75,
                include_positions: true,
            }
        }
    }

    fn fixture() -> Fixture {
        let mut inverted_index = BTreeMap::new();
        inverted_index.insert(
            "green".to_string(),
            posting(&[
                ("title", "a", 1, &[(0, 5)]),
                ("body", "a", 2, &[(0, 5), (9, 5)]),
                ("body", "b", 1, &[(4, 5)]),
            ]),
        );
        inverted_index.insert("plumb".to_string(), posting(&[("body", "b", 1, &[(0, 5)])]));

        Fixture {
            fields: vec!["title".to_string(), "body".to_string()],
            field_boosts: [("title".to_string(), 10.0), ("body".to_string(), 1.0)]
                .into_iter()
                .collect(),
            doc_boosts: [("a".to_string(), 1.0), ("b".to_string(), 2.0)]
                .into_iter()
                .collect(),
            field_lengths: [
                (FieldRef::new("a", "title"), 3),
                (FieldRef::new("a", "body"), 12),
                (FieldRef::new("b", "title"), 4),
                (FieldRef::new("b", "body"), 20),
            ]
            .into_iter()
            .collect(),
            inverted_index,
        }
    }

    #[test]
    fn header_is_exactly_sixty_four_bytes_and_offsets_are_ordered() {
        let fixture = fixture();
        let snapshot = fixture.snapshot();
        let bytes = write_index(&snapshot);

        assert_eq!(&bytes[0..4], b"MARZ");
        // Section offsets must ascend and the last must equal the file length,
        // or a reader's bounds checks are meaningless.
        let offsets: Vec<u32> = (40..64)
            .step_by(4)
            .map(|off| u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()))
            .collect();
        assert_eq!(offsets[0], HEADER_LEN as u32);
        for pair in offsets.windows(2) {
            assert!(pair[0] <= pair[1], "offsets not ascending: {offsets:?}");
        }
        assert_eq!(*offsets.last().unwrap() as usize, bytes.len());
    }

    #[test]
    fn meta_section_length_matches_its_computed_size() {
        // The layout arithmetic and the writer must agree, or every following
        // section is misplaced.
        let fixture = fixture();
        let snapshot = fixture.snapshot();
        let meta = build_meta(&snapshot);
        assert_eq!(meta.len(), meta_len(&snapshot));
    }

    #[test]
    fn doc_refs_are_collected_from_lengths_and_boosts() {
        // A document with no indexed terms must still get an id.
        let mut fixture = fixture();
        fixture.doc_boosts = [("c".to_string(), 1.0)].into_iter().collect();
        let snapshot = fixture.snapshot();
        assert_eq!(collect_doc_refs(&snapshot), ["a", "b", "c"]);
    }

    #[test]
    fn shared_prefix_stops_on_a_character_boundary() {
        // Byte-wise comparison would report 3 shared bytes for these two, since
        // 検 and 験 share their first two UTF-8 bytes. Splitting there would
        // store an invalid suffix.
        assert_eq!(shared_prefix_len("検索", "験索"), 0);
        assert_eq!(shared_prefix_len("検索", "検討"), "検".len());
        assert_eq!(shared_prefix_len("running", "runner"), 4);
        assert_eq!(shared_prefix_len("", "abc"), 0);
        assert_eq!(shared_prefix_len("abc", "abc"), 3);
    }

    #[test]
    fn uniform_position_lengths_cost_one_byte_each() {
        // The CJK case: every bigram is two characters long, so the length is
        // written once for the block.
        let positions: Vec<(usize, usize)> = (0..10).map(|i| (i, 2)).collect();
        let mut out = Vec::new();
        write_position_block(&mut out, &positions);
        assert_eq!(out.len(), 1 + 10, "uniform length + one delta per position");
    }

    #[test]
    fn varying_position_lengths_interleave() {
        let mut out = Vec::new();
        write_position_block(&mut out, &[(0, 3), (5, 7)]);
        assert_eq!(out.len(), 1 + 2 + 2, "sentinel + two (delta, length) pairs");
        assert_eq!(out[0], 0, "sentinel must be zero");
    }

    #[test]
    fn empty_position_block_writes_nothing() {
        let mut out = Vec::new();
        write_position_block(&mut out, &[]);
        assert!(out.is_empty());
    }

    #[test]
    fn omitting_positions_shrinks_the_file() {
        let fixture = fixture();
        let mut snapshot = fixture.snapshot();
        let with = write_index(&snapshot).len();
        snapshot.include_positions = false;
        let without = write_index(&snapshot).len();
        assert!(without < with, "{without} should be under {with}");
    }

    #[test]
    fn postings_offsets_have_a_terminator() {
        let fixture = fixture();
        let snapshot = fixture.snapshot();
        let doc_refs = collect_doc_refs(&snapshot);
        let doc_ids: HashMap<&str, u32> = doc_refs
            .iter()
            .enumerate()
            .map(|(i, r)| (*r, i as u32))
            .collect();
        let field_ids: HashMap<&str, u32> = fixture
            .fields
            .iter()
            .enumerate()
            .map(|(i, name)| (name.as_str(), i as u32))
            .collect();
        let terms: Vec<&String> = fixture.inverted_index.keys().collect();
        let (postings, _, offsets) = build_postings(&snapshot, &doc_ids, &field_ids, &terms);

        assert_eq!(offsets.len(), terms.len() + 1);
        assert_eq!(*offsets.last().unwrap() as usize, postings.len());
    }
}
