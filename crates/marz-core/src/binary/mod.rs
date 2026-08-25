//! Binary index format: a zero-copy, memory-mappable on-disk representation.
//!
//! # Why replace JSON
//!
//! Measured on real Wikipedia corpora, the JSON index spent its bytes like
//! this (Japanese, 2.89 MB; Chinese and Korean are within a few points):
//!
//! | part | share |
//! |---|---|
//! | JSON punctuation — braces, quotes, commas, colons | 51% |
//! | document reference strings | 33% |
//! | position digits | 11% |
//! | term strings | 2% |
//! | term-frequency digits | 1% |
//!
//! Two facts drive the whole design. Over half the file is syntax, which a
//! binary layout removes outright. And a third is document references repeated
//! in full inside every posting — 23 bytes each, about 300 times per document —
//! which an interned document table replaces with a small integer.
//!
//! # What it achieves
//!
//! Measured on the same corpora, whole index, positions included:
//!
//! | corpus | JSON | binary | ratio | without positions |
//! |---|---|---|---|---|
//! | Chinese | 1.26 MB | 350 KB | 3.6× | 259 KB (4.9×) |
//! | Japanese | 2.54 MB | 439 KB | 5.8× | 257 KB (9.9×) |
//! | Korean | 2.13 MB | 404 KB | 5.3× | 249 KB (8.5×) |
//!
//! The composition inverts once the syntax is gone. Document references, a third
//! of the JSON, fall to 1.5% — interning did what the measurement predicted. The
//! remainder is postings (41–43%), positions (19–38%) and the term dictionary
//! (19–37%): real data rather than framing.
//!
//! The dictionary's share is larger for Chinese because that corpus has 13.4k
//! distinct bigrams against Japanese's 9.0k, and about 40% of the section is the
//! postings offset table — a flat four bytes per term. Delta-coding that table
//! would shrink it, but it is what makes locating a term's postings O(1) instead
//! of a scan, so it stays.
//!
//! Front-coding the term dictionary, the textbook first move, was worth only 2%
//! of the JSON file and was never the main lever. It is done because it makes
//! the dictionary binary-searchable in place.
//!
//! # Layout
//!
//! All integers are little-endian. Offsets are byte offsets from the start of
//! the file, so a reader can address any section without parsing the ones
//! before it.
//!
//! ```text
//! +----------------------------------+
//! | header            64 bytes fixed |
//! +----------------------------------+
//! | meta     language, field names,  |
//! |          field boosts            |
//! +----------------------------------+
//! | docs     reference heap, boosts, |
//! |          per-field lengths       |
//! +----------------------------------+
//! | terms    front-coded dictionary, |
//! |          block index, postings    |
//! |          offset table            |
//! +----------------------------------+
//! | postings per term: df, then per  |
//! |          field a delta-encoded   |
//! |          (doc, tf) list          |
//! +----------------------------------+
//! | positions  delta-encoded runs,   |
//! |            addressed from        |
//! |            postings              |
//! +----------------------------------+
//! ```
//!
//! ## Header (64 bytes)
//!
//! ```text
//! off  size  field
//!   0     4  magic = "MARZ"
//!   4     2  format_version
//!   6     2  flags (bit 0: positions present)
//!   8     4  doc_count       distinct interned references
//!  12     4  document_count  BM25's N
//!  16     4  field_count
//!  20     4  term_count
//!  24     8  k1  (f64)
//!  32     8  b   (f64)
//!  40     4  meta_offset
//!  44     4  docs_offset
//!  48     4  terms_offset
//!  52     4  postings_offset
//!  56     4  positions_offset
//!  60     4  end_offset (logical file length)
//! ```
//!
//! `doc_count` and `document_count` are separate because they can differ:
//! `doc_count` is how many distinct references were interned, while
//! `document_count` is how many documents were added and is what BM25's IDF
//! divides by. Adding the same reference twice increments only the latter.
//!
//! The number of dictionary blocks is not stored — it is
//! `term_count.div_ceil(TERMS_PER_BLOCK)`, and deriving it removes a field that
//! could contradict the data it describes.
//!
//! BM25 parameters and boosts are stored as `f64`, not `f32`. They are a
//! handful of values in the whole file, and narrowing them would make binary
//! scores differ from JSON scores in the last decimal places for no useful
//! saving.
//!
//! ## Documents
//!
//! Document references are sorted and assigned sequential ids, so a posting
//! list is an ascending run of ids that delta-encodes well, and a reference can
//! be found by binary search.
//!
//! Per-field lengths are a fixed-width `u32` matrix indexed by
//! `doc_id * field_count + field_id`. Scoring needs random access to them, so
//! varints — which would be smaller — would force a scan.
//!
//! ## Terms
//!
//! The dictionary is sorted and front-coded in blocks of
//! [`TERMS_PER_BLOCK`]. Each block stores its first term in full, then each
//! later term as `(shared_prefix_len, suffix_len, suffix)`. A block index of
//! offsets allows binary search to a block, then a short linear scan inside it.
//!
//! Blocking is what keeps the dictionary usable without decompressing it: a
//! fully front-coded list would have to be read from the start.
//!
//! A term's numeric id is its position in sorted order, which is also its index
//! into the postings offset table.
//!
//! ## Reading untrusted bytes
//!
//! A reader may be handed a truncated download or a corrupted cache entry.
//! Every accessor in [`reader`] returns [`FormatError`] rather than panicking,
//! and every offset is bounds-checked against the section it addresses. There
//! is no `unsafe` code and no transmute: multi-byte integers are assembled with
//! `from_le_bytes`, which also makes the format independent of host alignment
//! and endianness.

pub mod reader;
pub(crate) mod varint;
pub(crate) mod writer;

pub use reader::BinaryIndex;

/// File magic identifying a Marz binary index.
pub const MAGIC: [u8; 4] = *b"MARZ";

/// Format version written by this build.
///
/// A reader rejects anything it does not recognize rather than guessing, since
/// misreading an index produces wrong search results rather than an error.
pub const FORMAT_VERSION: u16 = 1;

/// Size of the fixed header.
pub const HEADER_LEN: usize = 64;

/// Terms per front-coded dictionary block.
///
/// A block is scanned linearly once binary search has located it, so this
/// trades dictionary size against lookup work. Sixteen keeps the block index
/// near 6% of dictionary size while bounding a lookup to fifteen prefix
/// reconstructions.
pub const TERMS_PER_BLOCK: usize = 16;

/// Flag bit: the file contains a positions section.
///
/// An index may be built without positions, which loses highlighting and CJK
/// phrase verification but drops roughly a tenth of the file.
pub const FLAG_HAS_POSITIONS: u16 = 1 << 0;

/// An error encountered while reading a binary index.
///
/// Every variant means the bytes are not a valid index — there is no partial
/// success. Variants name the section at fault so a caller can report
/// something more useful than "corrupt".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormatError {
    /// The input is shorter than the fixed header.
    TooShort {
        /// Bytes required.
        needed: usize,
        /// Bytes supplied.
        got: usize,
    },
    /// The magic bytes are not [`MAGIC`].
    BadMagic([u8; 4]),
    /// The format version is not one this build understands.
    UnsupportedVersion {
        /// Version found in the file.
        found: u16,
        /// Version this build writes.
        supported: u16,
    },
    /// A section's declared offset or length falls outside the file.
    SectionOutOfBounds {
        /// Section name.
        section: &'static str,
        /// Offset the header declared.
        offset: usize,
        /// Logical length of the file.
        end: usize,
    },
    /// Section offsets are not in ascending order, so a section overlaps
    /// another or has negative length.
    SectionsNotOrdered {
        /// The earlier section.
        first: &'static str,
        /// The section that should follow it but does not.
        second: &'static str,
    },
    /// A read ran past the end of its section.
    Truncated {
        /// Section name.
        section: &'static str,
    },
    /// A varint was longer than any `u64` encoding, so the stream is not a
    /// varint stream.
    MalformedVarint,
    /// Bytes that should have been UTF-8 were not.
    InvalidUtf8 {
        /// Section name.
        section: &'static str,
    },
    /// A document id was outside `0..document_count`.
    InvalidDocId(u32),
    /// A field id was outside `0..field_count`.
    InvalidFieldId(u32),
    /// A term id was outside `0..term_count`.
    InvalidTermId(u32),
    /// A front-coded term declared a shared prefix longer than the term it
    /// shares with.
    InvalidSharedPrefix {
        /// Declared shared prefix length.
        shared: usize,
        /// Length actually available.
        available: usize,
    },
    /// The file declares positions but the postings reference none, or the
    /// reverse.
    PositionsUnavailable,
}

impl std::fmt::Display for FormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooShort { needed, got } => {
                write!(f, "index is {got} bytes, need at least {needed}")
            }
            Self::BadMagic(found) => {
                write!(f, "not a Marz index: magic bytes {found:?}")
            }
            Self::UnsupportedVersion { found, supported } => write!(
                f,
                "index format version {found} is not supported (this build reads {supported})"
            ),
            Self::SectionOutOfBounds {
                section,
                offset,
                end,
            } => write!(
                f,
                "section '{section}' starts at {offset} but the index ends at {end}"
            ),
            Self::SectionsNotOrdered { first, second } => {
                write!(f, "section '{second}' does not follow '{first}'")
            }
            Self::Truncated { section } => {
                write!(f, "section '{section}' ends unexpectedly")
            }
            Self::MalformedVarint => write!(f, "malformed varint"),
            Self::InvalidUtf8 { section } => {
                write!(f, "section '{section}' contains invalid UTF-8")
            }
            Self::InvalidDocId(id) => write!(f, "document id {id} out of range"),
            Self::InvalidFieldId(id) => write!(f, "field id {id} out of range"),
            Self::InvalidTermId(id) => write!(f, "term id {id} out of range"),
            Self::InvalidSharedPrefix { shared, available } => write!(
                f,
                "front-coded term shares {shared} bytes with a {available}-byte term"
            ),
            Self::PositionsUnavailable => {
                write!(f, "this index was built without positions")
            }
        }
    }
}

impl std::error::Error for FormatError {}

/// Parsed fixed header.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Header {
    /// Format version.
    pub format_version: u16,
    /// Flag bits.
    pub flags: u16,
    /// Number of distinct interned document references.
    pub doc_count: u32,
    /// BM25's `N`: documents added to the index.
    pub document_count: u32,
    /// Number of indexed fields.
    pub field_count: u32,
    /// Number of distinct terms.
    pub term_count: u32,
    /// BM25 term-frequency saturation parameter.
    pub k1: f64,
    /// BM25 length-normalization parameter.
    pub b: f64,
    /// Offset of the metadata section.
    pub meta_offset: u32,
    /// Offset of the document section.
    pub docs_offset: u32,
    /// Offset of the term dictionary section.
    pub terms_offset: u32,
    /// Offset of the postings section.
    pub postings_offset: u32,
    /// Offset of the positions section.
    pub positions_offset: u32,
    /// Logical end of the index.
    pub end_offset: u32,
}

impl Header {
    /// Whether the index carries token positions.
    pub fn has_positions(&self) -> bool {
        self.flags & FLAG_HAS_POSITIONS != 0
    }

    /// Number of front-coded dictionary blocks.
    pub fn term_block_count(&self) -> usize {
        (self.term_count as usize).div_ceil(TERMS_PER_BLOCK)
    }
}

/// Read a little-endian `u32` at `offset`, or fail if it does not fit.
pub(crate) fn read_u32(
    bytes: &[u8],
    offset: usize,
    section: &'static str,
) -> Result<u32, FormatError> {
    let end = offset
        .checked_add(4)
        .ok_or(FormatError::Truncated { section })?;
    let slice = bytes
        .get(offset..end)
        .ok_or(FormatError::Truncated { section })?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

/// Read a little-endian `f64` at `offset`, or fail if it does not fit.
pub(crate) fn read_f64(
    bytes: &[u8],
    offset: usize,
    section: &'static str,
) -> Result<f64, FormatError> {
    let end = offset
        .checked_add(8)
        .ok_or(FormatError::Truncated { section })?;
    let slice = bytes
        .get(offset..end)
        .ok_or(FormatError::Truncated { section })?;
    let mut buf = [0u8; 8];
    buf.copy_from_slice(slice);
    Ok(f64::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_u32_rejects_a_truncated_tail() {
        // Three bytes cannot hold a u32, and must not panic on the slice.
        assert!(read_u32(&[1, 2, 3], 0, "test").is_err());
        assert_eq!(read_u32(&[1, 0, 0, 0], 0, "test"), Ok(1));
    }

    #[test]
    fn read_u32_rejects_an_offset_that_overflows() {
        assert!(read_u32(&[0; 8], usize::MAX, "test").is_err());
    }

    #[test]
    fn read_f64_roundtrips() {
        let bytes = 1.25f64.to_le_bytes();
        assert_eq!(read_f64(&bytes, 0, "test"), Ok(1.25));
        assert!(read_f64(&bytes[..7], 0, "test").is_err());
    }

    #[test]
    fn error_messages_name_the_problem() {
        // These strings surface to users loading a broken index, so check they
        // say something actionable rather than Debug-formatted noise.
        let e = FormatError::BadMagic(*b"JSON");
        assert!(e.to_string().contains("not a Marz index"), "{e}");
        let e = FormatError::UnsupportedVersion {
            found: 9,
            supported: FORMAT_VERSION,
        };
        assert!(e.to_string().contains("version 9"), "{e}");
    }
}
