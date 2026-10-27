//! Structurally valid but adversarial index bytes.
//!
//! `hostile_input.rs` covers truncation, bit-rot and random noise — corruption
//! that looks like corruption. This file covers the class that does not: a file
//! that satisfies every *local* check (offsets ascending, sections in bounds,
//! varints well formed, ids in range) while its structure as a whole is a lie.
//!
//! These are the expensive bugs. A 100 KB file whose terms all point at one
//! shared position block costs gigabytes of decode work and is accepted by a
//! reader that checks each read in isolation; two documents interned under one
//! reference merge silently into one hit and get wrong scores; a forged
//! `document_count` rescales every IDF without any error anywhere.
//!
//! So the assertions here are structural: the file must be *rejected*, and
//! rejected before the work is done, not merely return odd results.

use std::sync::Arc;

use marz_core::languages::{English, Japanese};
use marz_core::{BinaryIndex, Index, IndexBuilder, Language};

// ---------------------------------------------------------------------------
// Byte-level helpers. The format is documented in `marz_core::binary`.
// ---------------------------------------------------------------------------

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn write_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

/// Header field offsets, named so the mutations below read as intent.
const DOC_COUNT: usize = 8;
const DOCUMENT_COUNT: usize = 12;
const TERM_COUNT: usize = 20;
const K1: usize = 24;
const B: usize = 32;
const DOCS_OFFSET: usize = 44;
const TERMS_OFFSET: usize = 48;
const POSTINGS_OFFSET: usize = 52;
const POSITIONS_OFFSET: usize = 56;
const END_OFFSET: usize = 60;

/// Encode a varint, mirroring `binary::varint::write_varint`.
fn varint(mut value: u64) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return out;
        }
        out.push(byte | 0x80);
    }
}

fn read_varint(bytes: &[u8], pos: &mut usize) -> u64 {
    let mut value = 0u64;
    let mut shift = 0;
    loop {
        let byte = bytes[*pos];
        *pos += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return value;
        }
        shift += 7;
    }
}

fn push_varint(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&varint(value));
}

/// Reassemble a file from sections, patching every offset in the header.
fn rebuild(original: &[u8], terms: &[u8], postings: &[u8], positions: &[u8]) -> Vec<u8> {
    let meta = &original[read_u32(original, 40) as usize..read_u32(original, DOCS_OFFSET) as usize];
    let docs = &original
        [read_u32(original, DOCS_OFFSET) as usize..read_u32(original, TERMS_OFFSET) as usize];

    let mut out = original[..64].to_vec();
    let meta_offset = 64usize;
    let docs_offset = meta_offset + meta.len();
    let terms_offset = docs_offset + docs.len();
    let postings_offset = terms_offset + terms.len();
    let positions_offset = postings_offset + postings.len();
    let end_offset = positions_offset + positions.len();

    for (offset, value) in [
        (40, meta_offset),
        (DOCS_OFFSET, docs_offset),
        (TERMS_OFFSET, terms_offset),
        (POSTINGS_OFFSET, postings_offset),
        (POSITIONS_OFFSET, positions_offset),
        (END_OFFSET, end_offset),
    ] {
        write_u32(&mut out, offset, value as u32);
    }
    out.extend_from_slice(meta);
    out.extend_from_slice(docs);
    out.extend_from_slice(terms);
    out.extend_from_slice(postings);
    out.extend_from_slice(positions);
    out
}

/// The postings section, decoded just enough to re-emit it.
///
/// One entry per field is `(field_id, entries)`, and one entry per posting is
/// `(doc_delta, term_frequency, position_count)`.
type DecodedEntries = Vec<(u64, u64, u64)>;

struct DecodedTerm {
    document_frequency: u64,
    fields: Vec<(u64, DecodedEntries)>,
}

fn decode_postings(slice: &[u8]) -> DecodedTerm {
    let mut pos = 0;
    let _positions_base = read_varint(slice, &mut pos);
    let document_frequency = read_varint(slice, &mut pos);
    let field_count = read_varint(slice, &mut pos);
    let mut fields = Vec::new();
    for _ in 0..field_count {
        let field_id = read_varint(slice, &mut pos);
        let entry_count = read_varint(slice, &mut pos);
        let mut entries = Vec::new();
        for _ in 0..entry_count {
            let doc_delta = read_varint(slice, &mut pos);
            let tf = read_varint(slice, &mut pos);
            let position_count = read_varint(slice, &mut pos);
            entries.push((doc_delta, tf, position_count));
        }
        fields.push((field_id, entries));
    }
    DecodedTerm {
        document_frequency,
        fields,
    }
}

/// A small positions-bearing English index.
fn fixture() -> (Index, Vec<u8>) {
    let mut builder = IndexBuilder::new(Arc::new(English) as Arc<dyn Language>);
    builder.ref_field("id").field("body", 1.0);
    for (id, text) in [
        ("doc-a", "keyboard cable keyboard keys"),
        ("doc-b", "cable mouse keyboard"),
        ("doc-c", "mouse pad keyboard"),
    ] {
        builder.add(id, 1.0, |name| match name {
            "body" => Some(text.to_string()),
            _ => None,
        });
    }
    let index = builder.build();
    let bytes = index.to_binary(true).unwrap();
    (index, bytes)
}

// ---------------------------------------------------------------------------
// #1: every term pointing at one shared position block.
// ---------------------------------------------------------------------------

/// Rewrite the file so every term's position blocks start at offset 0 and every
/// posting entry claims the whole positions section.
///
/// Structurally valid in the sense the reader checks locally: offsets ascend,
/// sections are in bounds, every varint parses, every id is in range, and each
/// term's own block length is consistent with the block it names — because it
/// names the *whole* block. Only the cross-term layout is a lie, and that is what
/// makes the decode cost multiplicative: `term_count` terms each expand the
/// entire positions section into owned `Vec`s.
fn forge_shared_positions(bytes: &[u8]) -> Vec<u8> {
    let terms_offset = read_u32(bytes, TERMS_OFFSET) as usize;
    let postings_offset = read_u32(bytes, POSTINGS_OFFSET) as usize;
    let term_count = read_u32(bytes, TERM_COUNT) as usize;
    let block_count = term_count.div_ceil(16);
    let table = (block_count + 1) * 4;

    let postings_offsets: Vec<usize> = (0..=term_count)
        .map(|i| read_u32(bytes, terms_offset + table + i * 4) as usize)
        .collect();

    let decoded: Vec<DecodedTerm> = (0..term_count)
        .map(|i| {
            decode_postings(
                &bytes[postings_offset + postings_offsets[i]
                    ..postings_offset + postings_offsets[i + 1]],
            )
        })
        .collect();

    // One shared block, uniform position length 2, so the block is 1 + N bytes.
    let total_positions: u64 = 20_000;
    let mut positions = Vec::new();
    push_varint(&mut positions, 2);
    for _ in 0..total_positions {
        push_varint(&mut positions, 1);
    }

    let mut postings = Vec::new();
    let mut new_offsets = Vec::with_capacity(term_count + 1);
    for term in &decoded {
        new_offsets.push(postings.len() as u32);
        push_varint(&mut postings, 0); // positions_base: the shared block
        push_varint(&mut postings, term.document_frequency);
        // Exactly one field and one entry, so the entry's claim of the whole
        // shared block still fits the section and decodes locally.
        let (field_id, entries) = term.fields.first().expect("fixture terms have fields");
        push_varint(&mut postings, 1);
        push_varint(&mut postings, *field_id);
        push_varint(&mut postings, 1);
        let (doc_delta, tf, _) = entries.first().expect("fixture terms have entries");
        push_varint(&mut postings, *doc_delta);
        push_varint(&mut postings, *tf);
        push_varint(&mut postings, total_positions);
    }
    new_offsets.push(postings.len() as u32);

    // Dictionary bytes and block offsets are unchanged; only the postings
    // offset table is replaced.
    let dictionary = &bytes[terms_offset + table + (term_count + 1) * 4..postings_offset];
    let mut terms = bytes[terms_offset..terms_offset + table].to_vec();
    for offset in &new_offsets {
        terms.extend_from_slice(&offset.to_le_bytes());
    }
    terms.extend_from_slice(dictionary);

    rebuild(bytes, &terms, &postings, &positions)
}

#[test]
fn aliased_position_blocks_are_rejected_before_allocation() {
    // The attack: 100 KB of file, gigabytes of decode work, because every term
    // claims every position. Measured on a real reproduction, that was 57 s and
    // 1.6 GB of RSS — in a browser tab, on a downloaded index.
    let (_index, bytes) = fixture();
    let forged = forge_shared_positions(&bytes);

    // The forgery must be structurally decodable, or the test proves nothing:
    // `open` is the cheap path and it accepts what a local check can accept.
    let opened = BinaryIndex::open(&forged).expect("forgery should be structurally openable");
    assert!(
        opened.postings(0).is_ok(),
        "the forged postings must decode, or validation is not what rejects it"
    );

    // Materialization must refuse it — and refuse it via `validate`, before the
    // postings are turned into owned structures.
    let err = Index::from_binary(&forged, Arc::new(English))
        .err()
        .expect("aliased position blocks must be rejected");
    assert!(
        matches!(
            err,
            marz_core::binary::FormatError::Truncated {
                section: "positions"
            }
        ),
        "expected a positions-layout error, got {err:?}"
    );
    assert!(
        opened.validate().is_err(),
        "validate must reject it directly"
    );
}

#[test]
fn a_legitimately_empty_positions_bearing_index_still_loads() {
    // The control for #1: validation must not reject a valid file just because
    // it has no positions to speak of. An empty index and an index whose
    // postings all have empty blocks are both legal.
    let mut builder = IndexBuilder::new(Arc::new(English) as Arc<dyn Language>);
    builder.ref_field("id").field("body", 1.0);
    let empty = builder.build();
    let bytes = empty.to_binary(true).unwrap();
    let loaded = Index::from_binary(&bytes, Arc::new(English)).expect("empty index must load");
    assert_eq!(loaded.document_count(), 0);

    // Prebuilt fixture, positions included: the ordinary case.
    let (index, bytes) = fixture();
    let loaded = Index::from_binary(&bytes, Arc::new(English)).expect("valid index must load");
    assert_eq!(loaded.document_count(), index.document_count());
    assert!(!loaded.search("keyboard").unwrap().is_empty());

    // And the same file with positions stripped is still valid.
    let without = index.to_binary(false).unwrap();
    Index::from_binary(&without, Arc::new(English)).expect("positions-free index must load");
}

// ---------------------------------------------------------------------------
// #8: duplicate or unsorted document references.
// ---------------------------------------------------------------------------

/// The byte range of document `id`'s reference in the heap.
fn doc_ref_range(bytes: &[u8], id: usize) -> (usize, usize) {
    let docs_offset = read_u32(bytes, DOCS_OFFSET) as usize;
    let start = read_u32(bytes, docs_offset + id * 4) as usize;
    let end = read_u32(bytes, docs_offset + (id + 1) * 4) as usize;
    let heap_base = (read_u32(bytes, DOC_COUNT) as usize + 1) * 4
        + read_u32(bytes, DOC_COUNT) as usize * 8
        + read_u32(bytes, DOC_COUNT) as usize * read_u32(bytes, 16) as usize * 4;
    (
        docs_offset + heap_base + start,
        docs_offset + heap_base + end,
    )
}

#[test]
fn duplicate_document_references_are_rejected() {
    // Two documents interned under one reference silently merge: `alpha` in both
    // returns one hit, not two, and its score jumps because the merged posting
    // looks like a single very relevant document.
    let (_index, bytes) = fixture();
    let mut forged = bytes.clone();
    let (first_start, first_end) = doc_ref_range(&forged, 0);
    let (last_start, last_end) = doc_ref_range(&forged, 2);
    assert_eq!(
        first_end - first_start,
        last_end - last_start,
        "fixture refs must be the same width for this mutation"
    );
    forged[last_start..last_end].copy_from_slice(&bytes[first_start..first_end]);

    assert!(
        Index::from_binary(&forged, Arc::new(English)).is_err(),
        "duplicate references must be rejected, not merged"
    );
}

#[test]
fn unsorted_document_references_are_rejected() {
    // References are sorted so `doc_id` can binary search. Swapping two makes
    // the search miss a reference that is present in the file.
    let (_index, bytes) = fixture();
    let mut forged = bytes.clone();
    let (a_start, a_end) = doc_ref_range(&forged, 0);
    let (b_start, b_end) = doc_ref_range(&forged, 1);
    let a: Vec<u8> = bytes[a_start..a_end].to_vec();
    let b: Vec<u8> = bytes[b_start..b_end].to_vec();
    forged[a_start..a_end].copy_from_slice(&b);
    forged[b_start..b_end].copy_from_slice(&a);

    assert!(
        Index::from_binary(&forged, Arc::new(English)).is_err(),
        "unsorted references must be rejected"
    );
}

#[test]
fn a_forged_document_id_beyond_the_table_is_rejected() {
    // The other direction: a posting that names a document that does not exist
    // must fail rather than panic on the lookup.
    let (_index, bytes) = fixture();
    let terms_offset = read_u32(&bytes, TERMS_OFFSET) as usize;
    let postings_offset = read_u32(&bytes, POSTINGS_OFFSET) as usize;
    let term_count = read_u32(&bytes, TERM_COUNT) as usize;
    let block_count = term_count.div_ceil(16);
    let table = (block_count + 1) * 4;
    let first_posting = postings_offset + read_u32(&bytes, terms_offset + table) as usize;

    // Skip positions_base, df and field_count, then field_id and entry_count,
    // and overwrite the first doc delta with a large value.
    let mut pos = first_posting;
    for _ in 0..4 {
        read_varint(&bytes, &mut pos);
    }
    let mut forged = bytes.clone();
    forged.splice(pos..=pos, varint(200));
    let new_end = forged.len() as u32;
    write_u32(&mut forged, END_OFFSET, new_end);
    let delta = new_end.wrapping_sub(bytes.len() as u32);
    for offset in [POSTINGS_OFFSET, POSITIONS_OFFSET] {
        let value = read_u32(&forged, offset);
        write_u32(&mut forged, offset, value.wrapping_add(delta));
    }
    // Keep the postings offset table in step with the splice.
    for i in 0..=read_u32(&forged, TERM_COUNT) as usize {
        let at = terms_offset + table + i * 4;
        let value = read_u32(&forged, at);
        if value as usize >= pos - postings_offset {
            write_u32(&mut forged, at, value.wrapping_add(delta));
        }
    }

    assert!(
        Index::from_binary(&forged, Arc::new(English)).is_err(),
        "an out-of-range document id must be rejected"
    );
}

// ---------------------------------------------------------------------------
// #9: forged header values.
// ---------------------------------------------------------------------------

#[test]
fn a_forged_document_count_is_rejected() {
    // `document_count` is BM25's N. Setting it to 0 or u32::MAX does not break
    // any bounds check; it just rescales every IDF, so results stay plausible
    // while the ranking is wrong. Two `alpha` scores moved 0.182 -> 0.693 in the
    // reported repro.
    let (_index, bytes) = fixture();
    for value in [0u32, 1, u32::MAX] {
        let mut forged = bytes.clone();
        write_u32(&mut forged, DOCUMENT_COUNT, value);
        assert!(
            Index::from_binary(&forged, Arc::new(English)).is_err(),
            "document_count {value} contradicts doc_count and must be rejected"
        );
    }
}

/// Forged BM25 parameters are *sanitized*, not rejected: both have a defined
/// fallback the builder itself uses, so a wrong value degrades to the default
/// instead of refusing an otherwise readable index.
///
/// This is deliberately different from `a_forged_document_count_is_rejected`.
/// `document_count` has no safe replacement — N is not derivable from the file,
/// and guessing it silently changes ranking — whereas `k1`/`b` have defaults the
/// builder already applies to out-of-range input.
#[test]
fn forged_bm25_parameters_are_sanitized_to_builder_defaults() {
    let (_index, bytes) = fixture();
    for (offset, value, expected) in [
        (K1, f64::NAN, 1.2),
        (K1, f64::INFINITY, 1.2),
        (K1, -1.0, 1.2),
        (B, f64::NAN, 0.75),
        (B, f64::NEG_INFINITY, 0.75),
        (B, 1.5, 1.0),
        (B, -0.5, 0.0),
    ] {
        let mut forged = bytes.clone();
        forged[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
        let loaded = Index::from_binary(&forged, Arc::new(English)).unwrap_or_else(|e| {
            panic!("BM25 parameter {value} at {offset} should sanitize, not reject: {e:?}")
        });
        let actual = if offset == K1 {
            loaded.k1()
        } else {
            loaded.b()
        };
        assert_eq!(
            actual, expected,
            "BM25 parameter {value} at {offset} must sanitize to {expected}"
        );
    }
    // A sanitized parameter must still produce finite, ordered scores.
    let mut forged = bytes.clone();
    forged[K1..K1 + 8].copy_from_slice(&f64::NAN.to_le_bytes());
    let loaded = Index::from_binary(&forged, Arc::new(English)).unwrap();
    assert!(loaded
        .search("keyboard")
        .unwrap()
        .iter()
        .all(|r| r.score.is_finite()));

    // The in-range values a writer can legitimately emit still load unchanged.
    for (offset, value) in [(K1, 0.0f64), (K1, 2.0), (B, 0.0), (B, 1.0)] {
        let mut honest = bytes.clone();
        honest[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
        let loaded = Index::from_binary(&honest, Arc::new(English)).unwrap();
        let actual = if offset == K1 {
            loaded.k1()
        } else {
            loaded.b()
        };
        assert_eq!(
            actual, value,
            "an honest {value} at {offset} must load as-is"
        );
    }
}

#[test]
fn a_positions_free_file_relabelled_as_positions_bearing_is_rejected() {
    // Flipping the flag bit on a positions-free index silently disables phrase
    // verification: every phrase boost becomes 1.0 and highlighting returns
    // empty spans, with no error to explain either.
    let (index, _) = fixture();
    let mut forged = index.to_binary(false).unwrap();
    let flags = u16::from_le_bytes([forged[6], forged[7]]) | 1;
    forged[6..8].copy_from_slice(&flags.to_le_bytes());

    assert!(
        Index::from_binary(&forged, Arc::new(English)).is_err(),
        "a relabelled positions flag must be rejected"
    );
}

// ---------------------------------------------------------------------------
// #17: `open` is constant-time, so whole-table checks live in `validate`.
// ---------------------------------------------------------------------------

#[test]
fn open_does_not_walk_the_document_table() {
    // The documented contract: `open` validates the header and computes section
    // bounds, and is constant-time in the corpus. `indexLanguage()` depends on
    // it. So a broken document table must survive `open` and be caught by
    // `validate` — which is exactly the split the two entry points have.
    let (_index, bytes) = fixture();
    let mut forged = bytes.clone();
    // Doc 1's reference is now also doc 0's: a duplicate, which is a document
    // table invariant, while every section offset stays sane.
    let (a_start, a_end) = doc_ref_range(&forged, 0);
    let (b_start, b_end) = doc_ref_range(&forged, 1);
    assert_eq!(
        a_end - a_start,
        b_end - b_start,
        "fixture refs must be the same width"
    );
    forged.copy_within(b_start..b_end, a_start);

    let opened = BinaryIndex::open(&forged).expect("open must not walk the document table");
    assert!(
        opened.validate().is_err(),
        "validate is where the document table is checked"
    );
    assert!(
        Index::from_binary(&forged, Arc::new(English)).is_err(),
        "materialization must reject what validate rejects"
    );
}

#[test]
fn a_valid_index_survives_validate_and_roundtrips() {
    // The counterweight to every rejection above: the writer's own output must
    // pass validation, in both position modes and for a CJK corpus where every
    // term has a position block.
    let (_index, bytes) = fixture();
    BinaryIndex::open(&bytes).unwrap().validate().unwrap();

    let mut builder = IndexBuilder::new(Arc::new(Japanese) as Arc<dyn Language>);
    builder.ref_field("id").field("body", 1.0);
    for (id, text) in [
        ("ja/1", "検索エンジンの仕組みについて説明します。"),
        ("ja/2", "機械学習は人工知能の一分野です。"),
    ] {
        builder.add(id, 1.0, |name| match name {
            "body" => Some(text.to_string()),
            _ => None,
        });
    }
    let ja = builder.build();
    for include_positions in [true, false] {
        let bytes = ja.to_binary(include_positions).unwrap();
        let opened = BinaryIndex::open(&bytes).expect("open");
        opened
            .validate()
            .expect("the writer's output must validate");
        let loaded = Index::from_binary(&bytes, Arc::new(Japanese)).expect("load");
        assert_eq!(loaded.document_count(), 2);
        assert!(loaded
            .search("検索")
            .unwrap()
            .iter()
            .any(|r| r.ref_id == "ja/1"));
    }
}
