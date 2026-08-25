//! LEB128 variable-length integer coding.
//!
//! Posting lists are the bulk of an index, and almost every number in them is
//! small: a term-frequency is usually 1, and a delta between adjacent document
//! ids or token positions is usually a handful. A fixed `u32` spends four bytes
//! on each; a varint spends one for anything under 128.
//!
//! The encoding is the standard unsigned LEB128 used by DWARF, Protocol Buffers
//! and WebAssembly: seven payload bits per byte, little-endian, with the high
//! bit set on every byte but the last.
//!
//! # Delta coding
//!
//! Varints only pay off on small numbers, so sequences are stored as deltas
//! rather than absolute values. Document ids within a posting list ascend, as do
//! token positions within a field, so each delta is small even when the absolute
//! values are not — position 40,000 in a long document costs three bytes as an
//! absolute value and one as a delta.
//!
//! Deltas are stored as `next - previous` with no zero-adjustment, which means a
//! repeated value encodes as 0. That is valid for positions, where two tokens
//! can share a start offset only if the tokenizer emitted them at the same
//! place, and it is what makes decoding a pure running sum with no special
//! cases.

use super::FormatError;

/// Maximum bytes in a `u64` LEB128 encoding: 64 bits at 7 bits per byte.
const MAX_VARINT_LEN: usize = 10;

/// Append `value` to `out` as a varint.
pub fn write_varint(out: &mut Vec<u8>, mut value: u64) {
    // The loop must run at least once so that zero encodes as a single 0x00
    // byte rather than nothing at all.
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// Number of bytes `write_varint` would append for `value`.
///
/// Only the tests need this: the writer builds each section into its own buffer
/// and reads the real length back, rather than predicting it. It is kept so a
/// test can assert an encoding costs what the format's size claims say it does.
#[cfg(test)]
pub fn varint_len(value: u64) -> usize {
    let mut len = 1;
    let mut value = value >> 7;
    while value != 0 {
        len += 1;
        value >>= 7;
    }
    len
}

/// A cursor over a varint-encoded byte slice.
///
/// Holds the section name so errors can say which part of the index was
/// malformed rather than just "bad varint".
#[derive(Debug, Clone)]
pub struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
    section: &'static str,
}

impl<'a> Cursor<'a> {
    /// Start reading `bytes`, attributing errors to `section`.
    pub fn new(bytes: &'a [u8], section: &'static str) -> Self {
        Self {
            bytes,
            pos: 0,
            section,
        }
    }

    /// Bytes consumed so far.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Whether the cursor has reached the end of its slice.
    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.pos >= self.bytes.len()
    }

    /// Remaining bytes, without consuming them.
    #[cfg(test)]
    pub fn remaining(&self) -> &'a [u8] {
        &self.bytes[self.pos.min(self.bytes.len())..]
    }

    /// Skip `n` bytes.
    pub fn skip(&mut self, n: usize) -> Result<(), FormatError> {
        let end = self.pos.checked_add(n).ok_or(FormatError::Truncated {
            section: self.section,
        })?;
        if end > self.bytes.len() {
            return Err(FormatError::Truncated {
                section: self.section,
            });
        }
        self.pos = end;
        Ok(())
    }

    /// Read the next varint.
    ///
    /// Fails on a truncated stream, and on a run of continuation bytes longer
    /// than any `u64` encoding — which would otherwise let a corrupt file
    /// silently shift bits off the top of the accumulator.
    pub fn read_varint(&mut self) -> Result<u64, FormatError> {
        let mut value: u64 = 0;
        let mut shift = 0u32;
        for _ in 0..MAX_VARINT_LEN {
            let byte = *self.bytes.get(self.pos).ok_or(FormatError::Truncated {
                section: self.section,
            })?;
            self.pos += 1;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
            shift += 7;
        }
        Err(FormatError::MalformedVarint)
    }

    /// Read a varint that must fit in a `u32`.
    pub fn read_u32(&mut self) -> Result<u32, FormatError> {
        let value = self.read_varint()?;
        u32::try_from(value).map_err(|_| FormatError::MalformedVarint)
    }

    /// Read a varint that must fit in a `usize`.
    pub fn read_usize(&mut self) -> Result<usize, FormatError> {
        let value = self.read_varint()?;
        usize::try_from(value).map_err(|_| FormatError::MalformedVarint)
    }

    /// Read a length-prefixed UTF-8 string.
    pub fn read_str(&mut self) -> Result<&'a str, FormatError> {
        let len = self.read_usize()?;
        let end = self.pos.checked_add(len).ok_or(FormatError::Truncated {
            section: self.section,
        })?;
        let slice = self
            .bytes
            .get(self.pos..end)
            .ok_or(FormatError::Truncated {
                section: self.section,
            })?;
        self.pos = end;
        std::str::from_utf8(slice).map_err(|_| FormatError::InvalidUtf8 {
            section: self.section,
        })
    }
}

/// Append a length-prefixed UTF-8 string.
pub fn write_str(out: &mut Vec<u8>, value: &str) {
    write_varint(out, value.len() as u64);
    out.extend_from_slice(value.as_bytes());
}

/// Bytes `write_str` would append.
///
/// Like [`varint_len`], only the tests need this — see the note there.
#[cfg(test)]
pub fn str_len(value: &str) -> usize {
    varint_len(value.len() as u64) + value.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Append an ascending sequence as deltas from `base`.
    ///
    /// The writer inlines this loop at each of its call sites, because each one
    /// interleaves something else with the deltas (a token length, a term
    /// frequency) and works from a different integer type. This copy exists so
    /// the tests can pin the encoding the format's size claims rest on.
    fn write_deltas(out: &mut Vec<u8>, values: &[usize], base: usize) {
        let mut previous = base;
        for &value in values {
            write_varint(out, (value.saturating_sub(previous)) as u64);
            previous = value;
        }
    }

    fn roundtrip(value: u64) {
        let mut buf = Vec::new();
        write_varint(&mut buf, value);
        assert_eq!(buf.len(), varint_len(value), "len mismatch for {value}");
        let mut cursor = Cursor::new(&buf, "test");
        assert_eq!(cursor.read_varint(), Ok(value));
        assert!(cursor.is_empty(), "{value} left trailing bytes");
    }

    #[test]
    fn varints_roundtrip_across_byte_boundaries() {
        // Every boundary where the encoding grows by a byte, plus the extremes.
        for value in [
            0,
            1,
            127,
            128,
            129,
            16_383,
            16_384,
            2_097_151,
            2_097_152,
            u32::MAX as u64,
            u64::MAX - 1,
            u64::MAX,
        ] {
            roundtrip(value);
        }
    }

    #[test]
    fn zero_occupies_one_byte() {
        // A loop written as `while value != 0` would emit nothing here, and the
        // stream would silently desynchronize.
        let mut buf = Vec::new();
        write_varint(&mut buf, 0);
        assert_eq!(buf, [0]);
    }

    #[test]
    fn small_values_cost_one_byte() {
        // The whole reason for the encoding: term frequencies and deltas are
        // nearly always below 128.
        for value in 0..128u64 {
            assert_eq!(varint_len(value), 1);
        }
        assert_eq!(varint_len(128), 2);
    }

    #[test]
    fn truncated_varint_is_an_error() {
        // 0x80 promises a continuation byte that never arrives.
        let mut cursor = Cursor::new(&[0x80], "test");
        assert_eq!(
            cursor.read_varint(),
            Err(FormatError::Truncated { section: "test" })
        );
    }

    #[test]
    fn overlong_varint_is_an_error() {
        // Eleven continuation bytes cannot be a u64. Without the length cap the
        // shift would exceed 63 and the value would be garbage.
        let bytes = [0x80u8; 11];
        let mut cursor = Cursor::new(&bytes, "test");
        assert_eq!(cursor.read_varint(), Err(FormatError::MalformedVarint));
    }

    #[test]
    fn read_u32_rejects_a_value_that_does_not_fit() {
        let mut buf = Vec::new();
        write_varint(&mut buf, u64::from(u32::MAX) + 1);
        let mut cursor = Cursor::new(&buf, "test");
        assert_eq!(cursor.read_u32(), Err(FormatError::MalformedVarint));
    }

    #[test]
    fn strings_roundtrip_including_multibyte() {
        let mut buf = Vec::new();
        write_str(&mut buf, "検索エンジン");
        write_str(&mut buf, "");
        write_str(&mut buf, "ascii");
        assert_eq!(
            buf.len(),
            str_len("検索エンジン") + str_len("") + str_len("ascii")
        );

        let mut cursor = Cursor::new(&buf, "test");
        assert_eq!(cursor.read_str(), Ok("検索エンジン"));
        assert_eq!(cursor.read_str(), Ok(""));
        assert_eq!(cursor.read_str(), Ok("ascii"));
        assert!(cursor.is_empty());
    }

    #[test]
    fn invalid_utf8_string_is_an_error() {
        // 0xff is not a valid UTF-8 lead byte. A corrupt index must not produce
        // a garbage term.
        let buf = [1u8, 0xff];
        let mut cursor = Cursor::new(&buf, "terms");
        assert_eq!(
            cursor.read_str(),
            Err(FormatError::InvalidUtf8 { section: "terms" })
        );
    }

    #[test]
    fn string_longer_than_the_buffer_is_an_error() {
        let buf = [200u8, b'a'];
        let mut cursor = Cursor::new(&buf, "terms");
        assert!(cursor.read_str().is_err());
    }

    #[test]
    fn deltas_roundtrip_as_a_running_sum() {
        let values = [0usize, 1, 2, 40, 41, 5000];
        let mut buf = Vec::new();
        write_deltas(&mut buf, &values, 0);

        let mut cursor = Cursor::new(&buf, "test");
        let mut decoded = Vec::new();
        let mut previous = 0usize;
        while !cursor.is_empty() {
            previous += cursor.read_usize().unwrap();
            decoded.push(previous);
        }
        assert_eq!(decoded, values);
    }

    #[test]
    fn ascending_positions_cost_one_byte_each() {
        // The claim that motivates delta coding: absolute position 5000 needs
        // two bytes, but a step of 1 from 4999 needs one.
        let values: Vec<usize> = (4000..4100).collect();
        let mut buf = Vec::new();
        write_deltas(&mut buf, &values, 0);
        assert_eq!(buf.len(), 2 + 99, "expected one varint of 2 bytes then 1s");
    }

    #[test]
    fn repeated_value_encodes_as_a_zero_delta() {
        let mut buf = Vec::new();
        write_deltas(&mut buf, &[7, 7], 0);
        let mut cursor = Cursor::new(&buf, "test");
        assert_eq!(cursor.read_usize(), Ok(7));
        assert_eq!(cursor.read_usize(), Ok(0));
    }

    #[test]
    fn skip_and_remaining_stay_in_bounds() {
        let mut cursor = Cursor::new(&[1, 2, 3, 4], "test");
        assert!(cursor.skip(2).is_ok());
        assert_eq!(cursor.remaining(), &[3, 4]);
        assert!(cursor.skip(3).is_err(), "must not skip past the end");
        assert!(cursor.skip(usize::MAX).is_err(), "must not overflow");
    }
}
