//! Query lexer and parser for lunr query syntax.
//!
//! Supports the same syntax as lunr:
//!
//! * `term` — optional term
//! * `+term` — required term
//! * `-term` — prohibited term
//! * `field:term` — field-scoped term
//! * `term^N` — boost (float, e.g. `^2.5`, exponent, e.g. `^1e3`;
//!   non-negative — `^-1` is a parse error, while a programmatically built
//!   negative boost clamps to `0.0`)
//! * `term~N` — fuzzy edit distance (non-negative integer; `*` inside a fuzzy
//!   term is an ordinary character, so `a\*b~1` still means literal `a*b`)
//! * `term~N` — fuzzy edit distance (non-negative integer)
//! * backslash escaping for special characters (`\*` is literal, not a wildcard)
//!
//! `start`/`end` on [`QueryParseError`] and lexemes are **character** offsets
//! into the query string, not byte offsets: slicing `query.as_bytes()` with
//! them panics on multibyte input. Slice with `query.chars()` or convert via
//! `char_indices`.

use crate::language::LanguageRef;
use crate::normalize::normalize_for_language;
use crate::query::{Clause, Presence, Query};

/// Error produced when a query string cannot be parsed.
#[derive(Debug, Clone, PartialEq)]
pub struct QueryParseError {
    /// Human-readable message.
    pub message: String,
    /// Start position in the query string.
    pub start: usize,
    /// End position in the query string.
    pub end: usize,
}

impl std::fmt::Display for QueryParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "query parse error at {}-{}: {}",
            self.start, self.end, self.message
        )
    }
}

impl std::error::Error for QueryParseError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LexemeType {
    Field,
    Term,
    EditDistance,
    Boost,
    Presence,
}

#[derive(Debug, Clone)]
struct Lexeme {
    type_: LexemeType,
    str: String,
    start: usize,
    end: usize,
}

/// Lexer for lunr query syntax.
struct QueryLexer<'a> {
    chars: Vec<char>,
    separators: &'a str,
    lexemes: Vec<Lexeme>,
    pos: usize,
    start: usize,
    escape_positions: Vec<usize>,
}

impl<'a> QueryLexer<'a> {
    fn new(input: &str, separators: &'a str) -> Self {
        Self {
            chars: input.chars().collect(),
            separators,
            lexemes: Vec::new(),
            pos: 0,
            start: 0,
            escape_positions: Vec::new(),
        }
    }

    fn run(&mut self) {
        let mut state = Some(LexState::Text);
        while let Some(s) = state {
            state = match s {
                LexState::Text => self.lex_text(),
                LexState::Field => self.lex_field(),
                LexState::Term => self.lex_term(),
                LexState::EditDistance => self.lex_edit_distance(),
                LexState::Boost => self.lex_boost(),
                LexState::Eos => self.lex_eos(),
            }
        }
    }

    fn slice_string(&mut self) -> String {
        let mut sub_slices: Vec<char> = Vec::new();
        let mut slice_start = self.start;
        // `pos` never exceeds `chars.len()` — see `escape_character` — but clamp
        // defensively so a future lexer state cannot panic on a multibyte query.
        let slice_end = self.pos.min(self.chars.len());
        let slice_start_clamped = slice_start.min(slice_end);

        for &escape_pos in &self.escape_positions {
            if escape_pos < slice_start_clamped || escape_pos >= slice_end {
                continue;
            }
            sub_slices.extend(self.chars[slice_start..escape_pos].iter().copied());
            // Escapes are retained when a later stage must still see them:
            // `\` before `*` marks a literal star, `\` before `\` a literal
            // backslash. Parity counting (`query::has_unescaped_wildcard`)
            // then distinguishes every case. All other escapes are removed.
            let escaped = self.chars.get(escape_pos + 1);
            if escaped == Some(&'*') || escaped == Some(&'\\') {
                sub_slices.push('\\');
            }
            slice_start = escape_pos + 1;
        }
        sub_slices.extend(
            self.chars[slice_start.min(slice_end)..slice_end]
                .iter()
                .copied(),
        );
        self.escape_positions.clear();

        sub_slices.into_iter().collect()
    }

    fn emit(&mut self, type_: LexemeType) {
        let str = self.slice_string();
        self.lexemes.push(Lexeme {
            type_,
            str,
            start: self.start,
            end: self.pos,
        });
        self.start = self.pos;
    }

    fn escape_character(&mut self) {
        // `pos` is already past the `\` (see `next`). If it is at the end there
        // is no escaped character: keep the trailing `\` as a literal instead
        // of advancing past the buffer and panicking in `slice_string`.
        if self.pos >= self.chars.len() {
            return;
        }
        self.escape_positions.push(self.pos - 1);
        self.pos += 1;
    }

    fn next(&mut self) -> Option<char> {
        if self.pos >= self.chars.len() {
            None
        } else {
            let ch = self.chars[self.pos];
            self.pos += 1;
            Some(ch)
        }
    }

    fn width(&self) -> usize {
        self.pos - self.start
    }

    fn ignore(&mut self) {
        if self.start == self.pos {
            self.pos += 1;
        }
        self.start = self.pos;
    }

    fn backup(&mut self) {
        self.pos -= 1;
    }

    fn accept_digit_run(&mut self) {
        while let Some(ch) = self.next() {
            if !ch.is_ascii_digit() {
                self.backup();
                break;
            }
        }
    }

    fn accept_number_run(&mut self) {
        // Boosts are floats (`term^2.5`), not just integers. Accept digits and
        // at most one `.`, plus an optional exponent (`^1e3`); the final
        // `parse::<f64>` still rejects garbage like `^..` with a proper
        // `QueryParseError` instead of silently lexing `^2` or splitting
        // `^1e3` into a boost plus a dead `e3` clause.
        let mut seen_dot = false;
        while let Some(ch) = self.next() {
            if ch.is_ascii_digit() {
                continue;
            }
            if ch == '.' && !seen_dot {
                seen_dot = true;
                continue;
            }
            if ch == 'e' || ch == 'E' {
                self.accept_exponent();
                continue;
            }
            self.backup();
            break;
        }
    }

    /// Try to consume an exponent tail (`e3`, `E-2`) at the current position.
    ///
    /// The caller has already consumed the `e`. Whatever follows becomes part
    /// of the boost lexeme — even garbage: `^1e` lexes as boost `"1e"`, which
    /// `parse` rejects with "boost must be numeric". Failing open instead
    /// (rewinding onto the `e`) would silently split the query into a boost
    /// plus a phantom `e` clause that outranks real matches.
    fn accept_exponent(&mut self) {
        if let Some(sign) = self.next() {
            if sign != '+' && sign != '-' {
                self.backup();
            }
        }
        while let Some(ch) = self.next() {
            if !ch.is_ascii_digit() {
                self.backup();
                break;
            }
        }
    }

    /// Consume trailing word characters into the current lexeme.
    ///
    /// `hello^12xyz` is a malformed boost, not a boost plus a phantom `xyz`
    /// clause that outranks real matches: folding the tail in makes
    /// `parse::<f64>` fail loudly. Stops at anything the main lexer treats
    /// structurally (`+ - : ~ ^` separators and whitespace keep their
    /// meaning), so `hello^12-34` still parses as boost plus prohibition.
    fn accept_word_tail(&mut self) {
        while let Some(ch) = self.next() {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '.' {
                continue;
            }
            self.backup();
            break;
        }
    }

    fn more(&self) -> bool {
        self.pos < self.chars.len()
    }

    fn is_separator(&self, ch: char) -> bool {
        self.separators.contains(ch)
    }

    fn lex_text(&mut self) -> Option<LexState> {
        loop {
            let ch = match self.next() {
                Some(ch) => ch,
                None => return Some(LexState::Eos),
            };

            if ch == '\\' {
                self.escape_character();
                continue;
            }

            if ch == ':' {
                return Some(LexState::Field);
            }

            if ch == '~' {
                self.backup();
                if self.width() > 0 {
                    self.emit(LexemeType::Term);
                }
                return Some(LexState::EditDistance);
            }

            if ch == '^' {
                self.backup();
                if self.width() > 0 {
                    self.emit(LexemeType::Term);
                }
                return Some(LexState::Boost);
            }

            if (ch == '+' || ch == '-') && self.width() == 1 {
                self.emit(LexemeType::Presence);
                return Some(LexState::Text);
            }

            if self.is_separator(ch) {
                return Some(LexState::Term);
            }
        }
    }

    fn lex_field(&mut self) -> Option<LexState> {
        self.backup();
        self.emit(LexemeType::Field);
        self.ignore();
        Some(LexState::Text)
    }

    fn lex_term(&mut self) -> Option<LexState> {
        if self.width() > 1 {
            self.backup();
            self.emit(LexemeType::Term);
        }
        self.ignore();
        if self.more() {
            Some(LexState::Text)
        } else {
            None
        }
    }

    fn lex_edit_distance(&mut self) -> Option<LexState> {
        self.ignore();
        self.accept_digit_run();
        self.accept_word_tail();
        self.emit(LexemeType::EditDistance);
        Some(LexState::Text)
    }

    fn lex_boost(&mut self) -> Option<LexState> {
        self.ignore();
        self.accept_number_run();
        self.accept_word_tail();
        self.emit(LexemeType::Boost);
        Some(LexState::Text)
    }

    fn lex_eos(&mut self) -> Option<LexState> {
        if self.width() > 0 {
            self.emit(LexemeType::Term);
        }
        None
    }
}

#[derive(Debug, Clone, Copy)]
enum LexState {
    Text,
    Field,
    Term,
    EditDistance,
    Boost,
    Eos,
}

/// Parser for lunr query syntax.
pub struct QueryParser<'a> {
    query: &'a mut Query,
    lexemes: Vec<Lexeme>,
    lexeme_idx: usize,
    current_clause: Clause,
    language: LanguageRef,
}

impl<'a> QueryParser<'a> {
    /// Create a parser for `query_string` that will append clauses to `query`.
    ///
    /// `separators` is the set of characters that split query terms and should
    /// match the tokenizer of the target language. `language` selects the
    /// query-side normalization, which must agree with indexing — Turkish is
    /// the case that matters (`I` folds to `ı`, not `i`).
    pub fn new(
        query_string: &str,
        query: &'a mut Query,
        separators: &str,
        language: &LanguageRef,
    ) -> Self {
        let mut lexer = QueryLexer::new(query_string, separators);
        lexer.run();
        Self {
            query,
            lexemes: lexer.lexemes,
            lexeme_idx: 0,
            current_clause: Clause::default(),
            language: language.clone(),
        }
    }

    /// Parse the query string and populate `query.clauses`.
    pub fn parse(mut self) -> Result<(), QueryParseError> {
        let mut state = Some(ParseState::Clause);
        while let Some(s) = state {
            state = match s {
                ParseState::Clause => self.parse_clause()?,
                ParseState::Presence => self.parse_presence()?,
                ParseState::Field => self.parse_field()?,
                ParseState::Term => self.parse_term()?,
                ParseState::EditDistance => self.parse_edit_distance()?,
                ParseState::Boost => self.parse_boost()?,
            }
        }
        Ok(())
    }

    fn peek_lexeme(&self) -> Option<&Lexeme> {
        self.lexemes.get(self.lexeme_idx)
    }

    fn consume_lexeme(&mut self) -> Option<&Lexeme> {
        let lexeme = self.lexemes.get(self.lexeme_idx);
        self.lexeme_idx += 1;
        lexeme
    }

    fn next_clause(&mut self) {
        let clause = std::mem::take(&mut self.current_clause);
        self.query.clause(clause);
    }

    fn parse_clause(&mut self) -> Result<Option<ParseState>, QueryParseError> {
        match self.peek_lexeme() {
            None => Ok(None),
            Some(lexeme) => match lexeme.type_ {
                LexemeType::Presence => Ok(Some(ParseState::Presence)),
                LexemeType::Field => Ok(Some(ParseState::Field)),
                LexemeType::Term => Ok(Some(ParseState::Term)),
                _ => Err(self.error(
                    &format!(
                        "expected either a field or a term, found {:?}",
                        lexeme.type_
                    ),
                    lexeme,
                )),
            },
        }
    }

    fn parse_presence(&mut self) -> Result<Option<ParseState>, QueryParseError> {
        let lexeme = self
            .consume_lexeme()
            .cloned()
            .ok_or_else(|| QueryParseError {
                message: "expected presence operator".to_string(),
                start: 0,
                end: 0,
            })?;

        match lexeme.str.as_str() {
            "-" => self.current_clause.presence = Presence::Prohibited,
            "+" => self.current_clause.presence = Presence::Required,
            _ => {
                return Err(self.error(
                    &format!("unrecognised presence operator '{}'", lexeme.str),
                    &lexeme,
                ))
            }
        }

        let next = self.peek_lexeme().ok_or_else(|| QueryParseError {
            message: "expecting term or field, found nothing".to_string(),
            start: lexeme.start,
            end: lexeme.end,
        })?;

        match next.type_ {
            LexemeType::Field => Ok(Some(ParseState::Field)),
            LexemeType::Term => Ok(Some(ParseState::Term)),
            _ => Err(self.error(
                &format!("expecting term or field, found {:?}", next.type_),
                next,
            )),
        }
    }

    fn parse_field(&mut self) -> Result<Option<ParseState>, QueryParseError> {
        let lexeme = self
            .consume_lexeme()
            .cloned()
            .ok_or_else(|| QueryParseError {
                message: "expected field".to_string(),
                start: 0,
                end: 0,
            })?;

        if !self.query.all_fields.contains(&lexeme.str) {
            let possible = self
                .query
                .all_fields
                .iter()
                .map(|f| format!("'{}'", f))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(self.error(
                &format!(
                    "unrecognised field '{}', possible fields: {}",
                    lexeme.str, possible
                ),
                &lexeme,
            ));
        }

        self.current_clause.fields = vec![lexeme.str.clone()];

        let next = self.peek_lexeme().ok_or_else(|| QueryParseError {
            message: "expecting term, found nothing".to_string(),
            start: lexeme.start,
            end: lexeme.end,
        })?;

        match next.type_ {
            LexemeType::Term => Ok(Some(ParseState::Term)),
            _ => Err(self.error(&format!("expecting term, found {:?}", next.type_), next)),
        }
    }

    fn parse_term(&mut self) -> Result<Option<ParseState>, QueryParseError> {
        let lexeme = self
            .consume_lexeme()
            .cloned()
            .ok_or_else(|| QueryParseError {
                message: "expected term".to_string(),
                start: 0,
                end: 0,
            })?;

        // Normalize with the language's own rules (Turkish folds `I` to `ı`,
        // not `i`) so the query side agrees with what the indexer stored.
        // Multi-language codes containing Turkish skip the global fold: no
        // single folding serves both members, so each member's tokenizer
        // folds for itself downstream (and the bypass paths union both).
        // Normalization leaves `*` — and the `\` retained before an escaped
        // star — untouched, so wildcard patterns survive. Wildcard detection
        // happens centrally in `Query::clause` from the text itself, so there
        // is exactly one rule and no flag to drift from the string.
        self.current_clause.term = if crate::normalize::uses_raw_passthrough(self.language.code()) {
            lexeme.str.clone()
        } else {
            normalize_for_language(self.language.code(), &lexeme.str)
        };

        let Some(next) = self.peek_lexeme() else {
            self.next_clause();
            return Ok(None);
        };

        match next.type_ {
            LexemeType::Term => {
                self.next_clause();
                Ok(Some(ParseState::Term))
            }
            LexemeType::Field => {
                self.next_clause();
                Ok(Some(ParseState::Field))
            }
            LexemeType::EditDistance => Ok(Some(ParseState::EditDistance)),
            LexemeType::Boost => Ok(Some(ParseState::Boost)),
            LexemeType::Presence => {
                self.next_clause();
                Ok(Some(ParseState::Presence))
            }
        }
    }

    fn parse_edit_distance(&mut self) -> Result<Option<ParseState>, QueryParseError> {
        let lexeme = self
            .consume_lexeme()
            .cloned()
            .ok_or_else(|| QueryParseError {
                message: "expected edit distance".to_string(),
                start: 0,
                end: 0,
            })?;

        let distance = lexeme
            .str
            .parse::<usize>()
            .map_err(|_| self.error("edit distance must be numeric", &lexeme))?;
        self.current_clause.edit_distance = Some(distance);

        let Some(next) = self.peek_lexeme() else {
            self.next_clause();
            return Ok(None);
        };

        match next.type_ {
            LexemeType::Term => {
                self.next_clause();
                Ok(Some(ParseState::Term))
            }
            LexemeType::Field => {
                self.next_clause();
                Ok(Some(ParseState::Field))
            }
            LexemeType::EditDistance => Ok(Some(ParseState::EditDistance)),
            LexemeType::Boost => Ok(Some(ParseState::Boost)),
            LexemeType::Presence => {
                self.next_clause();
                Ok(Some(ParseState::Presence))
            }
        }
    }

    fn parse_boost(&mut self) -> Result<Option<ParseState>, QueryParseError> {
        let lexeme = self
            .consume_lexeme()
            .cloned()
            .ok_or_else(|| QueryParseError {
                message: "expected boost".to_string(),
                start: 0,
                end: 0,
            })?;

        let boost = lexeme
            .str
            .parse::<f64>()
            .map_err(|_| self.error("boost must be numeric", &lexeme))?;
        // `1e400` parses to infinity, which would silently become 1.0 in
        // `Query::clause` and score exactly like no boost at all.
        if !boost.is_finite() {
            return Err(self.error("boost must be finite", &lexeme));
        }
        self.current_clause.boost = boost;

        let Some(next) = self.peek_lexeme() else {
            self.next_clause();
            return Ok(None);
        };

        match next.type_ {
            LexemeType::Term => {
                self.next_clause();
                Ok(Some(ParseState::Term))
            }
            LexemeType::Field => {
                self.next_clause();
                Ok(Some(ParseState::Field))
            }
            LexemeType::EditDistance => Ok(Some(ParseState::EditDistance)),
            LexemeType::Boost => Ok(Some(ParseState::Boost)),
            LexemeType::Presence => {
                self.next_clause();
                Ok(Some(ParseState::Presence))
            }
        }
    }

    fn error(&self, message: &str, lexeme: &Lexeme) -> QueryParseError {
        QueryParseError {
            message: message.to_string(),
            start: lexeme.start,
            end: lexeme.end,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum ParseState {
    Clause,
    Presence,
    Field,
    Term,
    EditDistance,
    Boost,
}

/// Parse a query string against the given fields.
///
/// `separators` defines which characters split query terms and should match the
/// language tokenizer (see [`Language::separator_chars`](crate::language::Language)).
/// `language` selects query-side normalization, which must agree with indexing.
pub fn parse_query(
    query_string: &str,
    all_fields: &[String],
    separators: &str,
    language: &LanguageRef,
) -> Result<Query, QueryParseError> {
    let mut query = Query::new(all_fields.to_vec());
    let parser = QueryParser::new(query_string, &mut query, separators, language);
    parser.parse()?;
    Ok(query)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::languages::English;
    use std::sync::Arc;

    fn lang() -> LanguageRef {
        Arc::new(English)
    }

    fn fields() -> Vec<String> {
        vec!["title".to_string(), "body".to_string()]
    }

    fn sep() -> &'static str {
        " \t\n\r\x0C\x0B\x0D\u{00A0}-"
    }

    #[test]
    fn parse_simple_term() {
        let q = parse_query("hello", &fields(), sep(), &lang()).unwrap();
        assert_eq!(q.clauses.len(), 1);
        assert_eq!(q.clauses[0].term, "hello");
        assert_eq!(q.clauses[0].fields, fields());
    }

    #[test]
    fn parse_field_scope() {
        let q = parse_query("title:hello", &fields(), sep(), &lang()).unwrap();
        assert_eq!(q.clauses[0].fields, vec!["title"]);
        assert_eq!(q.clauses[0].term, "hello");
    }

    #[test]
    fn parse_boost_and_edit_distance() {
        let q = parse_query("hello^3~2", &fields(), sep(), &lang()).unwrap();
        assert_eq!(q.clauses[0].boost, 3.0);
        assert_eq!(q.clauses[0].edit_distance, Some(2));
    }

    #[test]
    fn parse_exponent_boost() {
        let q = parse_query("hello^1e3", &fields(), sep(), &lang()).unwrap();
        assert_eq!(q.clauses.len(), 1);
        assert_eq!(q.clauses[0].boost, 1000.0);
    }

    #[test]
    fn reject_infinite_boost() {
        let err = parse_query("hello^1e400", &fields(), sep(), &lang()).unwrap_err();
        assert!(err.message.contains("finite"), "got: {}", err.message);
    }

    #[test]
    fn reject_malformed_exponent_boost() {
        // Fails closed: no phantom `e` clause.
        let err = parse_query("hello^1e", &fields(), sep(), &lang()).unwrap_err();
        assert!(err.message.contains("numeric"), "got: {}", err.message);
    }

    #[test]
    fn parse_presence_modifiers() {
        let q = parse_query("+foo -bar baz", &fields(), sep(), &lang()).unwrap();
        assert_eq!(q.clauses[0].presence, Presence::Required);
        assert_eq!(q.clauses[1].presence, Presence::Prohibited);
        assert_eq!(q.clauses[2].presence, Presence::Optional);
    }

    #[test]
    fn parse_wildcard() {
        let q = parse_query("foo*", &fields(), sep(), &lang()).unwrap();
        assert_eq!(q.clauses[0].term, "foo*");
        assert!(q.clauses[0].has_wildcard);
        assert!(!q.clauses[0].use_pipeline);
    }

    #[test]
    fn escaped_star_is_literal_not_wildcard() {
        let q = parse_query(r"foo\*", &fields(), sep(), &lang()).unwrap();
        assert_eq!(q.clauses.len(), 1);
        assert!(!q.clauses[0].has_wildcard);
        // Bypassed as a literal (like wildcards bypass stemming): the
        // pipeline would trim `foo\*` into a false-positive `foo`.
        assert!(!q.clauses[0].use_pipeline);
    }

    #[test]
    fn trailing_backslash_is_literal() {
        let q = parse_query("foo\\", &fields(), sep(), &lang()).unwrap();
        assert_eq!(q.clauses.len(), 1);
        assert_eq!(q.clauses[0].term, "foo\\");
    }

    #[test]
    fn reject_unknown_field() {
        let err = parse_query("unknown:x", &fields(), sep(), &lang()).unwrap_err();
        assert!(err.message.contains("unrecognised field"));
    }

    #[test]
    fn turkish_query_folds_like_turkish_indexing() {
        // `I` must become `ı`, matching `normalize_tr` at index time — the
        // default fold (`i`) would never meet the indexed term.
        let tr = crate::languages::registry::resolve("tr").language;
        let q = parse_query("Istanbul", &fields(), sep(), &tr).unwrap();
        assert_eq!(q.clauses[0].term, "ıstanbul");
    }
}
