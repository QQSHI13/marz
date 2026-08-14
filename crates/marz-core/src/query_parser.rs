//! Query lexer and parser for lunr query syntax.
//!
//! Supports the same syntax as lunr:
//!
//! * `term` — optional term
//! * `+term` — required term
//! * `-term` — prohibited term
//! * `field:term` — field-scoped term
//! * `term^N` — boost
//! * `term~N` — fuzzy edit distance
//! * backslash escaping for special characters

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
        let slice_end = self.pos;

        for &escape_pos in &self.escape_positions {
            sub_slices.extend(self.chars[slice_start..escape_pos].iter().copied());
            slice_start = escape_pos + 1;
        }
        sub_slices.extend(self.chars[slice_start..slice_end].iter().copied());
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
        self.emit(LexemeType::EditDistance);
        Some(LexState::Text)
    }

    fn lex_boost(&mut self) -> Option<LexState> {
        self.ignore();
        self.accept_digit_run();
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
}

impl<'a> QueryParser<'a> {
    /// Create a parser for `query_string` that will append clauses to `query`.
    ///
    /// `separators` is the set of characters that split query terms and should
    /// match the tokenizer of the target language.
    pub fn new(query_string: &str, query: &'a mut Query, separators: &str) -> Self {
        let mut lexer = QueryLexer::new(query_string, separators);
        lexer.run();
        Self {
            query,
            lexemes: lexer.lexemes,
            lexeme_idx: 0,
            current_clause: Clause::default(),
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

        self.current_clause.term = lexeme.str.to_lowercase();
        if self.current_clause.term.contains('*') {
            self.current_clause.use_pipeline = false;
        }

        let next = self.peek_lexeme();
        if next.is_none() {
            self.next_clause();
            return Ok(None);
        }
        let next = next.unwrap();

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

        let next = self.peek_lexeme();
        if next.is_none() {
            self.next_clause();
            return Ok(None);
        }
        let next = next.unwrap();

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
        self.current_clause.boost = boost;

        let next = self.peek_lexeme();
        if next.is_none() {
            self.next_clause();
            return Ok(None);
        }
        let next = next.unwrap();

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
pub fn parse_query(
    query_string: &str,
    all_fields: &[String],
    separators: &str,
) -> Result<Query, QueryParseError> {
    let mut query = Query::new(all_fields.to_vec());
    let parser = QueryParser::new(query_string, &mut query, separators);
    parser.parse()?;
    Ok(query)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fields() -> Vec<String> {
        vec!["title".to_string(), "body".to_string()]
    }

    fn sep() -> &'static str {
        " \t\n\r\x0C\x0B\x0D\u{00A0}-"
    }

    #[test]
    fn parse_simple_term() {
        let q = parse_query("hello", &fields(), sep()).unwrap();
        assert_eq!(q.clauses.len(), 1);
        assert_eq!(q.clauses[0].term, "hello");
        assert_eq!(q.clauses[0].fields, fields());
    }

    #[test]
    fn parse_field_scope() {
        let q = parse_query("title:hello", &fields(), sep()).unwrap();
        assert_eq!(q.clauses[0].fields, vec!["title"]);
        assert_eq!(q.clauses[0].term, "hello");
    }

    #[test]
    fn parse_boost_and_edit_distance() {
        let q = parse_query("hello^3~2", &fields(), sep()).unwrap();
        assert_eq!(q.clauses[0].boost, 3.0);
        assert_eq!(q.clauses[0].edit_distance, Some(2));
    }

    #[test]
    fn parse_presence_modifiers() {
        let q = parse_query("+foo -bar baz", &fields(), sep()).unwrap();
        assert_eq!(q.clauses[0].presence, Presence::Required);
        assert_eq!(q.clauses[1].presence, Presence::Prohibited);
        assert_eq!(q.clauses[2].presence, Presence::Optional);
    }

    #[test]
    fn parse_wildcard() {
        let q = parse_query("foo*", &fields(), sep()).unwrap();
        assert_eq!(q.clauses[0].term, "foo*");
        assert!(!q.clauses[0].use_pipeline);
    }

    #[test]
    fn reject_unknown_field() {
        let err = parse_query("unknown:x", &fields(), sep()).unwrap_err();
        assert!(err.message.contains("unrecognised field"));
    }
}
