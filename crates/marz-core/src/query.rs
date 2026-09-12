//! Query representation for lunr-compatible searches.

/// Presence constraint for a query clause.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Presence {
    /// The term may appear (default).
    Optional,
    /// The term must appear.
    Required,
    /// The term must not appear.
    Prohibited,
}

/// Automatic wildcard insertion for a query clause.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wildcard {
    /// No wildcard.
    None,
    /// Leading wildcard.
    Leading,
    /// Trailing wildcard.
    Trailing,
    /// Both leading and trailing wildcards.
    Both,
}

/// A single query clause.
#[derive(Debug, Clone)]
pub struct Clause {
    /// The clause term (may contain `*` wildcards after wildcard insertion).
    pub term: String,
    /// Fields this clause applies to. Defaults to all index fields.
    pub fields: Vec<String>,
    /// Boost applied to this clause.
    pub boost: f64,
    /// Optional fuzzy edit distance.
    pub edit_distance: Option<usize>,
    /// Whether to run the search pipeline over the term.
    pub use_pipeline: bool,
    /// Whether the term holds an unescaped `*` wildcard.
    ///
    /// Derived from the term text in [`Query::clause`] — by the single rule
    /// "unescaped `*` counts, `\` + `*` is literal" — never set by hand.
    /// Drives both pipeline disabling and wildcard expansion, so the two can
    /// never disagree about what counts as a wildcard.
    pub has_wildcard: bool,
    /// Automatic wildcard configuration.
    pub wildcard: Wildcard,
    /// Presence constraint.
    pub presence: Presence,
}

impl Default for Clause {
    fn default() -> Self {
        Self {
            term: String::new(),
            fields: Vec::new(),
            boost: 1.0,
            edit_distance: None,
            use_pipeline: true,
            has_wildcard: false,
            wildcard: Wildcard::None,
            presence: Presence::Optional,
        }
    }
}

/// Whether `term` starts with an unescaped `*`, counting backslash parity
/// like [`has_unescaped_wildcard`]: an even run (`\\*`) leaves a real
/// wildcard, an odd run (`\*`) escapes it.
fn starts_with_unescaped_wildcard(term: &str) -> bool {
    let chars: Vec<char> = term.chars().collect();
    let mut backslashes = 0;
    while backslashes < chars.len() && chars[backslashes] == '\\' {
        backslashes += 1;
    }
    chars.get(backslashes) == Some(&'*') && backslashes % 2 == 0
}

/// Whether `term` ends with an unescaped `*`, counting backslash parity:
/// an odd run (`a\*`) escapes it, an even run (`a\\*`) does not.
fn ends_with_unescaped_wildcard(term: &str) -> bool {
    let chars: Vec<char> = term.chars().collect();
    if chars.last() != Some(&'*') {
        return false;
    }
    let mut backslashes = 0;
    for &ch in chars[..chars.len() - 1].iter().rev() {
        if ch != '\\' {
            break;
        }
        backslashes += 1;
    }
    backslashes % 2 == 0
}

/// Whether `term` contains a `*` that is not backslash-escaped.
///
/// Backslashes pair left to right: an even run (`\\`) is literal text and the
/// star after it is a real wildcard; an odd run (`\*`) escapes the star.
/// Post-lexing text upholds this invariant because the lexer retains `\`
/// before both `*` and `\` (see `QueryLexer::slice_string`).
pub(crate) fn has_unescaped_wildcard(term: &str) -> bool {
    let chars: Vec<char> = term.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] != '\\' {
            if chars[i] == '*' {
                return true;
            }
            i += 1;
            continue;
        }
        let mut n = 0;
        while i < chars.len() && chars[i] == '\\' {
            n += 1;
            i += 1;
        }
        if i < chars.len() && chars[i] == '*' {
            if n % 2 == 0 {
                return true;
            }
            i += 1; // Escaped star: literal, skip it.
        }
    }
    false
}

/// Strip escape backslashes (`\*` → `*`, `\\` → `\`, trailing `\` stays).
///
/// Post-lexing text only holds backslashes from escaping, so this inverts the
/// lexer exactly. A star after an even run stays put: it is a real wildcard,
/// not an escape, and consuming it here would corrupt wildcard patterns.
/// Call only on paths without unescaped wildcards.
pub(crate) fn unescape_term(term: &str) -> String {
    if !term.contains('\\') {
        return term.to_string();
    }
    let chars: Vec<char> = term.chars().collect();
    let mut out = String::with_capacity(term.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] != '\\' {
            out.push(chars[i]);
            i += 1;
            continue;
        }
        let mut n = 0;
        while i < chars.len() && chars[i] == '\\' {
            n += 1;
            i += 1;
        }
        if i < chars.len() && chars[i] == '*' {
            out.extend(std::iter::repeat('\\').take(n / 2));
            if n % 2 == 1 {
                // Escaped star: literal text, consume it.
                out.push('*');
                i += 1;
            }
            // Even run: the star is a real wildcard, leave it.
        } else {
            // Literal backslashes (`\\` pairs) and a trailing lone `\`.
            out.extend(std::iter::repeat('\\').take(n.div_ceil(2)));
        }
    }
    out
}

/// A full search query.
#[derive(Debug, Clone)]
pub struct Query {
    /// Query clauses.
    pub clauses: Vec<Clause>,
    /// All fields available in the index.
    pub all_fields: Vec<String>,
}

impl Query {
    /// Create a new query scoped to the given fields.
    pub fn new(all_fields: Vec<String>) -> Self {
        Self {
            clauses: Vec::new(),
            all_fields,
        }
    }

    /// Add a clause, applying lunr defaults for missing fields.
    pub fn clause(&mut self, mut clause: Clause) -> &mut Self {
        if clause.fields.is_empty() {
            clause.fields = self.all_fields.clone();
        }
        // Boost first, even for empty terms below: validation must not depend
        // on which early return fires first.
        // A negative boost is meaningless — it would subtract from the score
        // and let a matching document rank below a non-matching one. Clamp it.
        // Non-finite falls back to 1.0 like field and document boosts: an
        // infinite query boost times a zero BM25 weight is NaN, which sorts
        // randomly. See `index.rs`.
        //
        // Note that zero is *kept*. lunr writes `clause.boost || 1`, which
        // silently rewrites an explicit `term^0` into `term^1` — the exact
        // opposite of what the user asked for. `Clause::default()` already
        // supplies 1.0 when no boost is given, so there is nothing to default
        // here and an explicit 0 can be honoured.
        clause.boost = if clause.boost.is_finite() {
            clause.boost.max(0.0)
        } else {
            1.0
        };
        // An empty term matches nothing. Guard before wildcard insertion:
        // auto-adding `*` would turn it into a match-all.
        if clause.term.is_empty() {
            self.clauses.push(clause);
            return self;
        }

        // Apply automatic wildcards, honoring escapes: a term already ending
        // in an unescaped `*` needs nothing appended, but a trailing escaped
        // star (`a\*`) is literal text that still wants its wildcard.
        if (clause.wildcard == Wildcard::Leading || clause.wildcard == Wildcard::Both)
            && !starts_with_unescaped_wildcard(&clause.term)
        {
            clause.term = format!("*{}", clause.term);
        }
        if (clause.wildcard == Wildcard::Trailing || clause.wildcard == Wildcard::Both)
            && !ends_with_unescaped_wildcard(&clause.term)
        {
            clause.term = format!("{}*", clause.term);
        }

        // A programmatically built term never passes the lexer, so derive the
        // wildcard flag here (the parser sets it from the lexeme instead).
        // Assigned, not OR-ed: a stale `true` with no `*` in the term would
        // force the pipeline off and miss stemmed matches. Escaped stars stay
        // literal: only unescaped `*` counts.
        clause.has_wildcard = has_unescaped_wildcard(&clause.term);

        // Wildcards disable the search pipeline. So do escaped stars: the
        // literal text (`hello` in `hello\*`) must not be stemmed or trimmed
        // into a false positive — expansion looks the raw text up exactly.
        if clause.has_wildcard || clause.term.contains("\\*") {
            clause.use_pipeline = false;
        }

        self.clauses.push(clause);
        self
    }

    /// Add a single term as a clause.
    pub fn term(&mut self, term: impl Into<String>) -> &mut Self {
        self.clause(Clause {
            term: term.into(),
            ..Clause::default()
        });
        self
    }

    /// Returns true if every clause is prohibited (a negated query).
    pub fn is_negated(&self) -> bool {
        !self.clauses.is_empty()
            && self
                .clauses
                .iter()
                .all(|c| c.presence == Presence::Prohibited)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wildcard_detection_counts_backslash_parity() {
        // Bare stars count; odd runs escape; even runs are literal text
        // followed by a real wildcard.
        assert!(has_unescaped_wildcard("foo*"));
        assert!(has_unescaped_wildcard("*foo"));
        assert!(!has_unescaped_wildcard(r"foo\*"));
        assert!(has_unescaped_wildcard(r"foo\\*"));
        assert!(!has_unescaped_wildcard("foo"));
        assert!(!has_unescaped_wildcard("foo\\"));
        assert!(!has_unescaped_wildcard(""));
    }

    #[test]
    fn unescaping_inverts_the_lexer_exactly() {
        assert_eq!(unescape_term("foo"), "foo");
        assert_eq!(unescape_term("foo*"), "foo*");
        assert_eq!(unescape_term(r"foo\*"), "foo*");
        assert_eq!(unescape_term(r"foo\\*"), "foo\\*");
        assert_eq!(unescape_term("foo\\"), "foo\\");
    }

    #[test]
    fn affix_checks_honor_escapes() {
        assert!(starts_with_unescaped_wildcard("*foo"));
        assert!(!starts_with_unescaped_wildcard("\\*foo"));
        assert!(!starts_with_unescaped_wildcard("foo"));
        assert!(ends_with_unescaped_wildcard("foo*"));
        assert!(!ends_with_unescaped_wildcard("foo\\*"));
        assert!(ends_with_unescaped_wildcard("foo\\\\*"));
        assert!(!ends_with_unescaped_wildcard("foo"));
    }

    #[test]
    fn auto_wildcard_still_applies_past_an_escaped_star() {
        // Trailing on a literal-star term must append a real wildcard.
        let mut query = Query::new(vec!["body".to_string()]);
        query.clause(Clause {
            term: "a\\*".to_string(),
            wildcard: Wildcard::Trailing,
            ..Clause::default()
        });
        assert_eq!(query.clauses[0].term, "a\\**");
        assert!(query.clauses[0].has_wildcard);
    }

    #[test]
    fn stale_wildcard_flag_is_cleared() {
        let mut query = Query::new(vec!["body".to_string()]);
        query.clause(Clause {
            term: "plain".to_string(),
            has_wildcard: true,
            ..Clause::default()
        });
        assert!(!query.clauses[0].has_wildcard);
        assert!(query.clauses[0].use_pipeline);
    }
}
