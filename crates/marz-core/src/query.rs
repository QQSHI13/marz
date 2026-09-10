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

/// Whether `term` contains a `*` that is not backslash-escaped.
///
/// The query lexer retains the backslash before an escaped star (see
/// `QueryLexer::slice_string`), so post-lexing text still carries the
/// distinction — except for a star escaped by an escaped backslash
/// (`\\*`), which lexes as a bare star anyway and arrives here already
/// resolved.
pub(crate) fn has_unescaped_wildcard(term: &str) -> bool {
    let mut escaped = false;
    for ch in term.chars() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' {
            escaped = true;
            continue;
        }
        if ch == '*' {
            return true;
        }
    }
    false
}

/// Strip the retained backslashes before escaped stars (`\*` → `*`).
///
/// Post-lexing text only ever holds backslashes in that position (every other
/// escape is removed by the lexer), so this is exact — and a no-op for terms
/// without escapes.
pub(crate) fn unescape_term(term: &str) -> String {
    if !term.contains('\\') {
        return term.to_string();
    }
    let mut out = String::with_capacity(term.len());
    let mut chars = term.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\\' && chars.peek() == Some(&'*') {
            continue;
        }
        out.push(ch);
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
        // An empty term matches nothing. Guard before wildcard insertion:
        // auto-adding `*` would turn it into a match-all.
        if clause.term.is_empty() {
            self.clauses.push(clause);
            return self;
        }
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

        // Apply automatic wildcards.
        if (clause.wildcard == Wildcard::Leading || clause.wildcard == Wildcard::Both)
            && !clause.term.starts_with('*')
        {
            clause.term = format!("*{}", clause.term);
        }
        if (clause.wildcard == Wildcard::Trailing || clause.wildcard == Wildcard::Both)
            && !clause.term.ends_with('*')
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
