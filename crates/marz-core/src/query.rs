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
            wildcard: Wildcard::None,
            presence: Presence::Optional,
        }
    }
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
        // A negative boost is meaningless — it would subtract from the score
        // and let a matching document rank below a non-matching one. Clamp it.
        //
        // Note that zero is *kept*. lunr writes `clause.boost || 1`, which
        // silently rewrites an explicit `term^0` into `term^1` — the exact
        // opposite of what the user asked for. `Clause::default()` already
        // supplies 1.0 when no boost is given, so there is nothing to default
        // here and an explicit 0 can be honoured.
        clause.boost = clause.boost.max(0.0);

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

        // Wildcards disable the search pipeline.
        if clause.term.contains('*') {
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
