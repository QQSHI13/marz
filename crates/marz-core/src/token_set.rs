//! Token set for wildcard and fuzzy expansion.
//!
//! The token set is stored as a trie. It supports exact lookup, `*` wildcard
//! expansion, and Levenshtein edit-distance expansion used by the fuzzy
//! query operator (`~N`).

use std::collections::HashMap;

/// A node in the token trie.
#[derive(Default)]
struct Node {
    final_: bool,
    edges: HashMap<char, Node>,
}

/// A set of tokens represented as a trie, used to expand wildcard terms.
#[derive(Default)]
pub struct TokenSet {
    root: Node,
}

impl TokenSet {
    /// Create an empty token set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a token set from a sorted list of terms.
    pub fn from_sorted(terms: &[String]) -> Self {
        let mut set = Self::new();
        for term in terms {
            set.insert(term);
        }
        set
    }

    /// Insert a term into the set.
    pub fn insert(&mut self, term: &str) {
        let mut node = &mut self.root;
        for ch in term.chars() {
            node = node.edges.entry(ch).or_default();
        }
        node.final_ = true;
    }

    /// Expand a query term. Exact terms return themselves if present.
    /// Terms containing `*` return all matching index terms.
    pub fn expand(&self, term: &str) -> Vec<String> {
        if term.contains('*') {
            let mut results = Vec::new();
            self.expand_wildcard(term, &self.root, String::new(), &mut results);
            results
        } else if self.contains(term) {
            vec![term.to_string()]
        } else {
            Vec::new()
        }
    }

    fn contains(&self, term: &str) -> bool {
        let mut node = &self.root;
        for ch in term.chars() {
            match node.edges.get(&ch) {
                Some(n) => node = n,
                None => return false,
            }
        }
        node.final_
    }

    fn expand_wildcard(
        &self,
        pattern: &str,
        node: &Node,
        prefix: String,
        results: &mut Vec<String>,
    ) {
        if pattern.is_empty() {
            if node.final_ {
                results.push(prefix);
            }
            return;
        }

        let mut chars = pattern.chars();
        let ch = chars.next().unwrap();
        let rest: String = chars.collect();

        if ch == '*' {
            // Zero or more characters
            self.expand_wildcard(&rest, node, prefix.clone(), results);
            for (edge_ch, edge_node) in &node.edges {
                let mut new_prefix = prefix.clone();
                new_prefix.push(*edge_ch);
                let new_pattern = format!("*{}", rest);
                self.expand_wildcard(&new_pattern, edge_node, new_prefix, results);
            }
        } else if let Some(edge_node) = node.edges.get(&ch) {
            let mut new_prefix = prefix;
            new_prefix.push(ch);
            self.expand_wildcard(&rest, edge_node, new_prefix, results);
        }
    }

    /// Expand a query term using Levenshtein edit distance.
    ///
    /// Returns all index terms whose edit distance to `term` is less than or
    /// equal to `max_edits`. A `max_edits` of `0` behaves like an exact lookup.
    pub fn expand_fuzzy(&self, term: &str, max_edits: usize) -> Vec<String> {
        if max_edits == 0 {
            return self.expand(term);
        }

        let pattern: Vec<char> = term.chars().collect();
        let pattern_len = pattern.len();

        // Initial distance vector: [0, 1, 2, ..., pattern_len].
        let mut initial = Vec::with_capacity(pattern_len + 1);
        for i in 0..=pattern_len {
            initial.push(i);
        }

        let mut results = Vec::new();
        self.expand_fuzzy_recursive(
            &self.root,
            String::new(),
            &pattern,
            &initial,
            max_edits,
            &mut results,
        );
        results
    }

    fn expand_fuzzy_recursive(
        &self,
        node: &Node,
        prefix: String,
        pattern: &[char],
        current: &[usize],
        max_edits: usize,
        results: &mut Vec<String>,
    ) {
        // If the current node represents a complete term and the distance to the
        // full pattern is within budget, record it.
        if node.final_ && current[pattern.len()] <= max_edits {
            results.push(prefix.clone());
        }

        // Prune when every possible edit distance in this branch is too large.
        if current.iter().all(|&d| d > max_edits) {
            return;
        }

        for (edge_ch, edge_node) in &node.edges {
            let mut next = Vec::with_capacity(pattern.len() + 1);
            // Deletion from the pattern.
            next.push(current[0] + 1);

            for j in 1..=pattern.len() {
                let cost = if *edge_ch == pattern[j - 1] { 0 } else { 1 };
                let insertion = next[j - 1] + 1;
                let deletion = current[j] + 1;
                let substitution = current[j - 1] + cost;
                next.push(insertion.min(deletion).min(substitution));
            }

            let mut new_prefix = prefix.clone();
            new_prefix.push(*edge_ch);
            self.expand_fuzzy_recursive(edge_node, new_prefix, pattern, &next, max_edits, results);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match() {
        let mut set = TokenSet::new();
        set.insert("hello");
        assert_eq!(set.expand("hello"), vec!["hello"]);
        assert!(set.expand("world").is_empty());
    }

    #[test]
    fn trailing_wildcard() {
        let mut set = TokenSet::new();
        set.insert("foo");
        set.insert("foobar");
        set.insert("bar");
        let mut results = set.expand("foo*");
        results.sort();
        assert_eq!(results, vec!["foo", "foobar"]);
    }

    #[test]
    fn leading_wildcard() {
        let mut set = TokenSet::new();
        set.insert("foobar");
        set.insert("barfoo");
        set.insert("baz");
        let mut results = set.expand("*bar");
        results.sort();
        assert_eq!(results, vec!["foobar"]);
    }

    #[test]
    fn fuzzy_expansion() {
        let mut set = TokenSet::new();
        set.insert("hello");
        set.insert("hallo");
        set.insert("help");
        set.insert("world");
        let mut results = set.expand_fuzzy("helo", 1);
        results.sort();
        assert_eq!(results, vec!["hello", "help"]);
    }

    #[test]
    fn fuzzy_expansion_distance_two() {
        let mut set = TokenSet::new();
        set.insert("hello");
        set.insert("hallo");
        set.insert("world");
        let mut results = set.expand_fuzzy("helo", 2);
        results.sort();
        assert_eq!(results, vec!["hallo", "hello"]);
    }
}
