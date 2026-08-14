//! Token set for wildcard expansion.
//!
//! This is a simplified implementation that supports exact terms and `*`
//! wildcards. Edit-distance (fuzzy) expansion will be added separately.

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

    fn expand_wildcard(&self, pattern: &str, node: &Node, prefix: String, results: &mut Vec<String>) {
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
        } else {
            if let Some(edge_node) = node.edges.get(&ch) {
                let mut new_prefix = prefix;
                new_prefix.push(ch);
                self.expand_wildcard(&rest, edge_node, new_prefix, results);
            }
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
}
