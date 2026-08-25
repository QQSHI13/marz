//! Token set for wildcard and fuzzy expansion.
//!
//! The token set is stored as a trie. It supports exact lookup, `*` wildcard
//! expansion, and edit-distance expansion used by the fuzzy query operator
//! (`~N`), which counts a swap of adjacent characters as a single edit.
//!
//! Fuzzy expansion walks the trie while carrying one row of the edit-distance
//! matrix per node, so the shared prefix of a thousand terms is scored once
//! rather than a thousand times, and a branch whose whole row has run over the
//! edit budget is abandoned without visiting its descendants.

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

    /// Expand a query term by edit distance, counting a transposition as one
    /// edit.
    ///
    /// Returns every indexed term reachable from `term` within `max_edits`
    /// edits, where an edit is an insertion, a deletion, a substitution, or a
    /// swap of two adjacent characters. A `max_edits` of `0` behaves like an
    /// exact lookup.
    ///
    /// # Why transpositions count as one edit
    ///
    /// Plain Levenshtein charges two edits for a swap, because it can only
    /// describe one as a deletion plus an insertion. That is the wrong price for
    /// a search box: transposing adjacent keys is among the most common typing
    /// errors, so under plain Levenshtein `recieve~1` fails to find `receive`
    /// and `teh~1` fails to find `the` — the two cases a user is most likely to
    /// expect to work.
    ///
    /// This is the restricted variant (optimal string alignment): a pair of
    /// characters may be swapped, but nothing else may be edited between them,
    /// so `ca` reaches `abc` in three edits rather than two. The restriction is
    /// what keeps the recurrence a two-row window instead of requiring the whole
    /// matrix, and at the edit budgets a query realistically uses — one or two —
    /// the distinction does not arise.
    ///
    /// # The budget is clamped to below the term's length
    ///
    /// An edit budget that reaches a term's own length makes the query
    /// degenerate: every character can be replaced, so every term of comparable
    /// length matches. On the Japanese corpus, `検索~2` matched 7,291 of 8,461
    /// terms — any two-character term is two substitutions from any other — and
    /// scoring them all took 650 ms to return results that had nothing to do
    /// with the query.
    ///
    /// So `max_edits` is capped at one below the number of characters in `term`,
    /// which is the weakest rule that rules the degenerate case out: at least
    /// one character of the query must survive. This costs nothing on the
    /// word-length terms where fuzzy matching is useful — `keyboard~2` is
    /// unaffected — and turns `検索~2` back into `検索~1`, which is as much
    /// fuzziness as a two-character term can carry and still mean anything.
    pub fn expand_fuzzy(&self, term: &str, max_edits: usize) -> Vec<String> {
        let pattern: Vec<char> = term.chars().collect();

        // At least one character of the query must survive; see above.
        let max_edits = max_edits.min(pattern.len().saturating_sub(1));
        if max_edits == 0 {
            return self.expand(term);
        }

        // Row zero of the edit matrix: turning the empty candidate into the
        // first j characters of the pattern costs j insertions.
        let initial: Vec<usize> = (0..=pattern.len()).collect();

        let mut walk = FuzzyWalk {
            pattern: &pattern,
            max_edits,
            results: Vec::new(),
        };
        walk.visit(&self.root, String::new(), &initial, None);
        walk.results
    }
}

/// State that stays fixed while walking the trie for fuzzy matches.
///
/// Held in a struct rather than threaded through arguments because the
/// transposition rule needs two rows of history plus the character that
/// produced the newer one, and a recursive function taking all of that
/// alongside the pattern and the result sink becomes hard to read.
struct FuzzyWalk<'a> {
    pattern: &'a [char],
    max_edits: usize,
    results: Vec<String>,
}

impl FuzzyWalk<'_> {
    /// Visit one trie node, whose path from the root spells `prefix`.
    ///
    /// `current` is the edit-matrix row for `prefix`, and `previous` is the row
    /// for `prefix` minus its last character, paired with that character. Both
    /// are needed to price a transposition, which compares the candidate's last
    /// two characters against the pattern's and reaches back two rows.
    fn visit(
        &mut self,
        node: &Node,
        prefix: String,
        current: &[usize],
        previous: Option<(&[usize], char)>,
    ) {
        if node.final_ && current[self.pattern.len()] <= self.max_edits {
            self.results.push(prefix.clone());
        }

        // Every entry in a row is at least the minimum of the row above it, and
        // at most one more, so once the whole row exceeds the budget no
        // descendant can come back under it. That holds with transpositions
        // too: a transposition costs one more than an entry two rows up, which
        // by the same bound is no less than the minimum of the row above.
        if current.iter().all(|&d| d > self.max_edits) {
            return;
        }

        for (edge_ch, edge_node) in &node.edges {
            let next = self.next_row(*edge_ch, current, previous);
            let mut new_prefix = prefix.clone();
            new_prefix.push(*edge_ch);
            self.visit(edge_node, new_prefix, &next, Some((current, *edge_ch)));
        }
    }

    /// Edit-matrix row for the candidate extended by `edge_ch`.
    fn next_row(
        &self,
        edge_ch: char,
        current: &[usize],
        previous: Option<(&[usize], char)>,
    ) -> Vec<usize> {
        let mut next = Vec::with_capacity(self.pattern.len() + 1);
        // Matching the empty pattern means deleting the whole candidate.
        next.push(current[0] + 1);

        for j in 1..=self.pattern.len() {
            let cost = usize::from(edge_ch != self.pattern[j - 1]);
            let mut best = (next[j - 1] + 1) // insertion
                .min(current[j] + 1) // deletion
                .min(current[j - 1] + cost); // substitution

            // Transposition: the candidate's last two characters are the
            // pattern's, swapped. `previous` is the row before the swap began.
            if j >= 2 {
                if let Some((before, parent_ch)) = previous {
                    if edge_ch == self.pattern[j - 2] && parent_ch == self.pattern[j - 1] {
                        best = best.min(before[j - 2] + 1);
                    }
                }
            }
            next.push(best);
        }
        next
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

    /// Optimal string alignment distance, computed on the full matrix.
    ///
    /// The trie walk keeps only a two-row window and prunes branches, which is
    /// where a bug would hide. This straightforward version is the oracle.
    fn osa_distance(a: &str, b: &str) -> usize {
        let a: Vec<char> = a.chars().collect();
        let b: Vec<char> = b.chars().collect();
        let mut d = vec![vec![0usize; b.len() + 1]; a.len() + 1];
        // Turning `a`'s first i characters into the empty string costs i
        // deletions, and the empty string into `b`'s first j costs j insertions.
        for (i, row) in d.iter_mut().enumerate() {
            row[0] = i;
        }
        for (j, cell) in d[0].iter_mut().enumerate() {
            *cell = j;
        }
        for i in 1..=a.len() {
            for j in 1..=b.len() {
                let cost = usize::from(a[i - 1] != b[j - 1]);
                d[i][j] = (d[i][j - 1] + 1)
                    .min(d[i - 1][j] + 1)
                    .min(d[i - 1][j - 1] + cost);
                if i >= 2 && j >= 2 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                    d[i][j] = d[i][j].min(d[i - 2][j - 2] + 1);
                }
            }
        }
        d[a.len()][b.len()]
    }

    #[test]
    fn transposition_costs_one_edit_not_two() {
        // The reason for the restricted-Damerau rule. Under plain Levenshtein
        // every one of these needs two edits and none of them would be found.
        for (typo, correct) in [
            ("teh", "the"),
            ("recieve", "receive"),
            ("adn", "and"),
            ("fro", "for"),
            ("cta", "cat"),
        ] {
            assert_eq!(
                osa_distance(typo, correct),
                1,
                "{typo} -> {correct} should be one transposition"
            );

            let mut set = TokenSet::new();
            set.insert(correct);
            assert_eq!(
                set.expand_fuzzy(typo, 1),
                vec![correct.to_string()],
                "{typo}~1 should find {correct}"
            );
        }
    }

    #[test]
    fn a_transposition_at_the_start_is_still_one_edit() {
        // The transposition rule reaches two rows back, so the first pair of
        // characters is the case most likely to be mishandled by an off-by-one.
        let mut set = TokenSet::new();
        set.insert("search");
        assert_eq!(set.expand_fuzzy("esarch", 1), vec!["search".to_string()]);
    }

    #[test]
    fn a_transposition_at_the_end_is_still_one_edit() {
        let mut set = TokenSet::new();
        set.insert("search");
        assert_eq!(set.expand_fuzzy("searhc", 1), vec!["search".to_string()]);
    }

    #[test]
    fn two_separate_transpositions_cost_two_edits() {
        let mut set = TokenSet::new();
        set.insert("balance");
        // "ablacne" swaps ba->ab and nc->cn: two edits, not one.
        assert!(set.expand_fuzzy("ablacne", 1).is_empty());
        assert_eq!(set.expand_fuzzy("ablacne", 2), vec!["balance".to_string()]);
    }

    #[test]
    fn transposition_is_restricted_not_unbounded() {
        // Optimal string alignment forbids editing between a swapped pair, so
        // "ca" -> "abc" is three edits. Unrestricted Damerau-Levenshtein would
        // call it two, and accepting it at ~2 would mean the recurrence is
        // reaching further back than the two-row window can justify.
        //
        // Both strings carry a "zz" prefix only so the query is long enough that
        // the length clamp allows a budget of three at all.
        assert_eq!(osa_distance("zzca", "zzabc"), 3);
        let mut set = TokenSet::new();
        set.insert("zzabc");
        assert!(set.expand_fuzzy("zzca", 2).is_empty());
        assert_eq!(set.expand_fuzzy("zzca", 3), vec!["zzabc".to_string()]);
    }

    /// Every string of length 1..=`max_len` over `alphabet`, sorted.
    fn all_strings(alphabet: &[char], max_len: usize) -> Vec<String> {
        let mut terms = Vec::new();
        let mut frontier = vec![String::new()];
        for _ in 0..max_len {
            let mut next = Vec::new();
            for base in &frontier {
                for ch in alphabet {
                    let mut term = base.clone();
                    term.push(*ch);
                    next.push(term);
                }
            }
            terms.extend(next.iter().cloned());
            frontier = next;
        }
        terms.sort();
        terms
    }

    #[test]
    fn trie_walk_agrees_with_the_full_matrix() {
        // Exhaustive over a small alphabet: every candidate the trie yields must
        // be within budget by the oracle, and every term the oracle says is
        // within budget must be yielded. Pruning bugs show up as the latter.
        //
        // Queries are compared at the clamped budget, since that is what the
        // walk is asked to compute — the clamp itself is tested separately.
        let terms = all_strings(&['a', 'b', 'c'], 4);
        let set = TokenSet::from_sorted(&terms);

        for query in [
            "", "a", "ab", "ba", "abc", "acb", "bacd", "cba", "aabb", "abcab",
        ] {
            for budget in 1..=3 {
                let effective = budget.min(query.chars().count().saturating_sub(1));
                let mut actual = set.expand_fuzzy(query, budget);
                actual.sort();
                let mut expected: Vec<String> = terms
                    .iter()
                    .filter(|t| osa_distance(query, t) <= effective)
                    .cloned()
                    .collect();
                expected.sort();
                assert_eq!(actual, expected, "query {query:?} budget {budget}");
            }
        }
    }

    #[test]
    fn the_edit_budget_cannot_reach_the_term_length() {
        // The degenerate case this rules out: with two edits allowed on a
        // two-character term, both characters can be replaced, so every
        // two-character term in the index matches and the query means nothing.
        // Measured on the Japanese corpus, that was 7,291 of 8,461 terms.
        let mut set = TokenSet::new();
        for term in ["検索", "機械", "学習", "模索", "索検"] {
            set.insert(term);
        }

        // 模索 is one substitution from 検索 and 索検 is one transposition, so
        // both stay reachable. 機械 and 学習 share no character and are exactly
        // two substitutions away — the terms the clamp exists to exclude.
        let mut two = set.expand_fuzzy("検索", 2);
        assert_eq!(
            two.len(),
            set.expand_fuzzy("検索", 1).len(),
            "~2 on a two-character term must be clamped to ~1"
        );
        two.sort();
        assert_eq!(two, ["検索", "模索", "索検"].map(String::from).to_vec());

        // A single character admits no edits at all: at ~1 every one-character
        // term in the index would match.
        assert_eq!(set.expand_fuzzy("検", 1), Vec::<String>::new());
    }

    #[test]
    fn the_clamp_does_not_affect_word_length_terms() {
        // Fuzzy matching is useful on words, and the clamp must not touch them:
        // "keyboard~2" has eight characters and a budget of two.
        let mut set = TokenSet::new();
        set.insert("keyboard");
        assert_eq!(
            set.expand_fuzzy("kaybaord", 2),
            vec!["keyboard".to_string()],
            "two edits on an eight-character term must still be allowed"
        );
    }

    #[test]
    fn fuzzy_matching_works_on_multibyte_terms() {
        // Distances are counted in characters, not bytes: a CJK bigram is one
        // edit from another that shares a character, even though every one of
        // them is three bytes long.
        let mut set = TokenSet::new();
        set.insert("検索");
        set.insert("模索");
        set.insert("機械");
        assert_eq!(osa_distance("索検", "検索"), 1);

        let mut results = set.expand_fuzzy("索検", 1);
        results.sort();
        assert_eq!(results, vec!["検索".to_string()]);

        let mut results = set.expand_fuzzy("検索", 1);
        results.sort();
        assert_eq!(results, vec!["検索".to_string(), "模索".to_string()]);
    }

    #[test]
    fn zero_edits_is_an_exact_lookup() {
        let mut set = TokenSet::new();
        set.insert("the");
        assert_eq!(set.expand_fuzzy("the", 0), vec!["the".to_string()]);
        assert!(set.expand_fuzzy("teh", 0).is_empty());
    }
}
