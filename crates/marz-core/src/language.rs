//! Language configuration and registry.
//!
//! Marz treats i18n as a core feature. Every locale implements the same
//! [`Language`] trait used by English, so no language is a second-class add-on.

use std::sync::Arc;

use crate::token::Token;

/// Language-specific text processing.
///
/// All languages — including English — implement this trait. The tokenizer,
/// trimmer, stop-word filter, and stemmer are called by the generic pipeline.
pub trait Language: Send + Sync {
    /// ISO-style language code, e.g. "en" or "zh".
    fn code(&self) -> &str;

    /// Tokenize text into tokens.
    fn tokenize(&self, text: &str) -> Vec<Token>;

    /// Trim non-word characters from a token. Return `false` to drop it.
    fn trim(&self, token: &mut Token) -> bool;

    /// Return `true` if the term is a stop word.
    fn is_stop_word(&self, term: &str) -> bool;

    /// Stem a term. Return the term unchanged if no stemming is needed.
    fn stem(&self, term: &str) -> String;

    /// Characters that separate query terms in the query lexer.
    ///
    /// These should match the separators used by [`Language::tokenize`].
    fn separator_chars(&self) -> &str {
        " \t\n\r\x0C\x0B\x0D\u{00A0}"
    }

    /// Labels for the indexing pipeline functions, used for serialization.
    ///
    /// An English pipeline returns `["trimmer", "stopWordFilter", "stemmer"]`.
    /// Languages without these functions return an empty list.
    fn pipeline_labels(&self) -> Vec<&'static str> {
        Vec::new()
    }

    /// Return `true` if `c` belongs to a script this language tokenizes into
    /// overlapping n-grams.
    ///
    /// Query handling needs this. A query term in an n-grammed script expands
    /// into several bigrams that must be *adjacent* in a document to count as a
    /// phrase match, whereas a word-tokenized term stands alone. Without this
    /// hook the query layer cannot tell the two cases apart.
    fn is_ngram_script(&self, _c: char) -> bool {
        false
    }
}

/// A language handle used throughout the engine.
pub type LanguageRef = Arc<dyn Language>;

/// Combine several languages into one configuration.
///
/// This follows the lunr-languages approach: each configured tokenizer runs.
/// Stop words and stemming are combined conservatively (see the method docs):
/// a mixed index cannot know which language a token belongs to, so every rule
/// errs toward keeping searchable text over deleting it.
pub struct MultiLanguage {
    code: String,
    languages: Vec<LanguageRef>,
    /// Union of member separator sets, in first-seen order.
    ///
    /// Stored, not computed per call: separators feed both indexing-time
    /// field validation and query-time lexing, which must agree.
    separators: String,
}

impl MultiLanguage {
    /// Tokenize one union-separator-free `piece`, shifting positions by the
    /// piece's start offset and deduplicating on (term, position).
    fn tokenize_piece(
        &self,
        seen: &mut std::collections::HashSet<(String, Option<(usize, usize)>)>,
        tokens: &mut Vec<Token>,
        piece: &str,
        piece_start: usize,
    ) {
        if piece.is_empty() {
            return;
        }
        for lang in &self.languages {
            for mut token in lang.tokenize(piece) {
                if let Some((start, len)) = token.position().map(|(s, l)| (s + piece_start, l)) {
                    token.metadata.insert(
                        crate::token::POSITION.to_string(),
                        crate::token::TokenMetadata::Pair(start, len),
                    );
                }
                let key = (token.term.clone(), token.position());
                if seen.insert(key) {
                    tokens.push(token);
                }
            }
        }
    }

    /// Create a multi-language configuration.
    ///
    /// The code joins members with `,` — never `-`, which variant codes like
    /// `en-snowball` already contain — so [`crate::languages::resolve_multi`]
    /// can split it back apart when reloading an index.
    ///
    /// Panics in debug builds on an empty member list: no member keeps any
    /// token, so such an index could never match anything.
    pub fn new(languages: Vec<LanguageRef>) -> Self {
        debug_assert!(
            !languages.is_empty(),
            "MultiLanguage needs at least one member language"
        );
        let code = languages
            .iter()
            .map(|l| l.code())
            .collect::<Vec<_>>()
            .join(",");
        let mut separators = String::new();
        for lang in &languages {
            for ch in lang.separator_chars().chars() {
                if !separators.contains(ch) {
                    separators.push(ch);
                }
            }
        }
        Self {
            code,
            languages,
            separators,
        }
    }
}

impl Language for MultiLanguage {
    fn code(&self) -> &str {
        &self.code
    }

    fn tokenize(&self, text: &str) -> Vec<Token> {
        // Split on the union separators FIRST, then run each member tokenizer
        // per piece with rebased positions. Without this, a member whose own
        // set lacks the separator emits terms the query lexer would split —
        // e.g. `built-in` whole from the Japanese side of an `en,ja` index —
        // ghost terms that bloat the index and distort BM25 field lengths,
        // unreachable by any query. Pieces carry no separator, so per-member
        // output needs no further splitting, only shifting.
        //
        // Deduplicate on (term, position) rather than term alone:
        // collapsing repeated words would destroy both the term frequency and
        // the positions that CJK phrase matching depends on.
        let mut seen = std::collections::HashSet::new();
        let mut tokens = Vec::new();
        let mut piece = String::new();
        let mut piece_start = 0usize; // char offset of the piece in `text`
        let mut char_idx = 0usize;
        let mut chars = text.chars().peekable();
        loop {
            match chars.peek() {
                Some(&ch) if self.separators.contains(ch) => {
                    self.tokenize_piece(&mut seen, &mut tokens, &piece, piece_start);
                    piece.clear();
                    chars.next();
                    char_idx += 1;
                    piece_start = char_idx;
                }
                Some(&ch) => {
                    if piece.is_empty() {
                        piece_start = char_idx;
                    }
                    piece.push(ch);
                    chars.next();
                    char_idx += 1;
                }
                None => {
                    self.tokenize_piece(&mut seen, &mut tokens, &piece, piece_start);
                    break;
                }
            }
        }
        // Concatenating per-language runs leaves tokens out of position order,
        // which breaks phrase adjacency and the binary writer's delta coding
        // (its `saturating_sub` would silently encode backward deltas as 0).
        // Sort by start offset; stable so same-offset tokens keep language order.
        tokens.sort_by_key(|t| t.position().map(|(s, _)| s).unwrap_or(usize::MAX));
        // Reassign sequential indices so downstream `index` metadata is dense.
        for (i, token) in tokens.iter_mut().enumerate() {
            if let Some(crate::token::TokenMetadata::Integer(idx)) =
                token.metadata.get_mut(crate::token::INDEX)
            {
                *idx = i;
            }
        }
        tokens
    }

    fn trim(&self, token: &mut Token) -> bool {
        // A token passes the trimmer if any configured language keeps it.
        // Trial on a clone: `trim` rewrites the term in place even when it
        // returns false, so sharing one token across members would let a
        // rejection corrupt the input of the next member.
        for lang in &self.languages {
            let mut trial = token.clone();
            if lang.trim(&mut trial) {
                *token = trial;
                return true;
            }
        }
        false
    }

    fn is_stop_word(&self, term: &str) -> bool {
        // Intersection, not union: a word that is a stop word in one member
        // language may be content in another (`the` as a transliterated brand
        // in Japanese text), and deleting content is the worse failure. Dense
        // stop words cost index size, not correctness — BM25's IDF already
        // scores near-universal terms near zero.
        !self.languages.is_empty() && self.languages.iter().all(|l| l.is_stop_word(term))
    }

    fn stem(&self, term: &str) -> String {
        // First stemmer that changes the term wins. Chaining every stemmer
        // lets German mangle English output (`running` → `run` → …); applying
        // at most one keeps each token in the language that claimed it.
        let result = term.to_string();
        for lang in &self.languages {
            let stemmed = lang.stem(&result);
            if stemmed != result {
                return stemmed;
            }
        }
        result
    }

    fn separator_chars(&self) -> &str {
        // Union of member sets (computed in `new`): validation and lexing
        // must split on the same characters or fields become unqueryable on
        // one side only.
        &self.separators
    }

    fn pipeline_labels(&self) -> Vec<&'static str> {
        // Sorted union of member labels, so the combination is deterministic
        // and `from_binary` can verify it. Member stemmer codes ride along
        // (see `SnowballLanguage`), so a trimmed build fails loudly here.
        let mut labels: Vec<&'static str> = self
            .languages
            .iter()
            .flat_map(|l| l.pipeline_labels())
            .collect();
        labels.sort_unstable();
        labels.dedup();
        labels
    }

    fn is_ngram_script(&self, c: char) -> bool {
        self.languages.iter().any(|l| l.is_ngram_script(c))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::languages::registry;

    fn multi(codes: &str) -> LanguageRef {
        registry::resolve_multi(codes).language
    }

    #[test]
    fn multi_code_joins_with_comma() {
        // Comma, never `-`: variant codes like `en-snowball` already contain
        // dashes, so only a comma splits back apart on reload.
        assert_eq!(multi("en,ja").code(), "en,ja");
        assert_eq!(multi("en").code(), "en");
    }

    #[test]
    fn separators_union_so_lexing_and_indexing_agree() {
        // English splits on `-`, Japanese alone does not: without the union,
        // `built-in` would index whole from one member while queries split it.
        let en_sep = registry::resolve("en")
            .language
            .separator_chars()
            .to_string();
        assert!(en_sep.contains('-'));
        let union = multi("en,ja").separator_chars().to_string();
        assert!(union.contains('-'), "union must keep member separators");
        let terms: Vec<String> = multi("en,ja")
            .tokenize("built-in")
            .into_iter()
            .map(|t| t.term)
            .collect();
        assert!(
            !terms.iter().any(|t| t == "built-in"),
            "ghost whole-hyphen term must not form: {terms:?}"
        );
    }

    #[test]
    #[cfg(feature = "de")]
    fn stemming_stops_at_the_first_stemmer_that_changes() {
        // Chaining would let German mangle English output (`run` → …).
        let ml = multi("en,de");
        assert_eq!(ml.stem("running"), "run");
    }

    #[test]
    fn stop_words_need_every_language_to_agree() {
        use crate::languages::English;
        let en: LanguageRef = Arc::new(English);
        assert!(en.is_stop_word("the"));
        // …but `the` may be content (a brand, a transliteration) in the other
        // member's text, so a mixed index keeps it. Dense stop words cost
        // size, not correctness: IDF already scores them near zero.
        assert!(!multi("en,ja").is_stop_word("the"));
    }

    #[test]
    #[cfg(feature = "de")]
    fn pipeline_labels_union_deterministically() {
        let a = multi("en,de").pipeline_labels();
        let b = multi("de,en").pipeline_labels();
        assert_eq!(a, b, "member order must not change the labels");
        assert!(a.contains(&"de"), "stemmer identity must ride along: {a:?}");
    }
}
