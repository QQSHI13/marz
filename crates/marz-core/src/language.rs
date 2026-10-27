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
    /// piece's start offset in default-normalized space (see `tokenize`).
    fn tokenize_piece_at(
        &self,
        seen: &mut std::collections::HashSet<(String, Option<(usize, usize)>)>,
        tokens: &mut Vec<Token>,
        piece: &str,
        norm_offset: usize,
    ) {
        if piece.is_empty() {
            return;
        }
        // Each member's output is collected before anything is emitted, so a
        // whole-run ghost can be judged against the finer tokens another member
        // produced for the same span. Judging per member as it went would mean
        // the verdict depended on member order.
        let mut per_member: Vec<Vec<Token>> = Vec::with_capacity(self.languages.len());
        for lang in &self.languages {
            let mut mine = Vec::new();
            for mut token in lang.tokenize(piece) {
                // Drop ghost mixed-script terms: e.g. the English side of an
                // `en,ja` index emits `hello検索` whole for a piece the
                // Japanese side splits into `hello` + `検索`. The ghost is
                // unreachable by any query and inflates field lengths/BM25.
                if crate::languages::cjk::is_mixed_script(&token.term) {
                    continue;
                }
                if let Some((start, len)) = token.position().map(|(s, l)| (s + norm_offset, l)) {
                    token.metadata.insert(
                        crate::token::POSITION.to_string(),
                        crate::token::TokenMetadata::Pair(start, len),
                    );
                }
                mine.push(token);
            }
            per_member.push(mine);
        }

        for (member, mine) in per_member.iter().enumerate() {
            // Whether *this* member bigrams a given script decides whether its
            // tokens for that script can be whole-run ghosts.
            let member_bigrams = |term: &str| {
                term.chars()
                    .next()
                    .is_some_and(|c| self.languages[member].is_ngram_script(c))
            };
            for token in mine {
                // A *word*-tokenizing member emits a whole CJK run as one term
                // (`エンジン`), while a bigramming member splits it (`エン`,
                // `ンジ`, `ジン`). The whole run is a ghost: no query produces
                // it (the query lexer bigrams too), yet it counts toward the
                // field length and inflates BM25 — on a one-document `en,ja`
                // index it scored the same query 2.304 instead of 2.016.
                //
                // Only tokens from a member that does *not* bigram this script
                // can be ghosts: for a bigramming member these spans *are* its
                // output, so dropping them would delete real terms.
                if !member_bigrams(&token.term) {
                    let covering: Vec<(usize, usize)> = per_member
                        .iter()
                        .enumerate()
                        .filter(|(other, _)| *other != member)
                        .flat_map(|(other, tokens)| {
                            tokens.iter().filter_map(move |t| {
                                let (start, len) = t.position()?;
                                // Only a bigramming member's tokens can tile a
                                // whole run; a sibling word-tokenizer's identical
                                // whole run would otherwise count as cover.
                                let other_bigrams = t
                                    .term
                                    .chars()
                                    .next()
                                    .is_some_and(|c| self.languages[other].is_ngram_script(c));
                                other_bigrams.then_some((start, start + len))
                            })
                        })
                        .collect();
                    if is_ghost_whole_run(token, &covering) {
                        continue;
                    }
                }
                let key = (token.term.clone(), token.position());
                if seen.insert(key) {
                    tokens.push(token.clone());
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

/// Whether a whole-run token is a ghost that finer tokens exactly cover.
///
/// `covering` holds the spans, in the same coordinate space, of tokens another
/// member produced inside the candidate's span. The candidate is a ghost when two
/// or more of them tile it exactly — no gaps, no overhang — because that is the
/// signature of a bigramming tokenizer's decomposition of the same source text.
///
/// Requiring at least two spans and exact tiling is what keeps legitimate cases
/// alive: a lone CJK character, or a run too short to bigram, is emitted whole by
/// both members and must survive.
fn is_ghost_whole_run(token: &Token, covering: &[(usize, usize)]) -> bool {
    let Some((start, len)) = token.position() else {
        return false;
    };
    if len < 2 {
        return false;
    }
    let end = start + len;
    let mut inside: Vec<(usize, usize)> = covering
        .iter()
        .copied()
        .filter(|(s, e)| *s >= start && *e <= end && *e > *s && (*s, *e) != (start, end))
        .collect();
    if inside.len() < 2 {
        return false;
    }
    inside.sort_unstable();
    let mut cursor = start;
    for (s, e) in inside {
        // A gap means the finer tokens are not this run's decomposition.
        if s > cursor {
            break;
        }
        cursor = cursor.max(e);
        if cursor == end {
            return true;
        }
    }
    false
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
        //
        // Offsets are counted in default-normalized space: member tokenizers
        // fold (half-width `ｶﾞ` is 2 chars becoming 1 `ガ`), so raw offsets
        // would overstate every later piece. Highlighting normalizes the same
        // way, so the two agree — except Turkish dotted-capital-`İ`, whose
        // foldings differ in length and are documented on `normalize`.
        let mut seen = std::collections::HashSet::new();
        let mut tokens = Vec::new();
        let mut piece = String::new();
        // Char offset of the piece in default-normalized `text`.
        let mut norm_offset = 0usize;
        let mut chars = text.chars().peekable();
        loop {
            match chars.peek() {
                Some(&ch) if self.separators.contains(ch) => {
                    self.tokenize_piece_at(&mut seen, &mut tokens, &piece, norm_offset);
                    norm_offset += crate::normalize::normalize(&piece).chars().count() + 1;
                    piece.clear();
                    chars.next();
                }
                Some(&ch) => {
                    piece.push(ch);
                    chars.next();
                }
                None => {
                    self.tokenize_piece_at(&mut seen, &mut tokens, &piece, norm_offset);
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
    fn no_whole_run_ghost_for_a_bigrammed_script() {
        // An `en,ja` index runs both tokenizers. The English one splits on
        // whitespace, so it emits the whole katakana run `エンジン` as one term,
        // while the Japanese one emits `エン`, `ンジ`, `ジン`. The whole run is
        // unreachable by any query — the query lexer bigrams too — but it counts
        // toward the field length and inflates BM25.
        let terms: Vec<String> = multi("en,ja")
            .tokenize("検索エンジン")
            .into_iter()
            .map(|t| t.term)
            .collect();
        assert!(
            !terms.iter().any(|t| t == "エンジン"),
            "whole-run ghost survived: {terms:?}"
        );
        assert!(terms.iter().any(|t| t == "エン"), "bigrams must survive");
        assert!(terms.iter().any(|t| t == "検索"), "bigrams must survive");

        let terms: Vec<String> = multi("en,zh")
            .tokenize("中文搜索引擎")
            .into_iter()
            .map(|t| t.term)
            .collect();
        assert!(
            !terms.iter().any(|t| t == "中文搜索引擎"),
            "whole-run ghost survived: {terms:?}"
        );
        assert!(terms.iter().any(|t| t == "中文"));
    }

    #[test]
    fn a_short_or_single_character_run_is_not_a_ghost() {
        // The other end of the rule: a run too short to bigram is emitted whole
        // by every member, and it is legitimately searchable — dropping it would
        // make an isolated character unfindable.
        let terms: Vec<String> = multi("en,ja")
            .tokenize("我 と 検")
            .into_iter()
            .map(|t| t.term)
            .collect();
        assert!(terms.contains(&"我".to_string()), "got {terms:?}");
        assert!(terms.contains(&"検".to_string()), "got {terms:?}");
        assert!(terms.contains(&"と".to_string()), "got {terms:?}");
    }

    #[test]
    fn an_equivalent_multi_language_document_scores_like_a_single_language_one() {
        // The point of removing the ghost: `en,ja` must not inflate field length
        // and rescale BM25 for text that is entirely Japanese.
        use crate::index::IndexBuilder;
        let text = "検索エンジンの仕組み";

        let mut single = IndexBuilder::new(multi("ja"));
        single.ref_field("id").field("body", 1.0);
        single.add("doc", 1.0, |name| match name {
            "body" => Some(text.to_string()),
            _ => None,
        });
        let single = single.build();

        let mut mixed = IndexBuilder::new(multi("en,ja"));
        mixed.ref_field("id").field("body", 1.0);
        mixed.add("doc", 1.0, |name| match name {
            "body" => Some(text.to_string()),
            _ => None,
        });
        let mixed = mixed.build();

        let one = single.search("検索").unwrap();
        let other = mixed.search("検索").unwrap();
        assert_eq!(one.len(), 1);
        assert_eq!(other.len(), 1);
        assert!(
            (one[0].score - other[0].score).abs() < 1e-9,
            "single-language scored {}, multi-language {}",
            one[0].score,
            other[0].score
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
