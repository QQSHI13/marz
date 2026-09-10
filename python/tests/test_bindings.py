"""Tests for the Marz Python bindings.

These test the *binding layer*, not the search engine — the engine's own
behaviour is covered by the Rust suite. What can only go wrong here is the
crossing: whether Python values arrive as the right Rust types, whether errors
become the right exceptions, whether offsets can index a Python `str`, and
whether releasing the GIL is safe.

Run with `pytest` after `maturin develop` in this directory.
"""

import threading

import pytest

import marz

# A tiny English corpus. Small enough that expected scores stay stable, and
# distinct enough that each query below has one obvious answer.
DOCS = [
    {"id": "a", "title": "Search engines", "body": "how a search engine works"},
    {"id": "b", "title": "Machine learning", "body": "learning from data"},
    {"id": "c", "title": "Keyboards", "body": "the keyboard is a device"},
]

# Japanese, to exercise the bigram path and multi-byte offsets.
JA_DOCS = [
    {"id": "ja1", "title": "検索エンジン", "body": "検索エンジンは情報を探すシステムです"},
    {"id": "ja2", "title": "機械学習", "body": "機械学習は人工知能の一分野です"},
]


def build(docs=DOCS, language="en", **kwargs):
    """Build an index over `docs` with title boosted, as a real caller would."""
    builder = marz.IndexBuilder(language, **kwargs)
    builder.field("title", 10.0)
    builder.field("body")
    builder.add_many(docs)
    return builder.build()


class TestModule:
    def test_languages_covers_every_family(self):
        codes = marz.languages()
        # Hand-written, Snowball-stemmed, and non-spacing respectively.
        for code in ["en", "zh", "ja", "ko", "de", "ru", "tr", "th", "km", "bo"]:
            assert code in codes, f"{code} missing from languages()"
        assert len(codes) > 40, f"only {len(codes)} codes"
        assert codes == sorted(codes), "languages() should be sorted"

    def test_version_is_exposed(self):
        assert marz.__version__.count(".") == 2

    def test_tokenize_splits_english_into_stemmed_words(self):
        assert marz.tokenize("The running dogs", "en") == ["the", "running", "dogs"]

    def test_tokenize_splits_cjk_into_overlapping_bigrams(self):
        # The whole CJK approach in one assertion: no dictionary, just bigrams.
        # 検索 is a run of 2 so it yields one bigram; エンジン yields three.
        assert marz.tokenize("検索エンジン", "ja") == ["検索", "エン", "ンジ", "ジン"]

    def test_tokenize_segments_thai_without_a_dictionary(self):
        # Thai has no spaces, so this used to come back as a single token and
        # nothing shorter than the whole phrase could match it.
        tokens = marz.tokenize("การค้นหา", "th")
        assert len(tokens) > 2
        # Bigrams are over grapheme clusters, so a term never begins with a
        # combining mark.
        assert "ค้น" in tokens, tokens

    def test_a_snowball_language_stems(self):
        # German plural to singular: the reason stemming exists. Checked through
        # a built index rather than `tokenize`, which is pre-pipeline by design
        # and so reports the split, not the stemmed terms.
        index = build(
            [{"id": "de1", "title": "Suchmaschinen", "body": "wie sie arbeiten"}],
            language="de",
        )
        # The query is the singular; the document only ever says the plural.
        assert [hit.ref for hit in index.search("Suchmaschine")] == ["de1"]

    def test_unknown_language_warns_and_falls_back(self):
        # Not an error: Marz supports 40-odd codes and the world has more, so a
        # code with no stemmer still gets a working whitespace-tokenized index.
        # But it must not be silent, or a typo degrades an index unnoticed.
        with pytest.warns(UserWarning, match='unknown language code "klingon"'):
            assert marz.tokenize("qapla batlh", "klingon") == ["qapla", "batlh"]

    def test_an_unstemmed_real_language_works(self):
        # Vietnamese is what the fallback is *for*, not a typo.
        with pytest.warns(UserWarning):
            assert marz.tokenize("công cụ tìm kiếm", "vi") == [
                "công",
                "cụ",
                "tìm",
                "kiếm",
            ]


class TestBuilder:
    def test_unknown_language_warns_at_construction(self):
        # Warned once, where the mistake was made, rather than at every later
        # call that happens to touch the language.
        with pytest.warns(UserWarning, match='unknown language code "xx"'):
            marz.IndexBuilder("xx")

    def test_multi_warning_names_the_bad_member(self):
        # The working member must not be smeared: name `engish`, not the list.
        with pytest.warns(UserWarning, match="engish in"):
            marz.IndexBuilder("en,engish")

    @pytest.mark.parametrize("name", ["a:b", "a^b", "a~b", "a\\b", "+a", "-a", "a b"])
    def test_operator_field_names_are_rejected(self, name):
        builder = marz.IndexBuilder("en")
        with pytest.raises(ValueError, match="unqueryable|must not"):
            builder.field(name)

    def test_multi_turkish_wildcard_covers_both_folds(self):
        builder = marz.IndexBuilder("en,tr")
        builder.field("body")
        builder.add({"id": "en-doc", "body": "hello world"})
        builder.add({"id": "tr-doc", "body": "Istanbul"})
        index = builder.build()
        assert [h.ref for h in index.search("HELLO*")] == ["en-doc"]
        assert [h.ref for h in index.search("ISTANBUL*")] == ["tr-doc"]

    def test_documents_need_a_field_to_be_indexed_into(self):
        builder = marz.IndexBuilder("en")
        with pytest.raises(ValueError, match="declare at least one field"):
            builder.add({"id": "a"})

    def test_a_field_cannot_be_declared_twice(self):
        builder = marz.IndexBuilder("en")
        builder.field("body")
        with pytest.raises(ValueError, match="already declared"):
            builder.field("body")

    def test_the_reference_field_cannot_also_be_a_search_field(self):
        builder = marz.IndexBuilder("en", ref_field="id")
        with pytest.raises(ValueError, match="is the reference field"):
            builder.field("id")

    def test_a_document_without_a_reference_is_rejected(self):
        builder = marz.IndexBuilder("en")
        builder.field("body")
        with pytest.raises(ValueError, match="missing its reference field"):
            builder.add({"body": "no id here"})

    def test_a_missing_search_field_is_fine(self):
        # Documents are routinely missing an optional field.
        builder = marz.IndexBuilder("en")
        builder.field("title")
        builder.field("body")
        builder.add({"id": "a", "body": "no title on this one"})
        assert builder.build().document_count == 1

    def test_an_explicitly_none_field_is_fine(self):
        builder = marz.IndexBuilder("en")
        builder.field("body")
        builder.add({"id": "a", "body": None})
        assert builder.build().document_count == 1

    @pytest.mark.parametrize("value", [42, {"nested": 1}, ["a"], 1.5, True])
    def test_a_non_string_field_raises_instead_of_being_stringified(self, value):
        # Coercing would index "{'nested': 1}" as searchable text, and the
        # mistake would only surface as a search that mysteriously misses.
        builder = marz.IndexBuilder("en")
        builder.field("body")
        with pytest.raises(TypeError, match="must be a str or None"):
            builder.add({"id": "a", "body": value})

    def test_any_mapping_works_not_just_dict(self):
        class Mapping:
            def __getitem__(self, key):
                return {"id": "x", "body": "mapping protocol"}[key]

        builder = marz.IndexBuilder("en")
        builder.field("body")
        builder.add(Mapping())
        assert [hit.ref for hit in builder.build().search("mapping")] == ["x"]

    def test_add_many_accepts_any_iterable(self):
        builder = marz.IndexBuilder("en")
        builder.field("body")
        builder.add_many({"id": str(i), "body": f"doc {i}"} for i in range(5))
        assert builder.staged == 5

    def test_add_many_is_not_atomic(self):
        # Documented behaviour, asserted so it cannot change silently.
        builder = marz.IndexBuilder("en")
        builder.field("body")
        with pytest.raises(ValueError):
            builder.add_many([{"id": "ok", "body": "fine"}, {"body": "no ref"}])
        assert builder.staged == 1

    def test_building_twice_gives_two_equivalent_indexes(self):
        # A build() that consumed the builder would turn a stray second call
        # into a silently empty index.
        builder = marz.IndexBuilder("en")
        builder.field("body")
        builder.add({"id": "a", "body": "hello"})
        assert builder.build().document_count == 1
        assert builder.build().document_count == 1

    def test_clear_discards_documents_but_keeps_fields(self):
        builder = marz.IndexBuilder("en")
        builder.field("body")
        builder.add({"id": "a", "body": "hello"})
        builder.clear()
        assert builder.staged == 0
        assert builder.fields == ["body"]

    def test_accessors_report_the_configuration(self):
        builder = marz.IndexBuilder("ja", ref_field="location")
        builder.field("title", 10.0)
        builder.field("text")
        assert builder.language == "ja"
        assert builder.ref_field == "location"
        assert builder.fields == ["title", "text"]
        assert "staged=0" in repr(builder)


class TestSearch:
    def test_a_term_finds_the_document_containing_it(self):
        assert [hit.ref for hit in build().search("keyboard")] == ["c"]

    def test_results_are_ordered_by_descending_score(self):
        scores = [hit.score for hit in build().search("search learning")]
        assert scores == sorted(scores, reverse=True)

    def test_a_field_boost_outranks_a_body_match(self):
        builder = marz.IndexBuilder("en")
        builder.field("title", 10.0)
        builder.field("body")
        builder.add({"id": "in_title", "title": "python", "body": "unrelated"})
        builder.add({"id": "in_body", "title": "unrelated", "body": "python"})
        assert [hit.ref for hit in builder.build().search("python")] == [
            "in_title",
            "in_body",
        ]

    def test_a_document_boost_outranks_identical_text(self):
        builder = marz.IndexBuilder("en")
        builder.field("body")
        builder.add({"id": "low", "body": "same words"}, boost=1.0)
        builder.add({"id": "high", "body": "same words"}, boost=5.0)
        assert [hit.ref for hit in builder.build().search("same")] == ["high", "low"]

    def test_no_match_is_an_empty_list_not_an_error(self):
        assert build().search("nonexistentterm") == []

    def test_wildcards_and_fuzzy_and_field_scoping_all_reach_the_engine(self):
        index = build()
        assert [hit.ref for hit in index.search("keyboa*")] == ["c"]
        assert [hit.ref for hit in index.search("keybaord~1")] == ["c"]
        assert [hit.ref for hit in index.search("title:keyboards")] == ["c"]
        assert index.search("+search -learning")

    def test_a_bad_query_raises_query_error_with_a_readable_message(self):
        with pytest.raises(marz.QueryError) as caught:
            build().search("nosuchfield:x")
        # str() must be a sentence, not a formatted tuple.
        assert "unrecognised field" in str(caught.value)
        assert caught.value.query == "nosuchfield:x"
        assert caught.value.start == 0
        assert caught.value.end < len("nosuchfield:x") + 1

    @pytest.mark.parametrize("query", ["", "   ", "\t", "字*", "a" * 200, "*" * 10])
    def test_odd_but_legal_queries_do_not_crash(self, query):
        assert isinstance(build().search(query), list)

    @pytest.mark.parametrize("query", ["+", "-", ":", "field:", "^", "~"])
    def test_bare_operators_raise_rather_than_panicking(self, query):
        with pytest.raises(marz.QueryError):
            build().search(query)


class TestResults:
    def test_a_result_exposes_ref_score_and_terms(self):
        hit = build().search("keyboard")[0]
        assert hit.ref == "c"
        assert hit.score > 0
        assert hit.terms == ["keyboard"]
        assert "ref=" in repr(hit)

    def test_a_result_is_immutable(self):
        hit = build().search("keyboard")[0]
        with pytest.raises(AttributeError):
            hit.score = 99.0

    def test_matches_map_terms_to_fields_to_positions(self):
        hit = build().search("keyboard")[0]
        assert set(hit.matches) == {"keyboard"}
        assert "body" in hit.matches["keyboard"]
        for positions in hit.matches["keyboard"].values():
            assert all(isinstance(p, tuple) and len(p) == 2 for p in positions)

    def test_positions_are_character_offsets_that_index_a_python_str(self):
        # The point of this test: Rust counts bytes by default, and a byte
        # offset into Japanese text would slice mid-character or land in the
        # wrong place entirely.
        text = "これは検索エンジンです"
        builder = marz.IndexBuilder("ja")
        builder.field("body")
        builder.add({"id": "x", "body": text})
        hit = builder.build().search("検索")[0]
        start, length = hit.matches["検索"]["body"][0]
        assert text[start : start + length] == "検索"

    def test_positions_skip_the_punctuation_the_tokenizer_trimmed(self):
        # Two bugs at once, and both produce a highlight that is visibly wrong
        # rather than an exception.
        #
        # The astral prefix is there so a byte offset cannot pass: each 🎉 is
        # four UTF-8 bytes, so "keyboard" starts at character 4 and byte 13.
        # The trailing period is there so the trimmer has something to remove —
        # it must shorten the span rather than leave the term's original width,
        # which would highlight "keyboard." including the period.
        text = "🎉🎉🎉 keyboard. done"
        builder = marz.IndexBuilder("en")
        builder.field("body")
        builder.add({"id": "x", "body": text})
        hit = builder.build().search("keyboard")[0]

        start, length = hit.matches["keyboard"]["body"][0]
        assert (start, length) == (4, 8)
        assert text[start : start + length] == "keyboard"

    def test_leading_punctuation_shifts_the_position_forward(self):
        # The other end of the same trim. A term the tokenizer stripped a quote
        # off of must report where the *word* starts, not where the quote does.
        text = 'He said "keyboard" loudly'
        builder = marz.IndexBuilder("en")
        builder.field("body")
        builder.add({"id": "x", "body": text})
        hit = builder.build().search("keyboard")[0]

        start, length = hit.matches["keyboard"]["body"][0]
        assert text[start : start + length] == "keyboard"

    def test_only_positions_are_lost_when_positions_are_dropped(self):
        # A positionless index still reports which terms matched and which
        # fields they matched in — just not where. So a caller can still show
        # "matched in title" without being able to highlight the span.
        builder = marz.IndexBuilder("en")
        builder.field("title", 10.0)
        builder.field("body")
        builder.add({"id": "c", "title": "Keyboards", "body": "a device"})
        index = builder.build()

        full = marz.Index.from_bytes(index.to_bytes()).search("device")[0]
        assert full.matches == {"devic": {"body": [(2, 6)]}}

        thin = marz.Index.from_bytes(index.to_bytes(positions=False))
        hit = thin.search("device")[0]
        assert hit.ref == "c"
        assert hit.terms == ["devic"]
        # Same shape, same field key — the position list is what is empty.
        assert hit.matches == {"devic": {"body": []}}


class TestSerialization:
    def test_binary_roundtrip_preserves_refs_and_scores(self):
        original = build()
        restored = marz.Index.from_bytes(original.to_bytes())
        before = [(h.ref, h.score) for h in original.search("search engine")]
        after = [(h.ref, h.score) for h in restored.search("search engine")]
        assert before == after

    def test_dropping_positions_shrinks_the_index(self):
        index = build(JA_DOCS, "ja")
        assert len(index.to_bytes(positions=False)) < len(index.to_bytes())

    def test_to_bytes_returns_bytes(self):
        assert isinstance(build().to_bytes(), bytes)

    def test_index_language_reads_the_code_without_loading(self):
        assert marz.index_language(build(JA_DOCS, "ja").to_bytes()) == "ja"

    def test_from_bytes_infers_the_language(self):
        assert marz.Index.from_bytes(build(JA_DOCS, "ja").to_bytes()).language == "ja"

    def test_from_bytes_rejects_a_language_that_disagrees(self):
        # Loading a Japanese index as English would tokenize queries by rules
        # that cannot match the indexed bigrams — no results, no error.
        data = build(JA_DOCS, "ja").to_bytes()
        with pytest.raises(ValueError, match='built for language "ja"'):
            marz.Index.from_bytes(data, "en")

    def test_from_bytes_accepts_a_language_that_agrees(self):
        data = build(JA_DOCS, "ja").to_bytes()
        assert marz.Index.from_bytes(data, "ja").document_count == 2

    @pytest.mark.parametrize(
        "data", [b"", b"not an index", b"MARZ", b"\x00" * 64, bytes(range(64))]
    )
    def test_garbage_bytes_raise_format_error(self, data):
        with pytest.raises(marz.FormatError):
            marz.Index.from_bytes(data)

    def test_a_truncated_index_raises_format_error(self):
        data = build().to_bytes()
        for cut in (10, 40, 63, len(data) // 2, len(data) - 1):
            with pytest.raises(marz.FormatError):
                marz.Index.from_bytes(data[:cut])

    def test_format_error_is_a_value_error(self):
        assert issubclass(marz.FormatError, ValueError)


class TestIndexAccessors:
    def test_accessors_describe_the_index(self):
        index = build()
        assert index.fields == ["title", "body"]
        assert index.document_count == 3
        assert index.term_count > 0
        assert index.language == "en"
        assert len(index) == 3
        assert "documents=3" in repr(index)


class TestBm25Parameters:
    def test_k1_changes_the_scores_it_is_supposed_to_change(self):
        # k1 controls term-frequency saturation, so raising it must widen the
        # gap between a document with many occurrences and one with few.
        def gap(k1):
            builder = marz.IndexBuilder("en", k1=k1)
            builder.field("body")
            builder.add({"id": "many", "body": "y " * 10})
            builder.add({"id": "one", "body": "y and other words here"})
            scores = {h.ref: h.score for h in builder.build().search("y")}
            return scores["many"] - scores["one"]

        assert gap(0.5) < gap(1.2) < gap(3.0)

    def test_b_is_clamped_rather_than_rejected(self):
        # The engine clamps to [0, 1]; check the binding does not reject first.
        for value in (-1.0, 0.0, 0.5, 1.0, 2.0):
            index = build(b=value)
            assert index.search("keyboard")


class TestConcurrency:
    def test_searching_from_many_threads_is_safe_and_parallel(self):
        # The index is immutable and search releases the GIL, so this must not
        # crash, deadlock, or return inconsistent results.
        index = build(JA_DOCS, "ja")
        expected = [(h.ref, h.score) for h in index.search("検索")]
        results = []
        errors = []

        def worker():
            try:
                for _ in range(50):
                    results.append([(h.ref, h.score) for h in index.search("検索")])
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors
        assert len(results) == 200
        assert all(result == expected for result in results)

    def test_building_from_many_threads_is_safe(self):
        errors = []

        def worker(n):
            try:
                builder = marz.IndexBuilder("en")
                builder.field("body")
                builder.add({"id": str(n), "body": f"document {n}"})
                assert builder.build().document_count == 1
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors


class TestTypeStubs:
    """The stubs are hand-written, so nothing but a test keeps them honest."""

    @staticmethod
    def _stub_members():
        import ast
        from pathlib import Path

        source = Path(marz.__file__).with_name("_marz.pyi")
        if not source.exists():  # pragma: no cover
            pytest.skip("stubs not installed alongside the extension")
        tree = ast.parse(source.read_text(encoding="utf-8"))

        module: set[str] = set()
        classes: dict[str, set[str]] = {}
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                classes[node.name] = {
                    child.name
                    for child in node.body
                    if isinstance(child, ast.FunctionDef)
                } | {
                    child.target.id
                    for child in node.body
                    if isinstance(child, ast.AnnAssign)
                    and isinstance(child.target, ast.Name)
                }
            elif isinstance(node, ast.FunctionDef):
                module.add(node.name)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                module.add(node.target.id)
        return module, classes

    def test_the_stub_declares_every_public_name(self):
        module, classes = self._stub_members()
        declared = module | set(classes)
        exported = set(marz.__all__)
        assert exported - declared == set(), "exported but not in the stub"
        assert declared - exported == set(), "in the stub but not exported"

    @pytest.mark.parametrize("name", ["Index", "IndexBuilder", "Result"])
    def test_the_stub_declares_every_member_of_each_class(self, name):
        _, classes = self._stub_members()
        cls = getattr(marz, name)
        real = {member for member in dir(cls) if not member.startswith("__")}
        if hasattr(cls, "__len__"):
            real.add("__len__")
        # __init__ appears in the stub as the constructor signature; PyO3 spells
        # it #[new], so it is not in dir() as a distinct member to compare.
        stub = classes[name] - {"__init__"}
        assert stub == real

    def test_both_exceptions_derive_from_value_error(self):
        # Callers should be able to catch ValueError without importing ours.
        assert issubclass(marz.QueryError, ValueError)
        assert issubclass(marz.FormatError, ValueError)

    def test_stub_signatures_match_runtime_defaults(self):
        # Names alone would let a changed default lie to type-checkers: pin
        # the defaults structually (PyO3 renders no annotations here).
        import inspect

        params = inspect.signature(marz.normalize).parameters
        assert list(params) == ["text", "language"]
        assert params["language"].default == "en"

        params = inspect.signature(marz.Index.to_bytes).parameters
        assert params["positions"].kind == inspect.Parameter.KEYWORD_ONLY
        assert params["positions"].default is True

        params = inspect.signature(marz.Index.from_bytes).parameters
        assert params["language"].default is None

        assert inspect.signature(marz.IndexBuilder.field).parameters["boost"].default == 1.0
        assert inspect.signature(marz.IndexBuilder.add).parameters["boost"].default == 1.0


class TestCjk:
    def test_a_cjk_query_finds_the_document_without_a_dictionary(self):
        index = build(JA_DOCS, "ja")
        assert [hit.ref for hit in index.search("検索")] == ["ja1"]
        assert [hit.ref for hit in index.search("機械学習")] == ["ja2"]

    def test_a_cjk_phrase_outranks_its_scattered_bigrams(self):
        # Phrase verification: 検索エンジン as a contiguous string should beat a
        # document that merely contains the same bigrams apart.
        builder = marz.IndexBuilder("ja")
        builder.field("body")
        builder.add({"id": "phrase", "body": "検索エンジンについて"})
        builder.add({"id": "scattered", "body": "エンジンの話と検索の話"})
        hits = [h.ref for h in builder.build().search("検索エンジン")]
        assert hits[0] == "phrase"

    @pytest.mark.parametrize(
        ("language", "text", "query"),
        [
            ("zh", "搜索引擎是什么", "搜索"),
            ("ja", "検索エンジンとは", "検索"),
            ("ko", "검색 엔진이란 무엇인가", "검색"),
        ],
    )
    def test_each_cjk_language_indexes_and_searches(self, language, text, query):
        builder = marz.IndexBuilder(language)
        builder.field("body")
        builder.add({"id": "x", "body": text})
        assert [hit.ref for hit in builder.build().search(query)] == ["x"]

    def test_mixed_script_text_is_searchable_from_either_script(self):
        builder = marz.IndexBuilder("ja")
        builder.field("body")
        builder.add({"id": "x", "body": "PythonでMLを検索する"})
        index = builder.build()
        assert [hit.ref for hit in index.search("python")] == ["x"]
        assert [hit.ref for hit in index.search("検索")] == ["x"]

    def test_turkish_queries_fold_like_turkish_indexing(self):
        # `I` folds to `ı`, not `i`: every casing of the query must meet the
        # indexed term.
        builder = marz.IndexBuilder("tr")
        builder.field("body")
        builder.add({"id": "x", "body": "Istanbul"})
        index = builder.build()
        for query in ["Istanbul", "ISTANBUL", "ıstanbul"]:
            assert [hit.ref for hit in index.search(query)] == ["x"]
        assert marz.normalize("Istanbul", "tr") == "ıstanbul"
        assert marz.normalize("Istanbul") == "istanbul"


class TestMultiLanguage:
    def test_comma_code_builds_searches_and_roundtrips(self):
        builder = marz.IndexBuilder("en,ja", ref_field="location")
        assert builder.language == "en,ja"
        builder.field("title", 10.0)
        builder.field("text")
        builder.add({"location": "en/", "title": "Search engine", "text": "offline"})
        builder.add({"location": "ja/", "title": "検索エンジン", "text": "テスト"})
        index = builder.build()
        assert [h.ref for h in index.search("search")] == ["en/"]
        assert [h.ref for h in index.search("検索エンジン")] == ["ja/"]
        data = index.to_bytes()
        assert marz.index_language(data) == "en,ja"
        reloaded = marz.Index.from_bytes(data)
        assert reloaded.language == "en,ja"
        assert [h.ref for h in reloaded.search("検索エンジン")] == ["ja/"]

    def test_member_order_and_spacing_are_canonical(self):
        builder = marz.IndexBuilder("en,ja")
        builder.field("body")
        builder.add({"id": "x", "body": "hello"})
        data = builder.build().to_bytes()
        # Spacing is formatting, not identity; order affects stemming.
        assert marz.Index.from_bytes(data, "en, ja").document_count == 1
        with pytest.raises(marz.FormatError):
            marz.Index.from_bytes(data, "ja,en")

    def test_trailing_comma_behaves_like_single_code(self):
        assert marz.IndexBuilder("en,").language == "en"


class TestNormalize:
    def test_normalize_folds_width_and_case(self):
        assert marz.normalize("ＲＵＳＴ") == "rust"
        assert marz.normalize("ｶﾞｲﾄﾞ") == "ガイド"

    def test_normalize_warns_on_unknown_language(self):
        with pytest.warns(UserWarning, match="klingon"):
            assert marz.normalize("Hello", "klingon") == "hello"

    def test_match_offsets_index_the_normalized_text(self):
        # Half-width katakana composes, shifting every later offset: highlight
        # marz.normalize(text), never the raw input.
        builder = marz.IndexBuilder("ja")
        builder.field("body")
        builder.add({"id": "x", "body": "ｶﾞｲﾄﾞの案内"})
        hit = builder.build().search("ガイド")[0]
        normalized = marz.normalize("ｶﾞｲﾄﾞの案内")
        for fields in hit.matches.values():
            for start, length in fields["body"]:
                assert normalized[start : start + length]

    def test_surrounding_whitespace_in_language_code_is_ignored(self):
        assert marz.IndexBuilder("  en  ").language == "en"
