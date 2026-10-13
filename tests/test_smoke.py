"""Fast tests for the eg2 package. No model download required.

Run with:  python -m pytest tests/
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import eg2  # noqa: E402
from eg2 import config, loader, multimodal, output, stats  # noqa: E402


class TestVersionParsing:
    @pytest.mark.parametrize(
        ("version", "expected"),
        [
            ("5.18.0.dev0", (5, 18, 0)),
            ("5.18.0", (5, 18, 0)),
            ("2.6.0+cpu", (2, 6, 0)),
            ("5.18.0+cu130", (5, 18, 0)),
            ("6.1.0rc1", (6, 1, 0)),
            ("not-a-version", (0,)),
        ],
    )
    def test_version_tuple(self, version, expected):
        assert config.version_tuple(version) == expected

    @pytest.mark.parametrize(
        ("version", "minimum", "expected"),
        [
            ("5.18.0", "5.18.0", True),
            ("5.18.0.dev0", "5.18.0", True),
            ("5.17.9", "5.18.0", False),
            ("4.49.0", "5.18.0", False),  # the bug this repo shipped with
            ("10.0.0", "9.0.0", True),  # not a string comparison
        ],
    )
    def test_at_least(self, version, minimum, expected):
        assert config.at_least(version, minimum) is expected


class TestModelConstants:
    def test_model_id(self):
        assert config.MODEL_ID == "google/embeddinggemma-2"

    def test_native_dim_matches_first_mrl_dim(self):
        assert config.MRL_DIMS[0] == config.NATIVE_DIM

    def test_modal_loading_configs(self):
        # Text-only must disable both encoders; full multimodal must disable neither.
        assert config.TEXT_ONLY == {"vision_config": None, "audio_config": None}
        assert config.FULL_MULTIMODAL == {}
        assert config.NO_AUDIO == {"audio_config": None}
        assert config.NO_VISION == {"vision_config": None}


class TestPrecisionGuard:
    """float16 silently breaks this model, so it must be rejected loudly."""

    @pytest.mark.parametrize("bad", ["float16", "fp16", "half", "FLOAT16"])
    def test_float16_rejected(self, bad):
        # Checked before torch is imported, so this works with or without torch.
        with pytest.raises(ValueError, match="float16"):
            loader.resolve_precision(bad)

    def test_float32_accepted_when_torch_present(self):
        torch = pytest.importorskip("torch")
        assert loader.resolve_precision("float32") is torch.float32

    def test_unknown_modality_rejected(self):
        # Argument validation must precede any heavy import.
        with pytest.raises(ValueError, match="Unknown modalities"):
            loader.load_model(modalities="not-a-modality")


class TestLazyImports:
    def test_import_eg2_does_not_require_torch(self):
        """The doctor must run on a machine with no third-party packages."""
        code = "import sys, eg2; print('torch' in sys.modules)"
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env={"PYTHONPATH": str(ROOT / "src"), "PATH": ""},
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "False"

    def test_unknown_attribute_raises_attribute_error(self):
        with pytest.raises(AttributeError):
            eg2.definitely_not_an_export


class TestDoctor:
    def test_report_runs_without_third_party_packages(self):
        """build_report must never raise, even on a totally empty environment."""
        report = eg2.build_report()
        assert report.checks, "doctor produced no checks"

    def test_report_detects_python_is_supported(self):
        import platform

        report = eg2.build_report()
        python_check = next(c for c in report.checks if c.name == "Python")
        if config.at_least(platform.python_version(), "3.10"):
            assert python_check.status != "fail"

    def test_fix_commands_use_python_dash_m(self):
        """Guards against the `pip`/`python` interpreter mismatch trap."""
        report = eg2.build_report()
        for check in report.failures:
            if check.fix:
                assert "python -m pip" in check.fix or check.fix.endswith("python -m pip")


class TestOutputHelpers:
    def test_print_report_does_not_raise(self, capsys):
        report = eg2.build_report()
        eg2.print_report(report)
        assert capsys.readouterr().out

    def test_table_renders_rows(self, capsys):
        output.table(["a", "b"], [["1", "2"], ["3", "4"]])
        out = capsys.readouterr().out
        assert "a" in out and "4" in out

    def test_kv_handles_empty(self, capsys):
        output.kv([])
        assert capsys.readouterr().out.strip() == ""


class TestLessonsCompile:
    @pytest.mark.parametrize(
        "lesson",
        sorted(p.name for p in (ROOT / "lessons").glob("L*.py")),
    )
    def test_lesson_is_valid_python(self, lesson):
        import warnings

        source = (ROOT / "lessons" / lesson).read_text(encoding="utf-8")
        # Invalid escape sequences are DeprecationWarning today and SyntaxWarning
        # in a future Python. Docstrings full of ASCII tables trip them, so treat
        # them as errors here rather than discovering them on a newer runtime.
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            warnings.simplefilter("error", SyntaxWarning)
            compile(source, lesson, "exec")


class TestCosine:
    """Cosine is the whole basis of comparison, so pin its behaviour."""

    def test_identical_vectors_score_one(self):
        v = [1.0, 2.0, 3.0]
        assert eg2.cosine(v, v) == pytest.approx(1.0)

    def test_perpendicular_scores_zero(self):
        assert eg2.cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_scores_minus_one(self):
        assert eg2.cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_length_is_ignored(self):
        """The defining property of cosine."""
        a = [1.0, 2.0, 3.0]
        assert eg2.cosine(a, [x * 10 for x in a]) == pytest.approx(1.0)
        assert eg2.cosine(a, [x * 0.01 for x in a]) == pytest.approx(1.0)

    def test_dot_product_is_not_length_invariant(self):
        a = [1.0, 2.0, 3.0]
        assert eg2.dot_product(a, a) != pytest.approx(eg2.dot_product(a, [x * 10 for x in a]))

    def test_zero_vector_does_not_crash(self):
        assert eg2.cosine([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ValueError, match="length"):
            eg2.cosine([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_matrix_diagonal_is_one(self):
        """The first thing beginners assume is a bug."""
        rows = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        matrix = eg2.cosine_matrix(rows, rows)
        for i in range(len(rows)):
            assert matrix[i][i] == pytest.approx(1.0)


class TestDocumentFormatting:
    """Regression guards for the double-prefix bug (§6.1 of PLAN.md)."""

    def test_document_prompt_is_not_empty(self):
        # This is the fact the whole bug turns on. If a future checkpoint
        # changes it, every "just pass Document with a title" assumption in
        # this repo must be revisited.
        assert eg2.PROMPTS["Document"] == "title: none | text: "

    def test_format_document_produces_expected_shape(self):
        assert eg2.format_document("My Title", "body") == "title: My Title | text: body"

    def test_format_document_handles_missing_title(self):
        assert eg2.format_document(None, "body") == "title: none | text: body"
        assert eg2.format_document("", "body") == "title: none | text: body"

    def test_untitled_document_gets_exactly_one_prefix(self):
        result = eg2.preview_input("body", prompt_name="Document", show_prompt=False)
        assert result == "title: none | text: body"
        assert result.count("title:") == 1
        assert result.count("text:") == 1

    def test_titled_document_is_not_double_prefixed(self):
        """The bug, pinned. These two must never be combined."""
        formatted = eg2.format_document("My Title", "body")
        result = eg2.preview_input(formatted, prompt_name=None, show_prompt=False)
        assert result == "title: My Title | text: body"
        assert result.count("title:") == 1

    def test_the_bug_would_be_visible(self):
        """Assert the failure mode still exists, so we know what we avoid."""
        formatted = eg2.format_document("My Title", "body")
        broken = eg2.preview_input(formatted, prompt_name="Document", show_prompt=False)
        assert broken == "title: none | text: title: My Title | text: body"
        assert broken.count("title:") == 2

    def test_unknown_prompt_name_lists_known_names(self):
        with pytest.raises(KeyError, match="Known names"):
            eg2.resolve_prompt("NotARealPrompt")

    def test_no_prompt_name_returns_text_unchanged(self):
        assert eg2.preview_input("body", prompt_name=None, show_prompt=False) == "body"


def _iter_source_files():
    """Every .py and .ipynb in the repo except this test file and caches."""
    for path in sorted(ROOT.rglob("*")):
        if path.suffix not in {".py", ".ipynb"}:
            continue
        parts = set(path.parts)
        if ".git" in parts or "__pycache__" in parts or "tests" in parts:
            continue
        yield path


def _python_sources(path: Path) -> list[str]:
    """The runnable Python in a file.

    For a notebook that is each code cell's source; for a script, the file
    itself. ast discards docstrings and comments later, which is what we want.
    """
    if path.suffix == ".ipynb":
        notebook = json.loads(path.read_text(encoding="utf-8"))
        return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    return [path.read_text(encoding="utf-8")]


def _parsed_sources(path: Path) -> list[tuple[int, ast.Module]]:
    """Parsed ASTs for the runnable Python in a file.

    Notebook cells may contain IPython magics (`!pip install ...`) which are not
    valid Python, so unparseable sources are skipped rather than failing the run.
    ast discards docstrings and comments, which is exactly what the hygiene
    checks need -- a correct example in a docstring is not a live call.
    """
    parsed = []
    for index, source in enumerate(_python_sources(path)):
        try:
            parsed.append((index, ast.parse(source)))
        except SyntaxError:
            continue
    return parsed


class TestSourceHygiene:
    """Guards against the two bugs fixed in Phase 1 creeping back in.

    These parse the AST rather than grepping text, so a correct example inside
    a docstring -- which is exactly where these two bugs get explained -- is not
    mistaken for a bug.
    """

    def test_no_trust_remote_code_in_live_calls(self):
        """EmbeddingGemma2Model is native to transformers; the flag is a no-op."""
        offenders = []
        for path in _iter_source_files():
            for index, tree in _parsed_sources(path):
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call):
                        continue
                    for keyword in node.keywords:
                        if keyword.arg == "trust_remote_code":
                            offenders.append(f"{path.name} src#{index} line {node.lineno}")
        assert not offenders, f"trust_remote_code reintroduced: {offenders}"

    def test_no_document_prefix_applied_to_pre_titled_text(self):
        """The double-prefix bug from PLAN.md section 6.1.

        A live call that passes a pre-titled collection AND
        prompt_name="Document" produces "title: none | text: title: X | text: Y".
        """
        # Variable names that already carry a manually formatted title.
        pre_titled_names = {"formatted_docs", "formatted_documents", "titled_docs"}

        offenders = []
        for path in _iter_source_files():
            for index, tree in _parsed_sources(path):
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call):
                        continue
                    func = node.func
                    if not (isinstance(func, ast.Attribute) and func.attr == "encode"):
                        continue

                    prompt = None
                    for keyword in node.keywords:
                        if keyword.arg == "prompt_name" and isinstance(keyword.value, ast.Constant):
                            prompt = keyword.value.value
                    if prompt != "Document":
                        continue

                    first_arg = node.args[0] if node.args else None
                    if isinstance(first_arg, ast.Name) and first_arg.id in pre_titled_names:
                        offenders.append(
                            f"{path.name} src#{index} line {node.lineno}: "
                            f"encode({first_arg.id}, prompt_name='Document')"
                        )
        assert not offenders, f"double-prefix reintroduced: {offenders}"

    def test_document_prompt_table_still_declares_the_trap(self):
        """If the Hub checkpoint ever changes this, revisit the lesson above.

        documents.py and L04 are both built around the Document prefix being the
        literal string "title: none | text: ". If that stops being true those
        files need rewriting, not just this assertion.
        """
        assert eg2.PROMPTS["Document"] == "title: none | text: "
        l04 = (ROOT / "lessons" / "L04_whats_the_prefix.py").read_text(encoding="utf-8")
        assert "title: none | text: title: My Title | text: the actual content" in l04


class TestStats:
    def test_vector_norm(self):
        assert eg2.vector_norm([3.0, 4.0]) == pytest.approx(5.0)

    def test_column_means(self):
        assert eg2.column_means([[0.0, 2.0], [2.0, 4.0]]) == pytest.approx([1.0, 3.0])

    def test_best_separating_dimension_finds_obvious_split(self):
        group_a = [[0.0, 1.0], [0.1, 1.0], [0.0, 1.1]]
        group_b = [[1.0, 0.0], [1.1, 0.0], [1.0, 0.1]]
        index, _ = eg2.best_separating_dimension(group_a, group_b)
        assert index == 0

    def test_best_separating_dimension_handles_empty(self):
        assert eg2.column_means([]) == []


class TestRankWithFamilies:
    """Backs the Phase 2 acceptance criterion (L07 measures prefix steering)."""

    def _setup(self):
        code = [[1.0, 0.0], [0.9, 0.1]]
        prose = [[0.0, 1.0], [0.1, 0.9]]
        return code + prose, ["CODE", "CODE", "PROSE", "PROSE"]

    def test_reports_best_score_per_family(self):
        vectors, families = self._setup()
        result = eg2.rank_with_families([1.0, 0.0], vectors, families)

        # Computed rather than hardcoded: cos([1,0], [0,1]) = 0.0 and
        # cos([1,0], [0.1,0.9]) = 0.1104, so the PROSE best is 0.1104.
        assert result["best_per_family"]["CODE"] == pytest.approx(1.0)
        assert result["best_per_family"]["PROSE"] == pytest.approx(eg2.cosine([1.0, 0.0], [0.1, 0.9]))
        assert result["best_family"] == "CODE"

    def test_margin_sign_flips_with_query_direction(self):
        """The whole point of L07: the winner depends on which way you point."""
        vectors, families = self._setup()

        toward_code = eg2.rank_with_families([1.0, 0.0], vectors, families)
        toward_prose = eg2.rank_with_families([0.0, 1.0], vectors, families)

        code_margin = toward_code["best_per_family"]["CODE"] - toward_code["best_per_family"]["PROSE"]
        prose_margin = toward_prose["best_per_family"]["PROSE"] - toward_prose["best_per_family"]["CODE"]

        assert code_margin > 0
        assert prose_margin > 0
        assert toward_code["best_family"] == "CODE"
        assert toward_prose["best_family"] == "PROSE"

    def test_scores_cover_every_document(self):
        vectors, families = self._setup()
        result = eg2.rank_with_families([1.0, 0.0], vectors, families)
        assert len(result["scores"]) == len(vectors)

    def test_rank_orders_best_first(self):
        vectors, families = self._setup()
        result = eg2.rank_with_families([1.0, 0.0], vectors, families, k=2)
        ranked = result["ranked"]
        assert ranked[0][1] >= ranked[1][1]


class TestTokenizationHelpers:
    """These need a real tokenizer, so they are marked slow."""

    @pytest.mark.slow
    def test_token_counts_are_positive(self):
        tokenization = pytest.importorskip("eg2.tokenization")
        tokenizer = tokenization.load_tokenizer()
        assert tokenization.count_tokens(tokenizer, "hello world") > 0
        assert tokenization.count_tokens(tokenizer, "") == 0

    @pytest.mark.slow
    def test_context_fraction_is_bounded(self):
        tokenization = pytest.importorskip("eg2.tokenization")
        tokenizer = tokenization.load_tokenizer()
        used, fraction = tokenization.fit_to_context(tokenizer, "short text")
        assert used > 0
        assert 0.0 < fraction < 1.0

    @pytest.mark.slow
    def test_show_tokens_returns_pairs(self):
        tokenization = pytest.importorskip("eg2.tokenization")
        tokenizer = tokenization.load_tokenizer()
        pairs = tokenization.show_tokens(tokenizer, "reverse a list", show=False)
        assert pairs and all(isinstance(t, str) and isinstance(i, int) for t, i in pairs)


class TestPhase2Acceptance:
    """PLAN.md Phase 2: L07 must print a measurable delta for all three choices."""

    def test_l07_covers_all_three_prefix_choices(self):
        l07 = (ROOT / "lessons" / "L07_prefix_matters.py").read_text(encoding="utf-8")
        for label in ("CodeRetrieval", "SearchQuery", "(none)"):
            assert label in l07, f"L07 missing prefix choice {label}"

    def test_l07_compares_against_a_mixed_corpus(self):
        """A same-family corpus could not show steering."""
        l07 = (ROOT / "lessons" / "L07_prefix_matters.py").read_text(encoding="utf-8")
        assert "CODE_DOCS" in l07 and "PROSE_DOCS" in l07
        assert len(eg2.PROMPTS) >= 8, "prompt table should cover the documented names"


class TestRankingStats:
    """Metrics behind lesson L08. Pure Python, so all of these run offline."""

    def test_ranks_assigns_one_to_the_highest(self):
        assert stats.ranks([10.0, 20.0, 30.0]) == [3.0, 2.0, 1.0]

    def test_ranks_averages_ties(self):
        # 9.0 takes rank 1; the tied 5.0s span ranks 2 and 3 and average to 2.5.
        assert stats.ranks([5.0, 5.0, 9.0]) == [2.5, 2.5, 1.0]

    def test_spearman_perfect_correlation(self):
        assert eg2.spearman([1.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.5, 0.9]) == pytest.approx(1.0)

    def test_spearman_perfect_inverse(self):
        assert eg2.spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)

    def test_spearman_ignores_magnitude(self):
        """Rank correlation cares about order, which is the whole point."""
        assert eg2.spearman([1.0, 2.0, 3.0], [100.0, 200.0, 300.0]) == pytest.approx(1.0)

    def test_spearman_is_defined_for_degenerate_input(self):
        assert eg2.spearman([1.0], [2.0]) == 0.0
        assert eg2.spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0

    def test_spearman_rejects_length_mismatch(self):
        with pytest.raises(ValueError):
            eg2.spearman([1.0], [1.0, 2.0])

    def test_topk_overlap(self):
        assert eg2.topk_overlap([0, 1, 2], [0, 1, 2], k=3) == 1.0
        assert eg2.topk_overlap([0, 1, 2], [2, 0, 1], k=3) == 1.0  # reorder is harmless
        assert eg2.topk_overlap([0, 1, 2], [7, 8, 9], k=3) == 0.0

    def test_topk_overlap_partial_swap(self):
        # |{0,1} n {0,1,9}| / |{0,1,2,9} u {0,1,9}| = 2/4
        assert eg2.topk_overlap([0, 1, 2], [0, 1, 9], k=3) == pytest.approx(0.5)

    def test_topk_indices_orders_best_first(self):
        assert eg2.topk_indices([0.1, 0.9, 0.5], k=2) == [1, 2]


class TestStorageReport:
    def test_768d_for_one_million_vectors(self):
        rows = eg2.storage_report([768], count=1_000_000)
        # 768 * 4 * 1e6 bytes, expressed in GiB
        assert rows[0]["gib"] == pytest.approx(768 * 4 * 1_000_000 / (1024**3))

    def test_savings_percentages(self):
        rows = eg2.storage_report([768, 512, 256, 128])
        assert [r["savings_pct"] for r in rows] == pytest.approx([0.0, 33.3333, 66.6667, 83.3333], abs=0.01)

    def test_storage_is_monotonically_smaller(self):
        rows = eg2.storage_report([768, 512, 256, 128])
        sizes = [r["bytes"] for r in rows]
        assert sizes == sorted(sizes, reverse=True)


class TestPhase3Acceptance:
    """PLAN.md Phase 3: L08 must use >=20 documents and >=5 queries."""

    def test_l08_corpus_and_query_counts(self):
        l08 = (ROOT / "lessons" / "L08_matryoshka.py").read_text(encoding="utf-8")
        namespace: dict = {}
        # Execute only the module-level constants, not the lesson body.
        tree = ast.parse(l08)
        wanted = {"CORPUS", "QUERIES", "DIMENSIONS"}
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in wanted for t in node.targets
            ):
                exec(compile(ast.Module(body=[node], type_ignores=[]), "L08", "exec"), namespace)

        assert len(namespace["CORPUS"]) >= 20, f"corpus is {len(namespace['CORPUS'])}, need >= 20"
        assert len(namespace["QUERIES"]) >= 5, f"queries is {len(namespace['QUERIES'])}, need >= 5"
        assert namespace["DIMENSIONS"] == [768, 512, 256, 128]

    def test_truncation_calls_are_all_normalised(self):
        """Regression guard for PLAN.md section 6.2.

        Any encode() call passing truncate_dim without normalize_embeddings=True
        reintroduces the silent ranking bug.
        """
        offenders = []
        for path in _iter_source_files():
            for index, tree in _parsed_sources(path):
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call):
                        continue
                    func = node.func
                    if not (isinstance(func, ast.Attribute) and func.attr == "encode"):
                        continue
                    keywords = {k.arg: k.value for k in node.keywords}
                    if "truncate_dim" not in keywords:
                        continue
                    normalised = keywords.get("normalize_embeddings")
                    is_true = isinstance(normalised, ast.Constant) and normalised.value is True
                    if not is_true:
                        offenders.append(
                            f"{path.name} src#{index} line {node.lineno}: "
                            "truncate_dim without normalize_embeddings=True"
                        )
        assert not offenders, f"section 6.2 reintroduced: {offenders}"


class TestMemoryHelpers:
    """L10's numbers are only worth anything if these work.

    Written because the Windows RSS path silently returned 0 at first: the
    handle from GetCurrentProcess() was narrowed to 32 bits without an explicit
    restype. A reading of zero looks like a valid number, so nothing complained.
    """

    def test_current_rss_is_plausible(self):
        rss = eg2.current_rss_bytes()
        if rss is None:
            pytest.skip("platform does not expose resident set size")
        assert rss > 5 * 1024 * 1024, f"implausibly small RSS: {rss}"

    def test_rss_tracks_a_real_allocation(self):
        from eg2 import memory

        if memory.current_rss_bytes() is None:
            pytest.skip("platform does not expose resident set size")

        before = memory.current_rss_bytes()
        ballast = bytearray(32 * 1024 * 1024)
        # A large allocation is reserved but not committed until written to,
        # so its pages enter the working set only once touched.
        for offset in range(0, len(ballast), 4096):
            ballast[offset] = 1
        after = memory.current_rss_bytes()
        del ballast

        assert (after - before) > 16 * 1024 * 1024, "RSS did not move after committing 32 MiB"

    def test_format_bytes_never_invents_a_number(self):
        assert eg2.format_bytes(None) == "unknown"

    def test_format_bytes(self):
        assert eg2.format_bytes(0) == "0 B"
        assert eg2.format_bytes(512) == "512 B"
        assert eg2.format_bytes(1536) == "1.5 KiB"
        assert eg2.format_bytes(1536 * 1024) == "1.5 MiB"

    def test_parameter_count_of_a_tiny_model(self):
        torch = pytest.importorskip("torch")
        # 4 inputs x 3 outputs weights, plus 3 biases
        assert eg2.parameter_count(torch.nn.Linear(4, 3)) == 15

    def test_empty_cache_is_safe_without_torch(self):
        eg2.empty_cache()


class TestModularLoadingConfig:
    def test_load_configs_cover_every_modality(self):
        keys = [modalities for modalities, _, _ in eg2.LOAD_CONFIGS]
        assert keys == ["text", "image", "audio", "full"]

    def test_every_config_has_a_label(self):
        for modalities, _, advertised in eg2.LOAD_CONFIGS:
            assert modalities in eg2.LOAD_CONFIG_LABELS
            assert advertised.endswith("M")

    def test_config_sizes_increase_with_encoders(self):
        sizes = [int(advertised.rstrip("M")) for _, _, advertised in eg2.LOAD_CONFIGS]
        assert sizes == sorted(sizes)
        assert sizes[0] == 270 and sizes[-1] == 740

    def test_text_only_disables_both_encoders(self):
        text_config = dict(eg2.LOAD_CONFIGS[0][1])
        assert text_config == {"vision_config": None, "audio_config": None}


class TestReadmeCorrection:
    def test_readme_no_longer_claims_191mb(self):
        """L10 replaces a wrong number; keep it from coming back."""
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        assert "191MB" not in readme
        assert "191 MB" not in readme
        assert "191MB" not in (ROOT / "docs" / "08-modular-encoders.md").read_text(encoding="utf-8").replace(
            '"~191MB"', ""
        )


class TestTop1Accuracy:
    """Backs the Phase 5 acceptance criterion (L14 cross-lingual)."""

    def test_perfect(self):
        assert eg2.top1_accuracy([0, 1, 2], [0, 1, 2]) == 1.0

    def test_half(self):
        assert eg2.top1_accuracy([0, 9, 2], [0, 1, 2]) == pytest.approx(2 / 3)

    def test_zero(self):
        assert eg2.top1_accuracy([5, 6], [0, 1]) == 0.0

    def test_empty_is_safe(self):
        assert eg2.top1_accuracy([], []) == 0.0

    def test_length_mismatch_is_safe(self):
        assert eg2.top1_accuracy([0, 1], [0, 1, 2]) == 0.0


class TestMultimodalHelpers:
    """Placeholder syntax and token budgets. No model needed."""

    def test_placeholders_are_the_documented_tokens(self):
        assert multimodal.PLACEHOLDERS == {
            "image": "<|image|>",
            "video": "<|video|>",
            "audio": "<|audio|>",
        }

    def test_interleave_preserves_order(self):
        built = multimodal.interleave([("Shoes.", "text"), ("<|image|>", "image"), ("Grip:", "text")])
        assert built == "Shoes. <|image|> Grip:"

    def test_interleave_repeats_for_multiple_media(self):
        built = multimodal.interleave(
            [("A", "text"), ("<|image|>", "image"), ("B", "text"), ("<|image|>", "image")]
        )
        assert built.count("<|image|>") == 2
        # Order matters: the placeholders must fall between the text segments.
        assert built.index("A") < built.index("<|image|>") < built.index("B")

    def test_interleave_rejects_unknown_modality(self):
        with pytest.raises(ValueError, match="Unknown modality"):
            multimodal.interleave([("x", "telepathy")])

    def test_token_costs_match_the_model_card(self):
        assert multimodal.TOKENS_PER_IMAGE == 280
        assert multimodal.TOKENS_PER_VIDEO_FRAME == 140
        assert multimodal.TOKENS_PER_AUDIO_SECOND == 25
        assert multimodal.CONTEXT_TOKENS == 8_192

    def test_budget_for_single_image(self):
        total, percent = multimodal.budget_for({"image": 1})
        assert total == 280
        assert percent == pytest.approx(280 / 8192 * 100)

    def test_budget_for_audio(self):
        total, _ = multimodal.budget_for({"audio": 4})
        assert total == 100

    def test_budget_accumulates_across_modalities(self):
        total, _ = multimodal.budget_for({"text_tokens": 30, "image": 2, "video": 10, "audio": 5})
        assert total == 30 + 560 + 1400 + 125

    def test_nineteen_images_fit_but_thirty_do_not(self):
        """Confirms the model card's ~29 images figure."""
        fits, _ = multimodal.budget_for({"image": 29})
        overflows, _ = multimodal.budget_for({"image": 30})
        assert fits <= multimodal.CONTEXT_TOKENS
        assert overflows > multimodal.CONTEXT_TOKENS

    def test_audio_requirements_returns_a_tuple(self):
        ok, missing = multimodal.audio_requirements()
        assert isinstance(ok, bool)
        assert isinstance(missing, list)

    def test_ffmpeg_check_does_not_raise(self):
        assert isinstance(multimodal.ffmpeg_available(), bool)


class TestSyntheticMedia:
    def test_draw_image_returns_an_image(self):
        pytest.importorskip("PIL")
        from PIL import Image

        for kind in ("sunflower", "pizza", "forest"):
            assert isinstance(multimodal.draw_image(kind), Image.Image)

    def test_draw_image_rejects_unknown_kind(self):
        pytest.importorskip("PIL")
        with pytest.raises(ValueError, match="Unknown image kind"):
            multimodal.draw_image("a photograph of my cat")

    def test_written_wav_is_mono_16k(self, tmp_path):
        import wave as wavemod

        path = multimodal.write_tone_wav(str(tmp_path / "tone.wav"), seconds=0.5)
        with wavemod.open(path) as handle:
            assert handle.getframerate() == 16_000
            assert handle.getnchannels() == 1
            assert handle.getnframes() > 0


class TestPhase5Acceptance:
    """PLAN.md Phase 5: L14 must cover at least 3 cross-lingual pairs."""

    def _lesson(self, name: str) -> str:
        return (ROOT / "lessons" / name).read_text(encoding="utf-8")

    def test_l14_uses_an_english_only_index(self):
        """Cross-lingual means the corpus is monolingual and queries are not."""
        l14 = self._lesson("L14_cross_lingual.py")
        assert "DOCUMENTS" in l14
        assert "CROSS_LINGUAL" in l14
        # Documents must be English only.
        assert '("English"' in l14

    def test_l14_covers_at_least_three_languages(self):
        l14 = self._lesson("L14_cross_lingual.py")
        namespace: dict = {}
        tree = ast.parse(l14)
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in {"QUERIES", "DOCUMENTS"} for t in node.targets
            ):
                exec(compile(ast.Module(body=[node], type_ignores=[]), "L14", "exec"), namespace)

        queries = namespace["QUERIES"]
        languages = {language for language, _, _ in queries}
        cross = [q for q in queries if q[0] != "English"]

        assert len(cross) >= 3, f"only {len(cross)} cross-lingual queries"
        assert len(languages) >= 4, f"only {len(languages)} languages: {languages}"

    def test_l14_every_query_targets_a_valid_document(self):
        l14 = self._lesson("L14_cross_lingual.py")
        namespace: dict = {}
        tree = ast.parse(l14)
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in {"QUERIES", "DOCUMENTS"} for t in node.targets
            ):
                exec(compile(ast.Module(body=[node], type_ignores=[]), "L14", "exec"), namespace)

        documents = namespace["DOCUMENTS"]
        for language, text, target in namespace["QUERIES"]:
            assert 0 <= target < len(documents), f"{language} targets index {target}"
            assert text.strip(), f"{language} query is empty"

    def test_all_four_modalities_have_a_lesson(self):
        """The README advertised audio and video with zero demos. Now covered."""
        assert (ROOT / "lessons" / "L11_multimodal_image.py").exists()
        assert (ROOT / "lessons" / "L12_interleaved_multimodal.py").exists()
        assert (ROOT / "lessons" / "L13_audio_and_video.py").exists()

    def test_l13_skips_rather_than_fails_on_missing_audio(self):
        """An optional modality must never crash a lesson."""
        l13 = self._lesson("L13_audio_and_video.py")
        assert "audio_requirements" in l13
        assert "if not audio_ok" in l13
        # The arithmetic section must run regardless.
        assert "budget_for" in l13