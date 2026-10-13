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
from eg2 import config, loader, output  # noqa: E402


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
        source = (ROOT / "lessons" / lesson).read_text(encoding="utf-8")
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