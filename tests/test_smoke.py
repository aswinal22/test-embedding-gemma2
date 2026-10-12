"""Fast tests for the eg2 package. No model download required.

Run with:  python -m pytest tests/
"""

from __future__ import annotations

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