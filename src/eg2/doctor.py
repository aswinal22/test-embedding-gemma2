"""`python -m eg2.doctor` -- the environment preflight check.

Run this before anything else. If a lesson fails, run this first: it tells you
which dependency is wrong instead of leaving you with a stack trace from deep
inside transformers.

Exits 0 if the environment can run the lessons, 1 otherwise.
"""

from __future__ import annotations

import importlib
import platform
import sys
from dataclasses import dataclass, field

from . import output
from .config import (
    MIN_PYTHON,
    MIN_SENTENCE_TRANSFORMERS,
    MIN_TORCH,
    MIN_TRANSFORMERS,
    MODEL_ID,
    at_least,
    version_tuple,
)

PASS = "pass"
FAIL = "fail"
INFO = "info"


@dataclass
class Check:
    """One diagnostic result."""

    name: str
    status: str
    detail: str = ""
    fix: str = ""


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "", fix: str = "") -> Check:
        check = Check(name, status, detail, fix)
        self.checks.append(check)
        return check

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.status == FAIL]

    @property
    def ok(self) -> bool:
        return not self.failures


def _module_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    return getattr(module, "__version__", "unknown")


def _system_ram_gb() -> float | None:
    """Total physical RAM in GB, or None if the platform will not say."""
    try:
        import os

        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round(pages * page_size / (1024**3), 1)
    except (AttributeError, ValueError, OSError, ImportError):
        pass

    # Windows has no sysconf. Ask the kernel instead.
    try:
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return round(status.ullTotalPhys / (1024**3), 1)
    except Exception:
        pass
    return None


def _check_interpreter(report: Report) -> None:
    """Which Python is actually running this?

    Worth its own line. On Windows it is extremely common for `python` and `pip`
    to belong to different installations -- which makes a correct `pip install`
    look like it did nothing, because the package landed in an interpreter you
    are not running.
    """
    report.add("Interpreter", INFO, f"{sys.executable}")
    report.add(
        "Install with",
        INFO,
        "use `python -m pip install ...` so packages land in THIS interpreter",
    )


def _check_python(report: Report) -> None:
    current = platform.python_version()
    required = ".".join(str(p) for p in MIN_PYTHON)
    if at_least(current, required):
        report.add("Python", PASS, f"{current}")
    else:
        report.add(
            "Python",
            FAIL,
            f"{current} is older than the required {required}",
            f"Install Python {required} or newer.",
        )


def _check_package(report: Report, import_name: str, label: str, minimum: str, install: str) -> None:
    version = _module_version(import_name)
    if version is None:
        report.add(label, FAIL, "not installed", f"Run: python -m pip install -U {install}")
        return
    if version == "unknown":
        report.add(label, INFO, f"installed, version unknown")
        return
    if at_least(version, minimum):
        report.add(label, PASS, f"{version}")
    else:
        report.add(
            label,
            FAIL,
            f"{version} is older than the required {minimum}",
            f"Run: python -m pip install -U {install}",
        )


def _check_model_type(report: Report) -> None:
    """The decisive check: can this transformers actually load our model?

    Comparing version strings only proves someone installed something. This asks
    the library itself whether it knows the `embedding_gemma2` architecture.
    """
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    except Exception:
        report.add(
            "embedding_gemma2 support",
            FAIL,
            "could not import transformers' auto config registry",
            "Run: python -m pip install -U transformers",
        )
        return

    if "embedding_gemma2" in CONFIG_MAPPING:
        report.add("embedding_gemma2 support", PASS, "model type is registered")
    else:
        report.add(
            "embedding_gemma2 support",
            FAIL,
            "this transformers build does not know the 'embedding_gemma2' model type",
            "Run: python -m pip install -U transformers",
        )


def _check_hardware(report: Report) -> None:
    ram = _system_ram_gb()
    if ram:
        report.add("System RAM", INFO, f"{ram} GB")
    else:
        report.add("System RAM", INFO, "unknown on this platform")

    try:
        import torch
    except ImportError:
        report.add("Compute device", FAIL, "torch not installed", "Run: python -m pip install torch")
        return

    if torch.cuda.is_available():
        bf16 = torch.cuda.is_bf16_supported()
        detail = f"CUDA: {torch.cuda.get_device_name(0)}"
        if bf16:
            detail += " (bfloat16 available)"
        else:
            detail += " (no native bfloat16; will use float32)"
        report.add("Compute device", PASS, detail)
    else:
        report.add(
            "Compute device",
            INFO,
            "CPU only -- every lesson in this repo is designed to run here",
        )


def _check_model_cache(report: Report) -> None:
    from .loader import model_cache_path

    path = model_cache_path()
    if path is None:
        report.add("Model cache", INFO, "huggingface_hub not installed")
        return
    import os

    if os.path.isdir(path):
        report.add("Model cache", INFO, f"already downloaded at {path}")
    else:
        report.add(
            "Model cache",
            INFO,
            f"not downloaded yet (first run downloads to {path})",
        )


def build_report() -> Report:
    """Run every check and return the results."""
    report = Report()
    _check_python(report)
    _check_interpreter(report)
    _check_package(report, "torch", "torch", MIN_TORCH, "torch")
    _check_package(
        report,
        "sentence_transformers",
        "sentence-transformers",
        MIN_SENTENCE_TRANSFORMERS,
        f"'sentence-transformers>={MIN_SENTENCE_TRANSFORMERS}'",
    )
    _check_package(report, "transformers", "transformers", MIN_TRANSFORMERS, f"'transformers>={MIN_TRANSFORMERS}'")
    _check_package(report, "numpy", "numpy", "1.24", "numpy")
    _check_package(report, "PIL", "pillow", "9.0", "pillow")
    _check_model_type(report)
    _check_hardware(report)
    _check_model_cache(report)

    extras = [name for name, label in (("matplotlib", "matplotlib"),) if _module_version(name)]
    report.add(
        "Notebook extras",
        INFO,
        f"installed: {', '.join(extras)}" if extras else "matplotlib not installed (only needed for the notebook)",
    )
    return report


def print_report(report: Report) -> None:
    """Render a report with colour and a clear pass/fail header."""
    output.section("Environment report")
    for check in report.checks:
        if check.status == PASS:
            marker = output.paint("[ ok ]", "green")
        elif check.status == FAIL:
            marker = output.paint("[FAIL]", "red")
        else:
            marker = output.paint("[info]", "dim")

        name = check.name.ljust(24)
        print(f"  {marker}  {name}  {check.detail}")

    failures = report.failures
    print()
    if failures:
        output.error(f"{len(failures)} problem(s) found. Fix these before running a lesson:")
        print()
        for check in failures:
            print(f"    {output.paint(check.name, 'bold')}: {check.fix}")
    else:
        output.ok("Environment looks good. Run the next lesson.")


def main(argv: list[str] | None = None) -> int:
    """Entry point for `python -m eg2.doctor`."""
    output.banner(
        "EmbeddingGemma 2 -- environment doctor",
        f"Checking whether this machine can run {MODEL_ID}",
    )
    report = build_report()
    print_report(report)
    output.footer("Next: python lessons/L00_check_environment.py")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))