"""Measuring how much memory this process is using.

Standard library only. The point of `lessons/L10_modular_loading.py` is to
replace a wrong number in this repo's README with a measured one, and that
requires actually measuring.

Three platforms, three mechanisms, because there is no stdlib call that gives
you resident set size portably:

    Windows   psapi.GetProcessMemoryInfo via ctypes
    Linux     /proc/self/status
    macOS     resource.getrusage  (peak, not current -- documented below)

If none of those work, the functions return None and callers should say so
rather than printing a fabricated zero.
"""

from __future__ import annotations

import sys


def current_rss_bytes() -> int | None:
    """Resident set size right now, in bytes. None if unavailable."""
    if sys.platform == "win32":
        return _windows_rss()
    if sys.platform.startswith("linux"):
        return _linux_rss()
    if sys.platform == "darwin":
        return _macos_rss()
    return None


def _windows_rss() -> int | None:
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:
        return None

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    try:
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)

        # argtypes/restypes are load-bearing, not decoration. GetCurrentProcess
        # returns a pseudo-handle whose value is -1; without restype=HANDLE,
        # ctypes narrows it to a 32-bit int on 64-bit Windows and the call
        # silently fails with WorkingSetSize == 0.
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

        handle = kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return None
        return int(counters.WorkingSetSize)
    except Exception:
        return None


def _linux_rss() -> int | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    # Value is in kibibytes, e.g. "VmRSS:\t  123456 kB"
                    parts = line.split()
                    return int(parts[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _macos_rss() -> int | None:
    """Peak RSS on macOS.

    `getrusage` reports the PEAK resident size, not the current one, and units
    are bytes on macOS (kilobytes on Linux). Callers should treat this as an
    upper bound rather than an instantaneous reading.
    """
    try:
        import resource

        # ru_maxrss is in BYTES on macOS and in KILOBYTES on Linux.
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None


def is_peak_measurement() -> bool:
    """True if current_rss_bytes() actually reports a peak, not a snapshot."""
    return sys.platform == "darwin"


def format_bytes(value: float | None, precision: int = 1) -> str:
    """Human-readable size. Returns 'unknown' rather than lying about None."""
    if value is None:
        return "unknown"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(value)
    index = 0
    while size >= 1024.0 and index < len(units) - 1:
        size /= 1024.0
        index += 1
    if index == 0:
        return f"{int(size)} {units[index]}"
    return f"{size:.{precision}f} {units[index]}"


def parameter_count(model) -> int:
    """Total number of parameters in a model's modules."""
    try:
        return sum(p.numel() for p in model.parameters())
    except Exception:
        return 0