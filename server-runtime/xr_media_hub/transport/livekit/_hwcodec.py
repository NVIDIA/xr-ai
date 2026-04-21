"""
Hardware video codec guard.

We require NVIDIA hardware video codecs (NVDEC + NVENC) because OpenH264,
the software fallback bundled with libwebrtc inside livekit-rtc, is
royalty-bearing for end users and must not be used in distribution.

The check is intentionally strict:
  - macOS: always fails — Apple dropped NVIDIA GPU support in macOS 10.14.
  - Linux: requires libnvcuvid.so (NVDEC) + libnvidia-encode.so (NVENC).
           VA-API on NVIDIA routes through these same libraries, so the
           check covers both the direct NVDEC path and the VA-API path.
  - Windows: requires nvcuvid.dll + nvEncodeAPI64.dll.

Override (development only)
───────────────────────────
Set the environment variable to bypass the check at your own risk:

    XR_AI_SKIP_HWCODEC_CHECK=1 uv run xr_media_hub

This will log a prominent warning and continue. Do NOT set this in
production — OpenH264 must not be used in distributed software.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import sys

log = logging.getLogger(__name__)

_SKIP_ENV = "XR_AI_SKIP_HWCODEC_CHECK"


def require_nvidia_video_codecs() -> None:
    """
    Raise RuntimeError unless NVIDIA NVDEC and NVENC are present.

    Fails on macOS unconditionally (no NVIDIA GPU support).
    On Linux and Windows, probes the NVIDIA Video SDK libraries directly.

    Set XR_AI_SKIP_HWCODEC_CHECK=1 to bypass (development only).
    """
    if os.environ.get(_SKIP_ENV):
        log.warning(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "  %s is set — hardware codec check SKIPPED\n"
            "  OpenH264 (royalty-bearing) may be used. DO NOT distribute.\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            _SKIP_ENV,
        )
        return

    if sys.platform == "darwin":
        _fail(
            "macOS",
            ["NVDEC", "NVENC"],
            "Apple dropped NVIDIA GPU support in macOS 10.14. "
            "Run the server on a Linux machine with an NVIDIA GPU.",
        )

    if sys.platform == "linux":
        _check_linux()
        return

    if sys.platform == "win32":
        _check_windows()
        return

    _fail(sys.platform, ["NVDEC", "NVENC"], f"Unsupported platform: {sys.platform}")


# ── platform checks ───────────────────────────────────────────────────────────

def _check_linux() -> None:
    missing = []

    # NVDEC — libnvcuvid: the NVIDIA CUVID / Video Decode API.
    # Present whenever the NVIDIA driver + Video SDK are installed.
    # VA-API on NVIDIA (nvidia-vaapi-driver) also requires this library.
    if not _so("nvcuvid", [".so.1", ".so"]):
        missing.append("NVDEC (libnvcuvid.so.1)")

    # NVENC — libnvidia-encode: the NVIDIA Video Encode API.
    if not _so("nvidia-encode", [".so.1", ".so"]):
        missing.append("NVENC (libnvidia-encode.so.1)")

    if missing:
        _fail(
            "Linux",
            missing,
            "Ensure the NVIDIA driver and CUDA Video SDK are installed and that\n"
            "  /dev/nvidia* devices are accessible to this process.\n"
            "  In Docker: pass --gpus all  (or --device /dev/nvcuvid etc.).",
        )


def _check_windows() -> None:
    missing = []

    if not _dll("nvcuvid"):
        missing.append("NVDEC (nvcuvid.dll)")

    # 64-bit systems ship nvEncodeAPI64.dll; 32-bit ship nvEncodeAPI.dll.
    if not _dll("nvEncodeAPI64") and not _dll("nvEncodeAPI"):
        missing.append("NVENC (nvEncodeAPI64.dll)")

    if missing:
        _fail(
            "Windows",
            missing,
            "Ensure the NVIDIA driver and CUDA Video SDK are installed.",
        )


# ── helpers ───────────────────────────────────────────────────────────────────

def _so(name: str, suffixes: list[str]) -> bool:
    if ctypes.util.find_library(name):
        return True
    for s in suffixes:
        try:
            ctypes.CDLL(f"lib{name}{s}")
            return True
        except OSError:
            pass
    return False


def _dll(name: str) -> bool:
    try:
        ctypes.CDLL(f"{name}.dll")
        return True
    except OSError:
        return False


def _fail(platform: str, missing: list[str], hint: str) -> None:
    lines = [
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "  NVIDIA hardware video codec required — refusing to start",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"  Platform : {platform}",
        f"  Missing  : {', '.join(missing)}",
        "",
        "  livekit-rtc bundles libwebrtc which includes OpenH264 as a",
        "  software fallback. OpenH264 is royalty-bearing for end users",
        "  and must not be used in this deployment.",
        "",
        f"  {hint}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]
    raise RuntimeError("\n".join(lines))
