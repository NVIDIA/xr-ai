"""
Hardware video codec guard.

We require NVIDIA hardware video codecs (NVDEC + NVENC) because OpenH264,
the software fallback bundled with libwebrtc inside livekit-rtc, is
royalty-bearing for end users and must not be used in distribution.

Call require_nvidia_video_codecs() before the LiveKit connector starts.
It raises RuntimeError immediately if the required libraries are absent,
preventing any silent fallback to software decode/encode.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import platform
import sys


def require_nvidia_video_codecs() -> None:
    """
    Raise RuntimeError if NVDEC or NVENC libraries are not found.

    Only enforced on Linux (the primary server target). macOS uses
    VideoToolbox (always present); Windows support is a future TODO.
    """
    if sys.platform != "linux":
        return

    missing = []

    # NVDEC — libnvcuvid: used by libwebrtc for hardware H.264/H.265 decode.
    if not _find_lib("nvcuvid"):
        missing.append("NVDEC (libnvcuvid.so)")

    # NVENC — libnvidia-encode: used by libwebrtc for hardware H.264 encode.
    if not _find_lib("nvidia-encode"):
        missing.append("NVENC (libnvidia-encode.so)")

    if missing:
        raise RuntimeError(
            "\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "  Hardware video codec required — refusing to start\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"  Missing: {', '.join(missing)}\n"
            "\n"
            "  livekit-rtc bundles libwebrtc, which includes OpenH264 as a\n"
            "  software fallback. OpenH264 is royalty-bearing for end users\n"
            "  and must not be used in this deployment.\n"
            "\n"
            "  Fix: ensure the NVIDIA driver and CUDA Video SDK are installed\n"
            "  and that /dev/nvidia* devices are accessible to this process.\n"
            "  In Docker: pass --gpus all (or --device /dev/nvidia0, etc.).\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        )


def _find_lib(name: str) -> bool:
    """Return True if lib<name> can be located or loaded."""
    if ctypes.util.find_library(name):
        return True
    # find_library may miss versioned .so on some distros; try loading directly.
    for suffix in (".so", ".so.1", ".so.0"):
        try:
            ctypes.CDLL(f"lib{name}{suffix}")
            return True
        except OSError:
            pass
    return False
