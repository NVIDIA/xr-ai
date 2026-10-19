# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query the single CUDA-visible GPU in a disposable process."""

import json
import re


def probe() -> dict:
    try:
        import PyNvVideoCodec as nvc
    except (ImportError, RuntimeError, OSError) as exc:
        return {"nvenc": None, "nvenc_error": str(exc)}
    try:
        nvc.GetEncoderCaps(0, "h264")
    except (nvc.PyNvVCException, RuntimeError, OSError) as exc:
        detail = str(exc)
        # Only explicit NVENC session rejection means unsupported. Other errors
        # (CUDA initialization, memory, driver mismatch) remain unverified.
        unsupported = re.search(r"nvEncOpenEncodeSessionEx[^\n]*returned error [12]\b", detail)
        return {"nvenc": False if unsupported else None, "nvenc_error": detail}
    return {"nvenc": True, "nvenc_error": ""}


if __name__ == "__main__":
    print(json.dumps(probe()))
