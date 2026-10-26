# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check real NeMo imports and checkpoint protection in each inference environment.

These qualification tests require the Linux GPU stack but no model downloads.
They catch integration failures that lock resolution and model stand-ins miss.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize(
    ("service", "imports"),
    [
        ("stt-server", "from nemo.collections.asr.models import ASRModel"),
        ("magpie-tts", "from nemo.collections.tts.models.magpietts import MagpieTTSModel"),
        (
            "speaker-stt",
            "from nemo.collections.asr.models import ASRModel, SortformerEncLabelModel\n"
            "from nemo.collections.asr.parts.utils.multispk_transcribe_utils import SpeakerTaggedASR",
        ),
    ],
)
def test_nemo_inference_dependencies_import(service: str, imports: str) -> None:
    if shutil.which("uv") is None:
        pytest.skip("uv not on PATH")
    project = _ROOT / "services" / service
    subprocess.run(
        ["uv", "--config-file", str(_ROOT / "uv.toml"), "sync",
         "--project", str(project)],
        check=True,
        timeout=1200,
    )
    probe = (
        "import torch\n"
        "assert torch.cuda.is_available(), 'CUDA unavailable for NeMo qualification'\n"
        "from importlib.metadata import version\n"
        "from packaging.version import Version\n"
        "for package, minimum in [('hydra-core', '1.3.7'), ('lightning', '2.6.6'), "
        "('pytorch-lightning', '2.6.6'), ('fsspec', '2026.6.0')]:\n"
        "    actual = version(package)\n"
        "    assert Version(actual) >= Version(minimum), f'{package} {actual} is below {minimum}'\n"
        "import hydra, lightning.pytorch\n"
        + imports
        + "\nimport runpy\n"
        + f"runpy.run_path({str(_ROOT / 'tests/_probe_lightning_checkpoint.py')!r}, run_name='__main__')\n"
    )
    result = subprocess.run(
        [str(project / ".venv/bin/python"), "-c", probe],
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, f"{service} dependency qualification failed:\n{result.stdout}\n{result.stderr}"
