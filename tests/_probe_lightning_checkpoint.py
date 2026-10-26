# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint security probe run inside each isolated NeMo service environment."""

from __future__ import annotations

import importlib
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch


def check_checkpoint_instantiators(namespace: str) -> None:
    lightning = importlib.import_module(namespace)
    cli = importlib.import_module(f"{namespace}.cli")

    class Minimal(lightning.LightningModule):
        def __init__(self):
            super().__init__()

    checkpoint = {
        "state_dict": {},
        "hyper_parameters": {"_instantiator": "os.system"},
        "pytorch-lightning_version": lightning.__version__,
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.ckpt"
        torch.save(checkpoint, path)
        with patch("os.system", side_effect=AssertionError("untrusted instantiator executed")) as untrusted:
            try:
                Minimal.load_from_checkpoint(path, strict=False, weights_only=True)
            except ValueError as error:
                assert "allowlist" in str(error), str(error)
            else:
                raise AssertionError("untrusted checkpoint instantiator accepted")
            untrusted.assert_not_called()

        checkpoint["hyper_parameters"]["_instantiator"] = f"{namespace}.cli.instantiate_module"
        torch.save(checkpoint, path)
        with patch.object(cli, "instantiate_module", side_effect=lambda cls, kwargs: cls()) as trusted:
            model = Minimal.load_from_checkpoint(path, strict=False, weights_only=True)
            assert isinstance(model, Minimal)
            trusted.assert_called_once_with(Minimal, {})
    print(f"{namespace}: untrusted instantiator blocked; trusted instantiator preserved", flush=True)


if __name__ == "__main__":
    for namespace in ("lightning.pytorch", "pytorch_lightning"):
        check_checkpoint_instantiators(namespace)
