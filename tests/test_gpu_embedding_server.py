# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch the real embedding server with the tea GPU profile and query its corpus.

Run on an idle supported GPU host with Docker and model-download access:
``tests/run_local_gpu_tests.sh test_gpu_embedding_server.py``.
The shared GPU fixture removes managed vLLM containers; stop other samples first.
This checks embedding startup and retrieval, not full-stack memory coexistence.
"""

import asyncio
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from _helpers_model_servers import profile_config
from rag_service import DenseIndex
from xr_ai_launcher import GPUInventoryError, detect_gpu_config
from xr_ai_models import OpenAICompatEmbedding
from xr_ai_vllm import stop_persistent_servers

from _helpers_subprocess import pick_free_port

pytestmark = [pytest.mark.asyncio, pytest.mark.gpu]
ROOT = Path(__file__).resolve().parents[1]


async def test_tea_embedding_startup_and_retrieval(tmp_path):
    if shutil.which("docker") is None:
        pytest.skip("Docker not on PATH")
    try:
        profile = detect_gpu_config()
    except GPUInventoryError as exc:
        pytest.skip(str(exc))
    config = yaml.safe_load(profile_config("embedding_server", profile).read_text())
    port = pick_free_port()
    # Docker ownership is discoverable even before inference binds its port.
    # A persistent pip child can outlive the wrapper and evade port-based cleanup.
    config.update(host="127.0.0.1", port=port, vllm_backend="docker")
    config_path = tmp_path / "embedding.yaml"
    config_path.write_text(yaml.safe_dump(config))
    ready_file = tmp_path / "ready"
    sample = ROOT / "agent-samples/tea-making-sample"
    rag = yaml.safe_load((sample / "yaml/rag_service.yaml").read_text())

    process = await asyncio.create_subprocess_exec(
        # The Docker wrapper needs only dependencies already in the test
        # environment; inference runs in the container, not a host vLLM install.
        sys.executable, str(ROOT / "services/embedding-server/embedding_server/__main__.py"),
        "--config", str(config_path), "--ready-file", str(ready_file),
    )
    try:
        async with asyncio.timeout(1800):
            while not ready_file.exists():
                assert process.returncode is None, f"embedding launcher exited: {process.returncode}"
                await asyncio.sleep(1)
        async with OpenAICompatEmbedding(
            f"http://127.0.0.1:{port}", config["served_model_name"],
        ) as embedder:
            assert await embedder.health()
            identity = await embedder.model_identity()
            assert identity and "Nemotron-3-Embed-1B-BF16" in identity
            vectors = np.asarray(await embedder.embed(["query: How long should green tea steep?"]))
            assert vectors.shape == (1, 2048)
            assert np.isfinite(vectors).all()
            assert np.linalg.norm(vectors) > 0
            index = await DenseIndex.build(
                (sample / "yaml" / rag["documents_dir"]).resolve(), embedder,
                cache_dir=tmp_path / "cache", embedding_dim=rag["embedding_dim"],
                chunk_size=rag["chunk_size"], overlap=rag["overlap"],
                batch_size=rag["batch_size"], min_score=rag["min_score"],
            )
            assert index.vectors.shape[1] == 2048
            for query, source in (
                ("How long should green tea steep and how hot should its water be?", "tea-brewing.md"),
                ("How do I safely handle hot water and avoid steam burns?", "hot-water-safety.md"),
            ):
                matches = await index.retrieve(query, top_k=1)
                assert matches, f"No result above min_score={rag['min_score']} for {query!r}"
                assert matches[0]["source"] == source, matches
                print(f"Retrieval {query!r}: {matches[0]['source']} score={matches[0]['score']:.6f}")
            query = "How do I configure BGP route advertisements between autonomous systems?"
            # Record the unfiltered score as well, so a failure can be diagnosed
            # without changing the threshold or rerunning a model load.
            vector = np.asarray((await embedder.embed(["query: " + query]))[0])
            scores = index.vectors @ (vector / np.linalg.norm(vector))
            print(f"Off-topic {query!r}: max_score={scores.max():.6f}; min_score={rag['min_score']}")
            assert await index.retrieve(query, top_k=2) == []
    finally:
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=30)
            except TimeoutError:
                process.kill()
                await process.wait()
        await asyncio.to_thread(stop_persistent_servers, [("embedding", port)])
