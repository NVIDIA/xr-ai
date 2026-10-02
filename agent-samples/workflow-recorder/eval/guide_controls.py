# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate guide control proposals against a live model without executing tools."""

import argparse
import asyncio
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml
from workflow_recorder_worker._workflow_engine import _FOREGROUND_MODEL_OPTIONS, SopEngineAgent, _Session
from workflow_recorder_worker._workflow_spec import load_workflow
from workflow_recorder_worker.catalog import CatalogGuide, GuideCatalog
from xr_ai_models import ChatMessage, load_models_config, make_llm
from xr_ai_tools.tool_calling import tool_definitions

_SAMPLE = Path(__file__).resolve().parents[1]


class _DryRunEngine(SopEngineAgent):
    async def _tool_loop(self, system, user, tools, *, foreground=False):
        # Capture the model's first proposal; no Tool.execute or live guide run.
        assert foreground
        self.proposal = await self._llm.chat(
            (ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)),
            tools=tool_definitions(tools), temperature=0.0, **_FOREGROUND_MODEL_OPTIONS,
        )
        return self.proposal.content or "Tool proposal only; not executed."


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=_SAMPLE / "yaml/models.json")
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("guide_controls.yaml"))
    args = parser.parse_args()
    cases = yaml.safe_load(args.cases.read_text())
    path = _SAMPLE / "skills/recording-to-guide/references/example.guide.yaml"
    workflow = replace(load_workflow(path), status="approved")
    guide = CatalogGuide(path, path.name, "eval-only", "", workflow, None)
    llm = make_llm(load_models_config(args.models), "llm")
    passed = 0
    try:
        with TemporaryDirectory(prefix="sop-control-eval-") as temporary:
            root = Path(temporary)
            catalog = GuideCatalog(root, root / "index.json", interval_s=1)
            catalog._guides = (guide,)  # In-memory fixture only; never scan or write the user's catalog.
            engine = _DryRunEngine(
                catalog=catalog, llm=llm, current_frame=None, image_query=None,
                vision_timeout_s=10, recorder=None,
            )
            for case in cases:
                query = case["query"]
                if engine.is_control(query):
                    raise ValueError("Recording and deterministic guide controls are not part of this model-only eval")
                engine._sessions.clear()
                if case.get("active", False):
                    engine._sessions["eval"] = _Session(
                        participant_id="eval", guide=guide,
                        state=workflow.initial_state(), step_id=workflow.start_step,
                    )
                answer = await engine._route(query, "eval")
                response = engine.proposal
                calls = [(call.name, json.loads(call.arguments)) for call in response.tool_calls or ()]
                expected = case.get("tool")
                ok = not calls if expected is None else len(calls) == 1 and calls[0][0] == expected
                if ok and expected is not None and "arguments" in case:
                    ok = calls[0][1] == case["arguments"]
                if ok and expected == "workflow__start":
                    ok = calls[0][1].get("selector") in {workflow.id, workflow.name}
                if ok and "contains" in case:
                    ok = all(text.casefold() in answer.casefold() for text in case["contains"])
                ok = ok and response.finish_reason not in {"length", "max_tokens"}
                passed += ok
                print(f"{'PASS' if ok else 'FAIL'} {case['name']}: {calls or answer.strip()[:240]}")
    finally:
        await llm.close()
    print(f"Guide control proposals: {passed}/{len(cases)}. No tools executed.")
    if not cases or passed != len(cases):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
