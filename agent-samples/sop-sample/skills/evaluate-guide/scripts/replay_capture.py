# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Headless recorded-camera replay using the selected checkout's real engine.

Run with the replay checkout's tests environment (hub and worker dependencies).
Only camera input and user commands are simulated. Models, prompts, tools,
polling, evidence gates, state transitions, aggregation and wall clock are real.
"""

import argparse
import asyncio
import hashlib
import json
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_frames(packet):
    directory = packet.parent
    metadata = json.loads(packet.read_text())
    index = directory / metadata.get("files", {}).get("frame_index", "frames/index.jsonl")
    if not index.is_file():
        index = directory / "frames/index.jsonl"
    rows = [json.loads(line) for line in index.read_text().splitlines() if line.strip()]
    if not rows or any(a["timestamp_us"] > b["timestamp_us"] for a, b in zip(rows, rows[1:])):
        raise ValueError("capture needs a nonempty chronological frame index")
    for row in rows:
        path = (directory / row["path"]).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file():
            raise ValueError(f"invalid capture frame: {path}")
    return rows


async def replay(args):
    root = args.replay_root.resolve()
    worker = root / "agent-samples/sop-sample/worker"
    sys.path.insert(0, str(worker))
    import nemo_relay
    import yaml
    from device_io_hub.ipc import ConnectorEndpoint, HubEndpoint
    from PIL import Image
    from sop_sample_worker._workflow_spec import parse_workflow
    from sop_sample_worker.guides import SelectedGuide
    from sop_sample_worker.replay import ReplayAgent
    from sop_sample_worker.replay_events import PARTICIPANT_JOINED_TOPIC, USER_QUERY_TOPIC
    from xr_ai_hub import PixelFormat, ProcessorEndpoint
    from xr_ai_models import load_models_config, make_llm, make_vlm
    from xr_ai_runtime import Agent, AgentRuntime, RuntimeContext, subscribe
    from xr_ai_tools.current_frame import CurrentFrameTool
    from xr_ai_tools.image import ImageRegistry
    from xr_ai_tools.vision import ImageQueryTool
    from xr_ai_voice import VOICE_OUTPUT_TOPIC, UserQuery, VoiceAggregationAgent, VoiceOutput, VoiceParticipantJoined

    settings_path = root / "agent-samples/sop-sample/yaml/replay.yaml"
    settings = yaml.safe_load(settings_path.read_text())
    models_path = (settings_path.parent / settings["models_config"]).resolve()
    model_config = json.loads(models_path.read_text())
    for name in ("llm", "vlm"):
        endpoint = model_config["models"][name]["endpoint"]["base_url"]
        if urlsplit(endpoint).hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError(f"{name} must use a local model endpoint, got {endpoint}")
    raw = args.guide.read_bytes()
    workflow = parse_workflow(raw)
    guide_hash = hashlib.sha256(raw).hexdigest()
    # Test-only eligibility; never approve or rewrite the author's actual draft.
    selected = SelectedGuide(args.guide.resolve(), guide_hash, replace(workflow, status="approved"))
    rows = capture_frames(args.packet)
    duration = (rows[-1]["timestamp_us"] - rows[0]["timestamp_us"]) / 1e6
    args.output.mkdir(parents=True, exist_ok=False)
    trace = []
    origin = time.monotonic()

    def record(kind, **data):
        event = {"kind": kind, "wall_s": round(time.monotonic() - origin, 3), **data}
        trace.append(event)
        with (args.output / "trace.jsonl").open("a") as stream:
            stream.write(json.dumps(event, default=str) + "\n")

    manifest = {
        "guide": str(args.guide.resolve()),
        "guide_sha256": guide_hash,
        "guide_version": workflow.version,
        "source_status": workflow.status,
        "packet": str(args.packet.resolve()),
        "packet_sha256": digest(args.packet),
        "frames": [
            {"frame_id": r["frame_id"], "path": r["path"], "sha256": digest(args.packet.parent / r["path"])}
            for r in rows
        ],
        "settings_sha256": digest(settings_path),
        "models_sha256": digest(models_path),
        "models": {name: model_config["models"][name] for name in ("llm", "vlm")},
        "runtime_files": {str(p.relative_to(root)): digest(p) for p in sorted(worker.rglob("*.py"))},
        "prompts": {str(p.relative_to(root)): digest(p) for p in sorted(worker.rglob("*.txt"))},
        "duration_s": duration,
        "tail_s": args.tail_s,
        "scope": "Real hub IPC, tools, engine, local models, aggregation and wall clock; recorded camera at 1x.",
        "simulated": ["camera feed", "typed Next after observed completion"],
        "excluded": ["browser", "LiveKit", "microphone/STT", "TTS audio", "human compliance"],
        "draft_test_only": True,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "evaluated.guide.yaml").write_bytes(raw)
    models = load_models_config(models_path)
    llm, vlm = make_llm(models, "llm"), make_vlm(models, "vlm")
    images = ImageRegistry()
    participant = "recorded-evaluation"
    completed = []

    class Observer(Agent):
        @subscribe(VOICE_OUTPUT_TOPIC)
        async def output(self, message: VoiceOutput, _ctx: RuntimeContext) -> None:
            record("agent_output", **message.model_dump(mode="json"))

    class TracedReplay(ReplayAgent):
        async def _trigger(self, pid, step, state):
            arguments = self._resolve(step.trigger.arguments, state)
            result = await super()._trigger(pid, step, state)
            record(
                "trigger",
                step=step.id,
                function=step.trigger.function,
                arguments=arguments,
                available=result[0],
                result=result[1],
            )
            return result

        def _commit(self, session, step, updates, message):
            result = super()._commit(session, step, updates, message)
            record(
                "commit",
                step=step.id,
                updates=updates,
                result=result,
                evidence_hits=session.evidence_hits,
                state=dict(session.state),
            )
            if result[0] and result[1] and step.id not in completed:
                completed.append(step.id)
            return result

    def relay_event(event):
        # SDK image sanitization removes media from Relay events.
        with (args.output / "relay.jsonl").open("a") as stream:
            stream.write(event.to_json() + "\n")

    subscription = "recorded-replay-" + uuid4().hex
    nemo_relay.subscribers.register(subscription, relay_event)
    tasks = []
    endpoints = []
    engine = aggregation = None
    try:
        for name, model in (("llm", llm), ("vlm", vlm)):
            if not await model.health():
                raise RuntimeError(f"local {name} health check failed")
        with tempfile.TemporaryDirectory(prefix="sop-replay-") as temporary:
            pull, pub = f"ipc://{temporary}/in", f"ipc://{temporary}/out"
            hub = HubEndpoint(pull_addr=pull, pub_addr=pub)
            endpoint = ProcessorEndpoint(sub_addr=pub, push_addr=pull)
            connector = ConnectorEndpoint(
                push_addr=pull,
                sub_addr=pub,
                connector_id=subscription,
                shm_name="sop_eval_" + uuid4().hex[:10],
                num_slots=4,
                max_frame_bytes=max(r["width"] * r["height"] * 3 for r in rows),
            )
            endpoints.extend((endpoint, connector, hub))
            tasks.extend((asyncio.create_task(hub.run()), asyncio.create_task(endpoint.run())))
            await asyncio.sleep(0.1)
            await connector.register()
            await endpoint.wait_until_running()
            await connector.notify_participant_joined(participant, pts_us=time.time_ns() // 1000)
            await asyncio.sleep(0.1)
            frames = CurrentFrameTool(
                endpoint=endpoint,
                images=images,
                frame_max_age_s=float(settings["frame_max_age_s"]),
                frame_timeout_s=float(settings["frame_timeout_s"]),
            )
            source_by_pts = {}
            source_by_uri = {}
            frame_handler = frames.handler

            async def selected_frame(request):
                result = await frame_handler(request)
                source = source_by_pts.get(result.timestamp_us)
                source_by_uri[result.image.uri] = source
                record("selected_frame", source=source)
                return result

            frames.handler = selected_frame
            query = ImageQueryTool(
                images=images,
                vlm=vlm,
                system_prompt=(worker / "sop_sample_worker/prompts/replay_vision.txt").read_text(),
            )
            image_handler = query.handler

            async def inspect(request):
                source = source_by_uri.get(request.image.uri)
                result = await image_handler(request)
                record("vision", source=source, question=request.query, **result.model_dump(mode="json"))
                return result

            query.handler = inspect
            runtime = AgentRuntime()
            aggregation = runtime.register("aggregation", VoiceAggregationAgent(llm=llm))
            engine = runtime.register(
                "replay",
                TracedReplay(
                    guide=selected,
                    llm=llm,
                    current_frame=frames,
                    image_query=query,
                    vision_timeout_s=float(settings["vision_timeout_s"]),
                    aggregation=aggregation,
                ),
            )
            runtime.register("observer", Observer())
            engine.bind_runtime(runtime)
            camera_started = time.monotonic()
            first_frame = asyncio.Event()

            async def camera():
                for row in rows:
                    source_s = (row["timestamp_us"] - rows[0]["timestamp_us"]) / 1e6
                    await asyncio.sleep(max(0, camera_started + source_s - time.monotonic()))

                    def decode():
                        with Image.open(args.packet.parent / row["path"]) as image:
                            rgb = image.convert("RGB")
                            return rgb.tobytes(), rgb.width, rgb.height

                    data, width, height = await asyncio.to_thread(decode)
                    pts = time.time_ns() // 1000
                    source_by_pts[pts] = {"frame_id": row["frame_id"], "path": row["path"], "source_s": source_s}
                    await connector.push_frame(
                        data, width, height, PixelFormat.RGB24, pts, participant, "recorded-camera"
                    )
                    first_frame.set()
                record("camera_finished")

            async with runtime:
                camera_task = asyncio.create_task(camera())
                tasks.append(camera_task)
                await asyncio.wait_for(first_frame.wait(), timeout=10)
                await asyncio.sleep(0.1)
                await runtime.publish(PARTICIPANT_JOINED_TOPIC, VoiceParticipantJoined(), participant_id=participant)
                sent = set()
                finished = False
                while time.monotonic() - camera_started < duration + args.tail_s:
                    if camera_task.done():
                        camera_task.result()
                    session = engine._sessions.get(participant)
                    if session is not None and session.step.is_complete(session.state):
                        marker = (session.step_id, session.revision)
                        if marker not in sent:
                            sent.add(marker)
                            last_step = session.step.next_step is None
                            record("simulated_user", text="Next", step=session.step_id)
                            await runtime.publish(
                                USER_QUERY_TOPIC,
                                UserQuery(text="Next", timestamp_us=time.time_ns() // 1000),
                                participant_id=participant,
                            )
                            await asyncio.sleep(0.05)
                            turn = engine._turns.get(participant)
                            if turn is not None:
                                await asyncio.wait_for(asyncio.shield(turn), timeout=90)
                            if (
                                last_step
                                and session.step_id == workflow.start_step
                                and not session.step.is_complete(session.state)
                            ):
                                finished = True
                                record("guide_reset_after_completion")
                                break
                    await asyncio.sleep(0.1)
                session = engine._sessions.get(participant)
                record(
                    "final",
                    completed_steps=completed,
                    all_steps=list(workflow.steps),
                    guide_finished=finished,
                    step=session.step_id if session else None,
                    state=dict(session.state) if session else None,
                )
                await engine.stop()
                await aggregation.stop()
    except Exception as error:
        record("error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if engine is not None:
            await engine.stop()
        if aggregation is not None:
            await aggregation.stop()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for endpoint in endpoints:
            endpoint.stop()
            endpoint.close()
        await nemo_relay.subscribers.flush_async()
        nemo_relay.subscribers.deregister(subscription)
        await llm.close()
        await vlm.close()
        if digest(args.guide) != guide_hash:
            raise RuntimeError("guide changed while replay was running; results are invalid")
        summary = {
            "guide_sha256": guide_hash,
            "completed_steps": completed,
            "final": next((e for e in reversed(trace) if e["kind"] == "final"), None),
            "errors": [e for e in trace if e["kind"] == "error"],
            "vision_calls": sum(e["kind"] == "vision" for e in trace),
        }
        (args.output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--guide", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory; never overwritten")
    parser.add_argument(
        "--tail-s", type=float, default=5, help="Real seconds after the last camera frame; no repeated frames"
    )
    args = parser.parse_args()
    if args.tail_s < 0:
        parser.error("--tail-s must be nonnegative")
    asyncio.run(replay(args))


if __name__ == "__main__":
    main()
