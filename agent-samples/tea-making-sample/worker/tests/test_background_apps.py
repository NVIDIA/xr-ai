# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from tea_making_worker.applications.change_events import ChangeCommitRequest
from tea_making_worker.applications.change_watch import ChangeWatchApplication
from tea_making_worker.applications.transcript import TranscriptApplication
from tea_making_worker.applications.video_log import VideoLogApplication
from tea_making_worker.desktop.runtime import DesktopRuntime
from tea_making_worker.desktop.spec import ApplicationSpec, DesktopSpec
from tea_making_worker.engine.text_output import TextOutputBridge
from tea_making_worker.runtime.scope import current_invocation
from tea_making_worker.runtime.state import SessionStore
from tea_making_worker.spec import load_workflow
from xr_ai_nat.functions.vision import LiveVisionResult

_WORKFLOW = Path(__file__).parents[2] / "yaml" / "workflow.yaml"


def _runtime(app: ApplicationSpec) -> DesktopRuntime:
    return DesktopRuntime(DesktopSpec(root_prompt="root", capabilities={}, applications={app.id: app}))


class _View:
    def __init__(self, *captions: str) -> None:
        self.captions = list(captions)
        self.requests = []

    async def ainvoke(self, request, *, to_type):
        self.requests.append(request)
        return LiveVisionResult(answer=self.captions.pop(0))


class _ChangeAgent:
    def __init__(self, app: ChangeWatchApplication) -> None:
        self.app = app

    async def ainvoke(self, request, *, to_type):
        call = current_invocation()
        await self.app.commit(
            call.session,
            call.context["change_watch.caption"],
            ChangeCommitRequest(important=True, summary="A person entered the room."),
        )
        return "committed"


class _SummaryAgent:
    def __init__(self, app: TranscriptApplication) -> None:
        self.app = app

    async def ainvoke(self, request, *, to_type):
        call = current_invocation()
        await self.app.commit_summary(
            call.session,
            call.context["transcript.state"],
            call.context["transcript.turns"],
            "The speaker discussed two project updates.",
        )
        return "committed"


class _VideoDeltaAgent:
    def __init__(self, app: VideoLogApplication) -> None:
        self.app = app
        self.requests: list[dict] = []

    async def ainvoke(self, request, *, to_type):
        self.requests.append(json.loads(request))
        call = current_invocation()
        await self.app.commit(
            call.session,
            call.context["video_log.state"],
            call.context["video_log.caption"],
            call.trace_id,
            f"Unique activity {len(self.requests)}.",
        )
        return "committed"


class BackgroundApplicationTest(unittest.IsolatedAsyncioTestCase):
    async def test_change_watch_uses_a_baseline_then_notifies_an_important_change(self) -> None:
        app_spec = ApplicationSpec(
            "change_watch",
            "Visual change watcher",
            "background",
            "watch changes",
            {
                "interval_s": 1,
                "history_size": 2,
                "default_instruction": "important changes",
                "caption_prompt": "Caption visible facts.",
                "event_prompt": "Commit important changes.",
            },
        )
        runtime = _runtime(app_spec)
        outputs: list[tuple[str, str]] = []

        async def output(_participant_id: str, label: str, message: str) -> None:
            outputs.append((label, message))

        app = ChangeWatchApplication(app_spec, runtime, output)
        view = _View("An empty room.", "A person entered the room.")
        app._view = view
        app._agent = _ChangeAgent(app)
        session = SessionStore(load_workflow(_WORKFLOW)).get("tester")

        await app.start(session, "people entering the room")
        await app.tick(session)
        app._states[session.participant_id].next_tick = 0
        await app.tick(session)

        self.assertEqual(
            outputs,
            [("Visual change watcher", "A person entered the room.")],
        )
        self.assertIn("Focus: people entering the room", view.requests[0]["question"])
        self.assertEqual(
            list(app._states[session.participant_id].captions),
            ["An empty room.", "A person entered the room."],
        )
        self.assertEqual(runtime.current(session), "root")

    async def test_text_output_bypasses_query_and_uses_application_label(self) -> None:
        class _Transport:
            messages = []

            async def send_return_data(self, message) -> None:
                self.messages.append(message)

        class _Session:
            transport = _Transport()

        bridge = TextOutputBridge(_Session())
        await bridge.send("viewer", "Visual change watcher", "A parcel moved.")

        message = _Session.transport.messages[0]
        self.assertEqual(message.participant_id, "viewer")
        self.assertEqual(message.topic, "Visual change watcher")
        self.assertEqual(message.data.decode(), "A parcel moved.")

    async def test_transcript_emits_a_labeled_periodic_summary_without_tts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app_spec = ApplicationSpec(
                "transcript",
                "Transcript recorder",
                "background",
                "record speech",
                {
                    "output_dir": Path(directory),
                    "summary_interval_s": 60,
                    "summary_prompt": "Summarize and commit.",
                },
            )
            runtime = _runtime(app_spec)
            outputs: list[tuple[str, str]] = []

            async def output(_participant_id: str, label: str, message: str) -> None:
                outputs.append((label, message))

            app = TranscriptApplication(app_spec, runtime, output)
            app._agent = _SummaryAgent(app)
            session = SessionStore(load_workflow(_WORKFLOW)).get("speaker")

            await asyncio.wait_for(app.start(session), timeout=2)
            await asyncio.wait_for(
                app.on_transcription(session, "First project update.", "one"),
                timeout=2,
            )
            await asyncio.wait_for(
                app.on_transcription(session, "Second project update.", "two"),
                timeout=2,
            )
            await asyncio.wait_for(app.tick(session), timeout=2)
            self.assertEqual(outputs, [])
            app._states[session.participant_id].next_summary = 0
            await asyncio.wait_for(app.tick(session), timeout=2)
            path = app._states[session.participant_id].path
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

            self.assertEqual([record["type"] for record in records], ["session", "utterance", "utterance", "summary"])
            self.assertEqual(
                outputs,
                [("Transcript recorder", "The speaker discussed two project updates.")],
            )
            self.assertEqual(runtime.current(session), "root")

    async def test_video_log_runs_every_tick_with_a_five_caption_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app_spec = ApplicationSpec(
                "video_log",
                "Video activity logger",
                "background",
                "log camera activity",
                {
                    "output_dir": Path(directory),
                    "interval_s": 2,
                    "history_size": 5,
                    "caption_prompt": "Describe the entire visible scene.",
                    "delta_prompt": "Commit only the unique current activity.",
                },
            )
            runtime = _runtime(app_spec)
            captions = tuple(f"Scene state {index}." for index in range(6))
            view = _View(*captions)
            app = VideoLogApplication(app_spec, runtime)
            agent = _VideoDeltaAgent(app)
            app._view = view
            app._agent = agent
            session = SessionStore(load_workflow(_WORKFLOW)).get("viewer")

            await app.start(session)
            for _ in captions:
                app._states[session.participant_id].next_tick = 0
                await app.tick(session)

            state = app._states[session.participant_id]
            records = [
                json.loads(line)
                for line in state.path.read_text(encoding="utf-8").splitlines()
            ]
            observations = [record for record in records if record["type"] == "observation"]

            self.assertEqual(len(observations), 6)
            self.assertEqual(list(state.captions), list(captions[-5:]))
            self.assertEqual(agent.requests[-1]["previous"], list(captions[1:5]))
            self.assertEqual(agent.requests[-1]["current"], captions[-1])
            self.assertTrue(all(request["question"] == app.caption_prompt for request in view.requests))
            self.assertEqual(runtime.current(session), "root")


if __name__ == "__main__":
    unittest.main()
