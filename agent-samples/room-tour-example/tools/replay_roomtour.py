# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end replay validation for room-tour-example against a real recording.

Feeds frames from a room-tour .MOV through the *real* VLMPerceptor (Cosmos via
the vlm-server) into the real SemanticTopoMap, exactly as the live worker would
— but injects the spoken commands as text (simulating STT) and captures the
agent's spoken output (simulating TTS). No hub, no mic, no glasses needed.

Scenario (mirrors the goal):
    start room tour → this is the office → this is the kitchen →
    this is the meeting room → stop room tour
    then, standing in the meeting room:  "take me to the monitor from here"
    then the user walks back toward the office while the agent guides live.

Run from the worker dir with the worker venv (which has xr-ai-models), with the
vlm-server reachable at the base_url in yaml/models.yaml:
    cd agent-samples/room-tour-example/worker
    uv run python ../tools/replay_roomtour.py [path/to/roomtour.MOV]
"""
from __future__ import annotations

import asyncio
import glob
import subprocess
import sys
from pathlib import Path

from PIL import Image

# Import the worker modules (run from the worker dir).
import agent as agent_mod
from agent import RoomTourBrain
from xr_ai_models import load_models_config, make_vlm

_HERE = Path(__file__).resolve().parent
_SAMPLE = _HERE.parent
_FRAMES_DIR = Path("/tmp/rt_replay")
_MODELS_YAML = _SAMPLE / "yaml" / "models.yaml"
_DEFAULT_VIDEO = Path("/home/wenxind/roomtour.MOV")
_FPS = "1/2.5"   # one frame every 2.5s of the walk

# Timeline segmentation of the continuous walk (frames are 2.5s apart).
# office first, then the hallway→kitchen, then the open meeting area.
SEG = {"office": (0, 4), "kitchen": (4, 10), "meeting room": (10, 99)}


def _extract(video: Path) -> None:
    _FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    for old in _FRAMES_DIR.glob("*.jpg"):
        old.unlink()
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
         "-vf", f"fps={_FPS},scale=768:-1", str(_FRAMES_DIR / "f_%03d.jpg")],
        check=True,
    )


def load_frames(video: Path) -> dict[str, list[Image.Image]]:
    if not list(_FRAMES_DIR.glob("*.jpg")):
        if not video.exists():
            sys.exit(f"video not found: {video} (pass the path as an argument)")
        print(f"extracting frames from {video} → {_FRAMES_DIR}")
        _extract(video)
    paths = sorted(glob.glob(str(_FRAMES_DIR / "*.jpg")))
    imgs = [Image.open(p).convert("RGB") for p in paths]
    print(f"loaded {len(imgs)} frames from {_FRAMES_DIR}")
    return {room: imgs[a:b] for room, (a, b) in SEG.items()}


class _FakeEndpoint:
    def on_data(self, cb): ...
    def on_frame(self, cb): ...
    async def request_frame(self, sig): return None


class _FakeTransport:
    endpoint = _FakeEndpoint()


async def main() -> None:
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_VIDEO
    rooms = load_frames(video)

    cfg = load_models_config(_MODELS_YAML)
    vlm = make_vlm(cfg, "vlm")
    print("waiting for VLM health…")
    for _ in range(120):
        if await vlm.health():
            break
        await asyncio.sleep(2)
    else:
        sys.exit("VLM never became healthy")
    print("VLM healthy ✓\n" + "=" * 70)

    brain = RoomTourBrain(
        transport=_FakeTransport(), vlm=vlm,
        nav_monitor_interval_s=1.5, nav_max_secs=300.0,
    )

    # Feed PIL frames directly: the "frame" the brain fetches IS a PIL image,
    # so make frame_to_pil the identity and have _get_frame return the current one.
    agent_mod.frame_to_pil = lambda f: f
    cur: dict[str, Image.Image | None] = {"img": None}

    async def _get_frame(pid):
        return cur["img"]
    brain._get_frame = _get_frame

    async def _speak(text, *, pid):
        print(f"   🔊 AGENT (live): {text}")
    brain._push_text = _speak

    async def say(text: str):
        print(f"\n👤 USER: {text}")
        async for chunk in brain._respond("rayban", text):
            if chunk:
                print(f"   🔊 AGENT: {chunk}")

    async def ingest(img: Image.Image):
        cur["img"] = img
        desc = await brain._perceive("rayban")
        if desc and not agent_mod._scene_is_empty(desc):
            print(f"     ↳ perceived: cap={desc.caption[:64]!r} "
                  f"objs={sorted(desc.object_labels())[:5]} ocr={sorted(desc.ocr_tokens())}")
            brain._ingest_labeled(desc)

    # ── the tour ────────────────────────────────────────────────────────────
    await say("Start room tour.")
    for room in ("office", "kitchen", "meeting room"):
        frames = rooms[room]
        if not frames:
            continue
        cur["img"] = frames[0]
        await say(f"This is the {room}.")          # labels + seeds one frame
        for img in frames[1:]:                      # the capture-loop's ingest path
            await ingest(img)
    await say("Stop room tour.")

    print("\n" + "=" * 70)
    print("MAP:", brain._map.stats())
    print("rooms by node:", {nid: brain._room_of(nid) for nid in brain._map.nodes})
    print("=" * 70)

    # ── the question, standing in the meeting room ────────────────────────────
    cur["img"] = rooms["meeting room"][-1]
    await say("Take me to the monitor from here.")

    # ── the user walks back: meeting → kitchen → office (live guidance) ────────
    print("\n[user starts walking — feeding frames to the live guidance loop]")
    walk = (rooms["meeting room"][::-1][:2]
            + rooms["kitchen"][::-1][:3]
            + rooms["office"][::-1][:3])
    for img in walk:
        cur["img"] = img
        await asyncio.sleep(9.0)        # hold each frame long enough for a live VLM tick
    await asyncio.sleep(15.0)           # linger on the office so arrival can fire
    brain._stop_nav()

    print("\n" + "=" * 70 + "\nREPLAY COMPLETE")
    await vlm.close()


if __name__ == "__main__":
    asyncio.run(main())
