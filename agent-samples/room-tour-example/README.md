<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# room-tour-example

A voice-driven **semantic room tour**: the wearer walks through a space naming
each room, the agent remembers what it saw where, and afterwards answers
spatial questions like *"where am I"* and *"where is the sofa"*.

```
"start room tour"
"this is the living room"   → (agent harvests sofa, tv, coffee table … )
"this is the kitchen"       → (agent harvests fridge, sink, stove … )
"stop tour"
"where am I"                → "You're in the kitchen."
"where is the sofa"         → "The sofa is in the living room, to your left."
"take me to the monitor from here"  → "You're in the meeting room. To get to the
   (live, as you walk:)                monitor, go through the kitchen, and you'll
                                       reach the office where the monitor is."
                                     → "Keep going down the hallway, toward the office."
                                     → "You've reached the office. The monitor is to your left."
```

## Technique — text-space SLAM (TextSLAM)

The XR input is a **monocular** camera (no depth, sparse/haphazard frames), so
this is a **direct port of [TextSLAM](https://github.com/nvddr/textslam)**, a
"text-space" SLAM rather than a geometric one: each frame is perceived into
**text** (caption + objects + OCR/signage), the **pixels are discarded**, and a
semantic-topological place graph is built and relocalized purely from that text
— no poses, no depth, no point cloud. The map is just text, tiny and
human-readable.

The upstream library is carried over **in-process** under `worker/textslam/`:
`types`, `scoring`, `relations`, `landmarks`, and `topomap`
(`SemanticTopoMap` — incremental build, relocalization, loop closure,
consolidation) are the original modules unchanged. Only the two model backends
are XR-native: `VLMPerceptor` perceives via the shared Cosmos VLM
(`xr-ai-models`) instead of Florence-2, and `HashingEmbedder` replaces BGE so
the worker pulls no embedding model. The brain (`worker/agent.py`) is thin glue
that feeds live frames into the map's online SLAM step and supervises naming
from speech.

- **Build** (during the tour): a background loop perceives the live view into a
  `SceneDescription` and calls `SemanticTopoMap.ingest` — TextSLAM's online step
  that, by the *same* embed→score→best operation used for relocalization,
  associates the frame to an existing place or starts a new one (with temporal /
  loop-closure edges). Saying *"this is the living room"* sets the room label
  applied to every node ingested while you scan it — voice-supervised naming
  layered on TextSLAM's otherwise-unsupervised association. `stop tour` runs the
  upstream `consolidate` + `index_landmarks` finalizers.
- **"where am I"** = `SemanticTopoMap.relocalize`: the current view is perceived
  and scored against each place by *text similarity* — caption cosine +
  object-set Jaccard + OCR/signage overlap (OCR weighted heaviest, since a
  unique sign like "KITCHEN" is the strongest anchor), **best-of-observations**
  so a place seen from two angles doesn't fragment. The best place above a
  minimum score wins (a guard against perceptual aliasing); its room label is
  the answer.
- **"where is the sofa"**: the object/sign is found in the stored place text
  (object labels + OCR) → its place's room; if you are standing in that room,
  the agent reads a live `left`/`center`/`right` bearing from the current frame
  → *"in the living room, to your left"*; otherwise just the room.
- **"take me to the monitor from here"** = **guided navigation**. The
  place-node graph that `ingest` builds (a temporal edge from each perceived
  frame to the next, plus loop closures) *is* the route network — no extra data
  structure. The destination is resolved — an **object** ("monitor") to the room
  that contains it (via the stored object/OCR text), or a **room** name — to a
  target node; the current view is relocalized to a start node; and
  `SemanticTopoMap.shortest_path` (BFS) gives the path. The agent speaks the
  room-by-room plan, then — because the wearer is *walking* — a background loop
  re-localizes **each live frame** and narrates progress proactively ("keep going
  down the hallway, toward the office"), **corrects wrong turns** ("that doesn't
  look right — turn around…") by re-routing from the new position, and announces
  arrival with a live bearing to the object ("you've reached the office, the
  monitor is to your left"). Say *"stop"* to cancel. Transit cues ("down the
  hallway", "across the open space", "through the doorway") come from each node's
  caption / portal objects.

The bearing is the one non-TextSLAM affordance (re-acquired live per query); the
*map and localization are pure text*, matching the monocular, pose-free input.

It deliberately reuses the **already-built shared services** — STT (NeMo
Parakeet), VAD (Silero via `xr-ai-vad`), TTS (Piper), and the VLM (Cosmos) —
through `xr-ai-models` + `xr-ai-pipecat.make_voice_pipeline`, exactly like
`simple-vlm-example`. The brain (`worker/agent.py`) is a
`xr_ai_pipecat.BrainProcessor`; the voice gate is configured **always-on** (no
wake word) so plain commands like *"start room tour"* are heard directly.

## Run

```bash
cd agent-samples/room-tour-example
uv sync
uv run room_tour_example
```

This starts the hub, VLM, STT, TTS, and the worker (the place map lives
in-process in the worker — no separate server). Then connect any client
(web/iOS/Android/glasses), **turn on the camera**, and speak the tour commands
above. The camera streams continuously (always-on); the agent only calls the
VLM when it needs to perceive a frame or answer a question.

## Voice commands

| Say | Effect |
|---|---|
| `start room tour` | begin a fresh map |
| `this is the <room>` | label the room you're scanning (e.g. "this is the kitchen") |
| `stop tour` | finalize (consolidate + index) — queries work before this too |
| `where am I` / `what room is this` | identify the current room |
| `where is the <object>` | locate an object (room + live direction if you're there) |
| `take me to the <room/object>` (e.g. "…the monitor from here") | **guided navigation**: spoken room-by-room plan, then live progress/correction as you walk, ending with a bearing to the object |
| `stop` / `cancel` / `we're here` (while guiding) | end the guided navigation |

## Limitations

- Direction is a coarse left/center/right read of the current frame, not metric
  3-D — there is no pose/SLAM. A future variant could consume a pose topic (see
  the `mono-slam-example` proposal) for true bearings and cross-room directions.
- Object recall is only as good as what the VLM named while you panned; scan
  each room slowly and fully.
- Navigation is **topological, not metric**: the route follows the place-node
  edges built during the tour (roughly, retrace the path you walked), and is
  phrased by what each place looks like — it has no floor-plan geometry, so it
  won't find a physical shortcut you never toured. It also relies on
  relocalizing your current view to a start node, so look around if it can't
  tell where you are.
- The map is in-memory and per-session (lost on worker restart).
