<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai-voice

The public voice-session API for XR agents. Pipecat implements the pipeline,
but applications work with XR concepts rather than Pipecat modules:

- `VoiceSession` owns readiness, hub transport, private pipeline assembly,
  signals, execution, and cleanup. Its default hub transport opens only after
  service probes succeed.
- `VoiceHandler` is an async callable from `VoiceQuery` to text or a text stream.
- `TextMessageInput` routes typed participant messages through the same turn path as speech.
- `HubVoiceTransport` is available when an application needs to share one transport explicitly.

## Usage

```python
from xr_ai_voice import VadConfig, VoiceQuery, VoiceSession
from xr_ai_voicegate import VoiceGateConfig

async def handle(query: VoiceQuery) -> str:
    # query.participant_id / .text / .fresh_match / .timestamp_us
    # timestamp_us is Unix-epoch µs anchored to when the user spoke.
    return f"You said: {query.text}"

session = VoiceSession(
    stt=stt, tts=tts, vad=VadConfig(), voice_gate=VoiceGateConfig(),
)
async with session:            # awaits STT/TTS readiness
    await session.run(handle)  # starts hub IPC, touches ready_file, then runs
```

A handler may also return an `AsyncIterator[str]` to stream the reply token by
token. Typed messages route through the same path via `TextMessageInput`; data
received outside an active `run()` is ignored.

`VoiceSession.run()` accepts participant lifecycle callbacks, a turn observer,
and explicit follow-up policies. `queue_queries` preserves FIFO execution per
participant instead of cancelling the active handler. With
`interrupt_on_supersede`, each queued turn flushes speech left from the
preceding response when it starts. Explicit interruption frames such as stop
cancel the active turn and clear its participant queue. The `on_query_superseded`
callback fires only when a new query actually replaces a still-in-flight turn —
a follow-up that arrives after the previous turn finished, or that is queued, is
not a supersede.

All per-turn state — pending TTS text, the synthesis/order queue, interruption,
and hub flush — is keyed by participant id, so concurrent participants on one
hub never share a buffer or misroute each other's audio; a departing
participant's transport sender is released on leave. NAT applications create
the callable with `xr_ai_nat.adapters.as_voice_handler`; transcript recording
is a separate observer rather than a side effect of function invocation.

When wake phrases and the listening chime are enabled, the VAD/STT stage probes
the opening audio while the user is still speaking. A recognized phrase emits
the chime immediately, while only the final transcript enters the voice gate as
a query. STOP commands use the same early-probe path for immediate interruption.
