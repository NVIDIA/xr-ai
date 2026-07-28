<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai-voice

The public voice-session API for XR agents. Pipecat implements the pipeline,
but applications work with XR concepts rather than Pipecat modules:

- `VoiceSession` owns readiness, hub transport, private pipeline assembly, signals, execution, and cleanup.
- `VoiceHandler` is an async callable from `VoiceQuery` to text or a text stream.
- `TextMessageInput` routes typed participant messages through the same turn path as speech.
- `HubVoiceTransport` is available when an application needs to share one transport explicitly.

`VoiceSession.run()` accepts participant lifecycle callbacks, a turn observer,
and explicit follow-up policies. `queue_queries` preserves FIFO execution per
participant instead of cancelling the active handler. With
`interrupt_on_supersede`, each queued turn flushes speech left from the
preceding response when it starts. Explicit interruption frames such as stop
cancel the active turn and clear its participant queue. NAT applications create
the callable with `xr_ai_nat.adapters.as_voice_handler`; transcript recording
is a separate observer rather than a side effect of function invocation.

When wake phrases and the listening chime are enabled, the VAD/STT stage probes
the opening audio while the user is still speaking. A recognized phrase emits
the chime immediately, while only the final transcript enters the voice gate as
a query. STOP commands use the same early-probe path for immediate interruption.
