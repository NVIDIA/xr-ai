<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai-interaction

**TL;DR:** Let voice and text interfaces work with the same application agents.
Use these shared messages to express participant requests and responses while
keeping the interaction independent of how it is delivered.

`xr-ai-interaction` provides the transport-neutral messages exchanged while a
participant interacts with agents: `UserQuery`, `ResponseOutput`,
`ParticipantJoined`, `ParticipantLeft`, and `Interrupted`. It depends only on
Pydantic.

Input and output boundaries adapt these messages to a medium. `VoiceAgent`
publishes recognized speech as `UserQuery` and speaks `ResponseOutput` from
`VOICE_OUTPUT_TOPIC`; a text front end can publish and render the same
messages. Routing and conversation use them without depending on voice.
Applications choose topic names; the models define only the payloads.

## Reply and update output

A producer declares what kind of output it emits; the consumer decides when it
is rendered. `ResponseOutput.kind` is required:

- `reply` answers one user request and carries that request's `request_id`. A
  consumer renders replies for one request in order and drops output for a
  request that a newer request or an interruption superseded.
- `update` is unprompted information, such as a timer or a monitor reading. It
  has no `request_id`. The consumer renders updates when the participant is not
  being spoken to and may combine several into one. `urgent=True` marks an
  update that cuts current output and plays next; only updates can be urgent.
- `withdraw` carries no content. It asks the consumer to drop the updates the
  producer queued for the participant that have not started playing, for
  example when a notice describes a state that has since changed. Output that is
  already playing and other producers' updates are kept.

A reply is either one message without `response_id`, or a stream of fragments
that share a `response_id`, carry `final=False`, and end with an empty
`final=True` terminator. Updates are always finite. Producers never choose
interrupt flags or stream identifiers themselves.

`ResponseOutputs` frames these messages so agents do not build them by hand.
It wraps anything with an async `publish(topic, message, *, participant_id=...)`
method, such as `AgentRuntime` or the `RuntimeContext` passed to a subscriber:

```python
from xr_ai_interaction import ResponseOutputs

outputs = ResponseOutputs(ctx, VOICE_OUTPUT_TOPIC)

await outputs.update(participant_id, "The timer is done.")
await outputs.update(participant_id, "The oven is too hot.", urgent=True)
await outputs.withdraw(participant_id)

async with outputs.reply(participant_id, request_id=query_id) as reply:
    async for fragment in model_stream:
        await reply.send(fragment)
```

`Reply` publishes each fragment, then an empty terminator when the block exits,
including on an exception or cancellation, unless nothing was published or a
publish failed. Once its task is being canceled, `send` publishes no further
fragments. Pass `closed_errors` to
suppress the error a closed consumer raises for the terminator, and read
`reply.delivered` afterward to learn whether the terminator was published.

The framing contract on `ResponseOutput` in the Python reference below
describes how consumers assemble streamed fragments. Stream tracking and
duplicate handling belong to the consumer. Refer to
{ref}`voice-tuning-and-data-echo` for `VoiceAgent`'s output policy.

## Python API reference

The generated {doc}`Python reference </reference/python/xr_ai_interaction/index>`
documents the exported names and their fields and validation.
