# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Cross-talk isolation suite.

Hard guarantee under test: for any participant *X* connected to the hub,
return-side traffic addressed to *X* (data, audio, audio-flush) reaches
*only* X's connector — never any other client's connector.

The suite exercises this with progressively larger fan-outs:

* 3 clients, 1 agent           — basic isolation under N-way fan-out.
* 1 client, 3 agents           — agents observing the same inbound
                                 stream don't accidentally route each
                                 other's return traffic to peers.
* 3 clients, 3 agents          — full multi/multi matrix; each
                                 (agent → client) link is independent.
* 4 clients with random fan-in — heavy interleaved sends; assert origin
                                 attribution holds end-to-end.
* Late-joiners and leavers     — joining mid-session works; leaving
                                 cleanly stops further return traffic
                                 from reaching the connector.
* Topic-filtered processors    — agents subscribed to disjoint topic
                                 sets never observe each other's events.
"""
from __future__ import annotations

import asyncio
import random
from collections import Counter

import pytest

from xr_ai_agent import AudioChunk, DataMessage, Subscribe

from _helpers import setup_client, silence, teardown_clients, wait_for, wait_for_subscribed

pytestmark = pytest.mark.asyncio


# ── 3 clients, 1 agent ──────────────────────────────────────────────────────


async def test_three_clients_one_agent_no_cross_talk(
    hub, make_connector, make_processor, settle,
):
    """Agent replies to each of three clients individually; each client
    sees exactly one message and only the one addressed to it."""
    clients = [
        await setup_client(make_connector, "alice"),
        await setup_client(make_connector, "bob"),
        await setup_client(make_connector, "carol"),
    ]
    try:
        agent = make_processor()
        await settle()

        for fc in clients:
            await agent.send_return_data(DataMessage(
                participant_id=fc.pid, topic="reply",
                pts_us=0, data=f"hello {fc.pid}".encode(),
            ))

        await wait_for(lambda: all(fc.return_data for fc in clients))
        # Soak window — leak would manifest as an extra message in
        # someone's inbox.
        await asyncio.sleep(0.15)

        # Each client got exactly its own reply.
        for fc in clients:
            assert len(fc.return_data) == 1, (
                f"{fc.pid} saw {len(fc.return_data)} messages — expected 1"
            )
            assert fc.return_data[0].participant_id == fc.pid
            assert fc.return_data[0].data           == f"hello {fc.pid}".encode()
    finally:
        await teardown_clients(clients)


async def test_three_clients_one_agent_audio_no_cross_talk(
    hub, make_connector, make_processor, settle,
):
    """Same as above but for return audio — the per-pid track + subscribe
    permissions invariant must hold at the IPC layer too (the hub publishes
    on `return_audio.{pid}` and only that pid's connector subscribes)."""
    clients = [
        await setup_client(make_connector, "alice"),
        await setup_client(make_connector, "bob"),
        await setup_client(make_connector, "carol"),
    ]
    try:
        agent = make_processor()
        await settle()

        for i, fc in enumerate(clients):
            await agent.send_return_audio(silence(fc.pid, pts_us=10 + i))

        await wait_for(lambda: all(fc.return_audio for fc in clients))
        await asyncio.sleep(0.15)  # leak window

        for fc in clients:
            assert len(fc.return_audio) == 1
            assert fc.return_audio[0].participant_id == fc.pid

        # Sanity: nobody collected a peer's audio.
        for fc in clients:
            for chunk in fc.return_audio:
                assert chunk.participant_id == fc.pid
    finally:
        await teardown_clients(clients)


async def test_three_clients_one_agent_flush_no_cross_talk(
    hub, make_connector, make_processor, settle,
):
    """Per-client flush isolation — flushing alice must not flush bob."""
    clients = [
        await setup_client(make_connector, pid)
        for pid in ("alice", "bob", "carol")
    ]
    by_pid = {fc.pid: fc for fc in clients}

    try:
        agent = make_processor()
        await settle()

        await agent.flush_return_audio("bob")

        await wait_for(lambda: bool(by_pid["bob"].return_audio_flush))
        await asyncio.sleep(0.15)  # leak window

        assert len(by_pid["bob"].return_audio_flush)   == 1
        assert by_pid["bob"].return_audio_flush[0].participant_id == "bob"
        assert by_pid["alice"].return_audio_flush      == []
        assert by_pid["carol"].return_audio_flush      == []
    finally:
        await teardown_clients(clients)


# ── 1 client, 3 agents ──────────────────────────────────────────────────────


async def test_one_client_three_agents_observe_full_stream(
    hub, make_connector, make_processor, settle,
):
    """Three independent agents in the same hub all see every inbound
    message from a single client. No agent silently absorbs another's
    delivery."""
    seen: list[list[DataMessage]] = [[], [], []]

    def cb_for(idx):
        async def _cb(msg): seen[idx].append(msg)
        return _cb

    agents = [make_processor() for _ in range(3)]
    for i, ag in enumerate(agents):
        ag.on_data(cb_for(i))
    await settle()

    fc = await setup_client(make_connector, "alice")
    try:
        for i in range(5):
            await fc.connector.push_data(DataMessage(
                participant_id="alice", topic="chat",
                pts_us=i, data=f"m{i}".encode(),
            ))

        await wait_for(lambda: all(len(s) == 5 for s in seen))

        for s in seen:
            assert [m.data for m in s] == [b"m0", b"m1", b"m2", b"m3", b"m4"]
    finally:
        await teardown_clients([fc])


# ── 3 clients × 3 agents (the matrix) ───────────────────────────────────────


async def test_three_clients_three_agents_full_matrix(
    hub, make_connector, make_processor, settle,
):
    """The canonical multi/multi scenario.

    Every agent receives every inbound message, then each agent emits
    one return-data message addressed to one of the three clients. We
    assert:

      * each agent saw all three inbound messages;
      * each client received exactly the agents' replies addressed to
        it (one per agent → 3 replies);
      * no client sees a reply addressed to anyone else.
    """
    clients = [
        await setup_client(make_connector, pid)
        for pid in ("alice", "bob", "carol")
    ]
    by_pid = {fc.pid: fc for fc in clients}

    seen_inbound: list[list[DataMessage]] = [[], [], []]

    def cb_for(idx):
        async def _cb(msg): seen_inbound[idx].append(msg)
        return _cb

    agents = [make_processor() for _ in range(3)]
    for i, ag in enumerate(agents):
        ag.on_data(cb_for(i))

    # The agents started after the clients connected, so they need the
    # roster replay to learn who's already there. Wait until each agent
    # has auto-subscribed to all three pids before pushing any data.
    await wait_for(lambda: all(
        len(ag.subscribed_participants) == 3 for ag in agents
    ))

    try:
        # Each client speaks once; each agent should observe all three.
        for fc in clients:
            await fc.connector.push_data(DataMessage(
                fc.pid, "chat", 1, f"in-{fc.pid}".encode(),
            ))

        await wait_for(lambda: all(len(s) == 3 for s in seen_inbound))

        for s in seen_inbound:
            pids = {m.participant_id for m in s}
            assert pids == {"alice", "bob", "carol"}

        # Now every agent replies to every client (matrix fan-out).
        for ai, ag in enumerate(agents):
            for fc in clients:
                await ag.send_return_data(DataMessage(
                    participant_id=fc.pid, topic="reply",
                    pts_us=100 + ai, data=f"a{ai}->{fc.pid}".encode(),
                ))

        await wait_for(lambda: all(
            len(fc.return_data) == 3 for fc in clients
        ))
        await asyncio.sleep(0.15)  # leak window

        # Each client got 3 replies — one per agent — all addressed to it.
        for fc in clients:
            assert len(fc.return_data) == 3, (
                f"{fc.pid} got {len(fc.return_data)} replies — expected 3"
            )
            for msg in fc.return_data:
                assert msg.participant_id == fc.pid
            # Senders span all three agents (no two-from-one-agent surprise).
            sender_indices = sorted(
                int(m.data.decode().split("->")[0][1:]) for m in fc.return_data
            )
            assert sender_indices == [0, 1, 2]

    finally:
        await teardown_clients(clients)


# ── 4 clients with random interleaved sends ────────────────────────────────


async def test_four_clients_interleaved_origin_attribution(
    hub, make_connector, make_processor, settle,
):
    """Heavy interleaved fan-in: 4 clients each push 25 messages in
    randomised order; the agent must attribute every message to the
    correct origin and lose none of them."""
    clients = [
        await setup_client(make_connector, pid)
        for pid in ("alice", "bob", "carol", "dave")
    ]

    seen: list[DataMessage] = []
    async def cb(msg): seen.append(msg)

    agent = make_processor()
    agent.on_data(cb)
    await wait_for_subscribed(agent, pids=[fc.pid for fc in clients])

    # Build a globally-interleaved schedule that still keeps each pid's
    # local sequence numbers monotonic (alice's messages still go out
    # 0,1,2,…,24 — they're just spread among other pids' messages).
    rng = random.Random(0xC0FFEE)
    pid_sequence: list[str] = []
    for fc in clients:
        pid_sequence.extend([fc.pid] * 25)
    rng.shuffle(pid_sequence)

    by_pid  = {fc.pid: fc for fc in clients}
    next_i  = {fc.pid: 0  for fc in clients}
    try:
        for pid in pid_sequence:
            i = next_i[pid]
            next_i[pid] += 1
            await by_pid[pid].connector.push_data(DataMessage(
                pid, "chat", i, f"{pid}-{i}".encode(),
            ))

        await wait_for(lambda: len(seen) == 100, timeout=3.0)

        # Origin counts.
        counts = Counter(m.participant_id for m in seen)
        assert counts == {"alice": 25, "bob": 25, "carol": 25, "dave": 25}

        # Per-pid order is preserved (each client's messages appear in
        # send order, even if interleaved with other clients').
        for pid in counts:
            indices = [
                int(m.data.decode().split("-", 1)[1])
                for m in seen if m.participant_id == pid
            ]
            assert indices == list(range(25)), (
                f"{pid} messages reordered: {indices}"
            )

    finally:
        await teardown_clients(clients)


# ── late-join / leave ───────────────────────────────────────────────────────


async def test_late_joining_client_receives_only_post_join_traffic(
    hub, make_connector, make_processor, settle,
):
    """A client joining mid-session does not retroactively see earlier
    messages, and existing clients are not disturbed by the join."""
    alice = await setup_client(make_connector, "alice")
    try:
        agent = make_processor()
        await settle()

        # Send to alice while she's the only one.
        await agent.send_return_data(DataMessage(
            "alice", "reply", 1, b"first",
        ))
        await wait_for(lambda: len(alice.return_data) >= 1)

        # bob joins late.
        bob = await setup_client(make_connector, "bob")

        # Agent now sends to both.
        await agent.send_return_data(DataMessage("alice", "reply", 2, b"second"))
        await agent.send_return_data(DataMessage("bob",   "reply", 3, b"hi-bob"))

        await wait_for(lambda: len(alice.return_data) >= 2 and len(bob.return_data) >= 1)
        await asyncio.sleep(0.15)

        assert [m.data for m in alice.return_data] == [b"first", b"second"]
        assert [m.data for m in bob.return_data]   == [b"hi-bob"]

        await teardown_clients([bob])
    finally:
        await teardown_clients([alice])


async def test_left_participant_receives_no_further_return_traffic(
    hub, make_connector, make_processor, settle,
):
    """After ``notify_participant_left``, the connector unsubscribes from
    return topics. Any further send to that pid is dropped at the hub
    (participant unknown) and even if it weren't, the connector would
    not deliver it."""
    alice = await setup_client(make_connector, "alice")
    bob   = await setup_client(make_connector, "bob")
    try:
        agent = make_processor()
        await settle()

        await alice.connector.notify_participant_left("alice", pts_us=99)
        await asyncio.sleep(0.1)

        await agent.send_return_data(DataMessage(
            "alice", "reply", 1, b"too late",
        ))
        await agent.send_return_data(DataMessage(
            "bob",   "reply", 2, b"bob-still-there",
        ))

        await wait_for(lambda: bool(bob.return_data))
        await asyncio.sleep(0.2)  # confirm alice got nothing

        assert alice.return_data == []
        assert [m.data for m in bob.return_data] == [b"bob-still-there"]
    finally:
        await teardown_clients([alice, bob])


# ── topic-filter isolation ──────────────────────────────────────────────────


async def test_disjoint_topic_filters_no_cross_observation(
    hub, make_connector, make_processor, settle,
):
    """Two agents subscribe to disjoint topic sets; neither should observe
    the other's events even when both fire on the same hub."""
    data_seen:        list[DataMessage] = []
    audio_seen:       list[AudioChunk]  = []
    data_audio_seen:  list[AudioChunk]  = []
    audio_data_seen:  list[DataMessage] = []

    async def cb_data_for_data_agent(m):       data_seen.append(m)
    async def cb_audio_leak_into_data_agent(m): data_audio_seen.append(m)
    async def cb_audio_for_audio_agent(m):     audio_seen.append(m)
    async def cb_data_leak_into_audio_agent(m): audio_data_seen.append(m)

    # Agent A subscribes only to data; agent B only to audio. Each is
    # filtered at the ZMQ kernel via the per-pid Subscribe flag, so
    # neither sees traffic in the other's category.
    data_agent  = make_processor(filter=Subscribe.DATA)
    audio_agent = make_processor(filter=Subscribe.AUDIO)
    data_agent .on_data(cb_data_for_data_agent)
    data_agent .on_audio(cb_audio_leak_into_data_agent)
    audio_agent.on_audio(cb_audio_for_audio_agent)
    audio_agent.on_data(cb_data_leak_into_audio_agent)
    await settle()

    fc = await setup_client(make_connector, "alice")
    try:
        await fc.connector.push_data(DataMessage(
            "alice", "chat", 1, b"hi",
        ))
        await fc.connector.push_audio(silence("alice", pts_us=2))

        await wait_for(lambda: data_seen and audio_seen)
        await asyncio.sleep(0.15)

        assert [m.data for m in data_seen] == [b"hi"]
        assert len(audio_seen) == 1
        # Cross-leakage:
        assert data_audio_seen == []
        assert audio_data_seen == []
    finally:
        await teardown_clients([fc])
