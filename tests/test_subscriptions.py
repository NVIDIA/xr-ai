"""
Tests for the participant-keyed subscription API on ProcessorEndpoint.

Covers:
* ``Subscribe.ALL`` / ``Subscribe.AUDIO`` / ``Subscribe.DATA`` / ``Subscribe.VIDEO``
  filters at the ZMQ kernel.
* ``auto_subscribe=True`` (default): agents auto-subscribe to every joining
  participant; auto-unsubscribe on leave.
* ``auto_subscribe=False``: agents see only participant + control until
  they call ``subscribe(pid)`` explicitly.
* Roster catch-up: an endpoint started mid-session learns about already-
  connected participants via ``request_roster()``.
* Per-pid filter override and idempotent re-subscription.
* Subscribe-before-join works (ZMQ holds the SUBSCRIBE).
* Pid-prefix collision: ``alice`` and ``alice2`` traffic does not bleed
  across thanks to the trailing-dot in the topic prefix.
"""
from __future__ import annotations

import asyncio

import pytest

from xr_ai_agent import AudioChunk, DataMessage, Subscribe

from _helpers import setup_client, silence, teardown_clients, wait_for, wait_for_subscribed

pytestmark = pytest.mark.asyncio


# ── auto_subscribe=True (default) ──────────────────────────────────────────


async def test_auto_subscribe_tracks_join_and_leave(
    hub, make_connector, make_processor, settle,
):
    """Default ``auto_subscribe=True``: subscribed_participants follows
    the connected_participants set automatically."""
    agent = make_processor()
    await settle()

    assert agent.subscribed_participants == frozenset()

    alice = await setup_client(make_connector, "alice")
    await wait_for(lambda: agent.subscribed_participants == {"alice"})

    bob = await setup_client(make_connector, "bob")
    await wait_for(lambda: agent.subscribed_participants == {"alice", "bob"})

    await alice.connector.notify_participant_left("alice", pts_us=99)
    await wait_for(lambda: agent.subscribed_participants == {"bob"})

    await teardown_clients([alice, bob])


# ── auto_subscribe=False ────────────────────────────────────────────────────


async def test_no_auto_subscribe_filters_data_until_explicit_subscribe(
    hub, make_connector, make_processor, settle,
):
    """``auto_subscribe=False``: agent sees participant events, but no
    data / audio / video until it calls ``subscribe(pid)`` itself."""
    seen: list[DataMessage] = []
    async def cb(msg): seen.append(msg)

    agent = make_processor(auto_subscribe=False)
    agent.on_data(cb)
    await settle()

    fc = await setup_client(make_connector, "alice")
    try:
        # No subscribe() called yet → kernel-level filter drops these.
        await fc.connector.push_data(DataMessage("alice", "chat", 1, b"early"))
        await asyncio.sleep(0.2)
        assert seen == []

        # Now subscribe; subsequent pushes are received.
        agent.subscribe("alice")
        await asyncio.sleep(0.05)  # let SUBSCRIBE propagate

        await fc.connector.push_data(DataMessage("alice", "chat", 2, b"late"))
        await wait_for(lambda: bool(seen))

        assert [m.data for m in seen] == [b"late"]
    finally:
        await teardown_clients([fc])


async def test_explicit_subscribe_picks_only_named_pid(
    hub, make_connector, make_processor, settle,
):
    """Single-client agent: ``auto_subscribe=False`` + explicit
    ``subscribe("alice")`` — bob's data never reaches the agent."""
    seen: list[DataMessage] = []
    async def cb(msg): seen.append(msg)

    agent = make_processor(auto_subscribe=False)
    agent.on_data(cb)
    agent.subscribe("alice")
    await asyncio.sleep(0.1)  # SUBSCRIBE propagation

    alice = await setup_client(make_connector, "alice")
    bob   = await setup_client(make_connector, "bob")
    try:
        await alice.connector.push_data(DataMessage("alice", "chat", 1, b"alice-msg"))
        await bob  .connector.push_data(DataMessage("bob",   "chat", 2, b"bob-msg"))

        await wait_for(lambda: bool(seen))
        await asyncio.sleep(0.15)  # leak window

        assert [m.data for m in seen] == [b"alice-msg"]
        assert agent.subscribed_participants == frozenset({"alice"})
    finally:
        await teardown_clients([alice, bob])


# ── filter modes ────────────────────────────────────────────────────────────


async def test_filter_audio_only_drops_data_at_kernel(
    hub, make_connector, make_processor, settle,
):
    """``filter=Subscribe.AUDIO`` — data and video are filtered at the
    ZMQ kernel; only audio reaches the dispatch loop."""
    saw_data:  list[DataMessage] = []
    saw_audio: list[AudioChunk]  = []

    async def cb_data(m):  saw_data.append(m)
    async def cb_audio(m): saw_audio.append(m)

    agent = make_processor(filter=Subscribe.AUDIO)
    agent.on_data(cb_data)
    agent.on_audio(cb_audio)
    await settle()

    fc = await setup_client(make_connector, "alice")
    try:
        await fc.connector.push_data(DataMessage("alice", "chat", 1, b"hi"))
        await fc.connector.push_audio(silence("alice", pts_us=2))

        await wait_for(lambda: bool(saw_audio))
        await asyncio.sleep(0.15)

        assert saw_data  == []
        assert len(saw_audio) == 1
    finally:
        await teardown_clients([fc])


async def test_filter_combination_data_plus_audio(
    hub, make_connector, make_processor, settle,
):
    """Combine flags with ``|`` — agent sees data and audio, no video."""
    saw_data:  list[DataMessage] = []
    saw_audio: list[AudioChunk]  = []

    async def cb_data(m):  saw_data.append(m)
    async def cb_audio(m): saw_audio.append(m)

    agent = make_processor(filter=Subscribe.DATA | Subscribe.AUDIO)
    agent.on_data(cb_data)
    agent.on_audio(cb_audio)
    await settle()

    fc = await setup_client(make_connector, "alice")
    try:
        await fc.connector.push_data(DataMessage("alice", "chat", 1, b"x"))
        await fc.connector.push_audio(silence("alice", pts_us=2))

        await wait_for(lambda: saw_data and saw_audio)

        assert len(saw_data)  == 1
        assert len(saw_audio) == 1
    finally:
        await teardown_clients([fc])


# ── per-pid filter override ────────────────────────────────────────────────


async def test_per_pid_filter_override_at_subscribe_time(
    hub, make_connector, make_processor, settle,
):
    """The default filter is ``Subscribe.ALL`` but a per-pid call to
    ``subscribe(pid, filter=...)`` overrides it just for that pid."""
    saw_alice_data:  list[DataMessage] = []
    saw_alice_audio: list[AudioChunk]  = []
    saw_bob_data:    list[DataMessage] = []
    saw_bob_audio:   list[AudioChunk]  = []

    async def cb_data(m):
        (saw_alice_data if m.participant_id == "alice" else saw_bob_data).append(m)
    async def cb_audio(m):
        (saw_alice_audio if m.participant_id == "alice" else saw_bob_audio).append(m)

    agent = make_processor(auto_subscribe=False)
    agent.on_data(cb_data)
    agent.on_audio(cb_audio)
    # alice → data only; bob → everything.
    agent.subscribe("alice", filter=Subscribe.DATA)
    agent.subscribe("bob")
    await asyncio.sleep(0.1)

    alice = await setup_client(make_connector, "alice")
    bob   = await setup_client(make_connector, "bob")
    try:
        await alice.connector.push_data(DataMessage("alice", "chat", 1, b"alice-data"))
        await alice.connector.push_audio(silence("alice", pts_us=2))
        await bob  .connector.push_data(DataMessage("bob",   "chat", 3, b"bob-data"))
        await bob  .connector.push_audio(silence("bob",   pts_us=4))

        await wait_for(lambda: saw_alice_data and saw_bob_data and saw_bob_audio)
        await asyncio.sleep(0.15)  # leak window for alice's audio

        assert [m.data for m in saw_alice_data]  == [b"alice-data"]
        assert saw_alice_audio                   == []   # filtered
        assert [m.data for m in saw_bob_data]    == [b"bob-data"]
        assert len(saw_bob_audio)                == 1
    finally:
        await teardown_clients([alice, bob])


async def test_resubscribe_with_different_filter_updates_live_subscriptions(
    hub, make_connector, make_processor, settle,
):
    """Calling ``subscribe(pid, filter=A)`` then ``subscribe(pid, filter=B)``
    diffs the active subscriptions — old categories drop out, new ones
    come in."""
    saw_data:  list[DataMessage] = []
    saw_audio: list[AudioChunk]  = []

    async def cb_data(m):  saw_data.append(m)
    async def cb_audio(m): saw_audio.append(m)

    agent = make_processor(auto_subscribe=False)
    agent.on_data(cb_data)
    agent.on_audio(cb_audio)
    agent.subscribe("alice", filter=Subscribe.DATA)
    await asyncio.sleep(0.1)

    fc = await setup_client(make_connector, "alice")
    try:
        # Phase 1: data only.
        await fc.connector.push_data(DataMessage("alice", "chat", 1, b"phase1"))
        await fc.connector.push_audio(silence("alice", pts_us=2))
        await wait_for(lambda: bool(saw_data))
        await asyncio.sleep(0.15)
        assert [m.data for m in saw_data] == [b"phase1"]
        assert saw_audio == []

        # Phase 2: switch to audio only.
        agent.subscribe("alice", filter=Subscribe.AUDIO)
        await asyncio.sleep(0.1)

        await fc.connector.push_data(DataMessage("alice", "chat", 3, b"phase2-data"))
        await fc.connector.push_audio(silence("alice", pts_us=4))
        await wait_for(lambda: bool(saw_audio))
        await asyncio.sleep(0.15)

        # Data did not grow; audio appeared.
        assert [m.data for m in saw_data] == [b"phase1"]   # unchanged
        assert len(saw_audio) == 1
    finally:
        await teardown_clients([fc])


async def test_unsubscribe_stops_all_traffic_for_pid(
    hub, make_connector, make_processor, settle,
):
    """After ``unsubscribe(pid)`` the agent receives nothing from that
    participant, but other pids continue normally."""
    saw_alice: list[DataMessage] = []
    saw_bob:   list[DataMessage] = []

    async def cb(m):
        (saw_alice if m.participant_id == "alice" else saw_bob).append(m)

    agent = make_processor()  # default auto_subscribe=True
    agent.on_data(cb)
    await settle()

    alice = await setup_client(make_connector, "alice")
    bob   = await setup_client(make_connector, "bob")
    await wait_for(lambda: agent.subscribed_participants == {"alice", "bob"})

    try:
        await alice.connector.push_data(DataMessage("alice", "chat", 1, b"a1"))
        await bob  .connector.push_data(DataMessage("bob",   "chat", 2, b"b1"))
        await wait_for(lambda: saw_alice and saw_bob)

        # Drop alice.
        agent.unsubscribe("alice")
        await asyncio.sleep(0.1)
        assert "alice" not in agent.subscribed_participants

        await alice.connector.push_data(DataMessage("alice", "chat", 3, b"a2"))
        await bob  .connector.push_data(DataMessage("bob",   "chat", 4, b"b2"))

        await wait_for(lambda: len(saw_bob) == 2)
        await asyncio.sleep(0.15)

        assert [m.data for m in saw_alice] == [b"a1"]   # unchanged
        assert [m.data for m in saw_bob]   == [b"b1", b"b2"]
    finally:
        await teardown_clients([alice, bob])


# ── roster catch-up ─────────────────────────────────────────────────────────


async def test_roster_catch_up_for_late_starting_agent(
    hub, make_connector, make_processor, settle,
):
    """An endpoint created *after* clients joined still discovers them —
    ``request_roster`` is fired automatically inside ``run()``."""
    alice = await setup_client(make_connector, "alice")
    bob   = await setup_client(make_connector, "bob")

    seen: list[DataMessage] = []
    async def cb(m): seen.append(m)

    # Agent created mid-session.
    agent = make_processor()
    agent.on_data(cb)

    # Without explicit roster catch-up the agent would miss alice/bob.
    await wait_for_subscribed(agent, pids=["alice", "bob"])

    try:
        await alice.connector.push_data(DataMessage("alice", "chat", 1, b"a"))
        await bob  .connector.push_data(DataMessage("bob",   "chat", 2, b"b"))

        await wait_for(lambda: len(seen) == 2)

        assert {m.participant_id for m in seen} == {"alice", "bob"}
    finally:
        await teardown_clients([alice, bob])


# ── subscribe before join ──────────────────────────────────────────────────


async def test_subscribe_before_pid_connects(
    hub, make_connector, make_processor, settle,
):
    """Calling ``subscribe(pid)`` before that participant exists is
    fine — ZMQ holds the SUBSCRIBE on the socket and the agent receives
    traffic the moment the pid joins."""
    seen: list[DataMessage] = []
    async def cb(m): seen.append(m)

    agent = make_processor(auto_subscribe=False)
    agent.on_data(cb)
    agent.subscribe("alice")  # alice doesn't exist yet
    await asyncio.sleep(0.1)

    fc = await setup_client(make_connector, "alice")
    try:
        await fc.connector.push_data(DataMessage("alice", "chat", 1, b"hello"))
        await wait_for(lambda: bool(seen))

        assert [m.data for m in seen] == [b"hello"]
    finally:
        await teardown_clients([fc])


# ── trailing-dot prefix correctness ────────────────────────────────────────


async def test_pid_prefix_collision_does_not_leak(
    hub, make_connector, make_processor, settle,
):
    """``alice`` and ``alice2`` are distinct prefixes thanks to the
    trailing dot in the SUBSCRIBE — neither sees the other's traffic."""
    seen_alice:  list[DataMessage] = []
    seen_alice2: list[DataMessage] = []

    async def cb(m):
        (seen_alice if m.participant_id == "alice" else seen_alice2).append(m)

    agent = make_processor(auto_subscribe=False)
    agent.on_data(cb)
    agent.subscribe("alice")  # NOT alice2
    await asyncio.sleep(0.1)

    alice  = await setup_client(make_connector, "alice")
    alice2 = await setup_client(make_connector, "alice2")
    try:
        await alice .connector.push_data(DataMessage("alice",  "chat", 1, b"a"))
        await alice2.connector.push_data(DataMessage("alice2", "chat", 2, b"a2"))

        await wait_for(lambda: bool(seen_alice))
        await asyncio.sleep(0.2)  # leak window

        assert [m.data for m in seen_alice]  == [b"a"]
        assert seen_alice2 == []  # alice2's traffic must not bleed in
    finally:
        await teardown_clients([alice, alice2])


# ── idempotency ─────────────────────────────────────────────────────────────


async def test_subscribe_is_idempotent(
    hub, make_connector, make_processor, settle,
):
    """Subscribing twice with the same filter → still one logical sub.
    Single message in, single dispatch out."""
    seen: list[DataMessage] = []
    async def cb(m): seen.append(m)

    agent = make_processor(auto_subscribe=False)
    agent.on_data(cb)
    agent.subscribe("alice")
    agent.subscribe("alice")  # second call, no-op
    agent.subscribe("alice")  # third
    await asyncio.sleep(0.1)

    fc = await setup_client(make_connector, "alice")
    try:
        await fc.connector.push_data(DataMessage("alice", "chat", 1, b"once"))
        await wait_for(lambda: bool(seen))
        await asyncio.sleep(0.15)

        assert len(seen) == 1, f"Expected 1 dispatch, got {len(seen)}"
    finally:
        await teardown_clients([fc])


async def test_unsubscribe_unknown_pid_is_no_op(
    hub, make_connector, make_processor, settle,
):
    agent = make_processor(auto_subscribe=False)
    await settle()
    agent.unsubscribe("nobody")  # must not raise
    assert agent.subscribed_participants == frozenset()
