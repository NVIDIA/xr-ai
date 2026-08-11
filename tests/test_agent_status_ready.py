# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
The client-facing readiness contract on ``_agent.status``.

Runs against a real HubEndpoint + ConnectorEndpoint + ProcessorEndpoint stack
over ``ipc://`` — no GPU, no Docker.

Three properties matter, all protocol-level rather than cosmetic:

  * ``ready`` means *routable*. A request sent the instant a client sees
    ``ready`` must reach the agent, so the announcement has to sit behind a
    confirmed subscription rather than race it.
  * ``ready`` is the room's state, not one agent's. The hub folds every
    responsible agent's state into the single status a client sees.
  * Only agents that opt into readiness, and only for participants they are
    subscribed to, get a say. A passive processor or an agent scoped to
    another pid is neither counted against a client nor allowed to speak
    for one.
"""
from __future__ import annotations

import asyncio

import pytest
from xr_ai_hub import DataMessage, Subscribe

from _helpers import setup_client, teardown_clients, wait_for

pytestmark = pytest.mark.asyncio


# ── ready implies a live subscription ────────────────────────────────────────


async def test_join_before_any_agent_reports_loading(hub, make_connector, settle):
    alice = await setup_client(make_connector, "alice")
    try:
        await wait_for(lambda: bool(alice.statuses))
        assert alice.statuses == ["loading"]
    finally:
        await teardown_clients([alice])


async def test_ready_is_withheld_until_the_subscription_is_confirmed(
    hub, make_connector, make_processor, settle,
):
    """``ready`` must not leave the endpoint before its subscription is live.

    The window is driven here rather than raced: over ``ipc://`` the SUBSCRIBE
    reaches the hub far sooner than the two round trips it takes ``ready`` to
    reach a client and a request to come back, so natural timing never exposes
    the gap that a loaded or remote hub would. Holding the barrier open shows
    the ordering the protocol relies on.
    """
    confirmed = asyncio.Event()

    agent = make_processor(announces_readiness=True)

    async def gated_wait(*_a, **_kw) -> bool:
        await confirmed.wait()
        return True

    agent.wait_for_subscriptions = gated_wait
    await agent.wait_until_running()
    await agent.mark_ready()

    alice = await setup_client(make_connector, "alice")
    try:
        await wait_for(lambda: "alice" in agent.subscribed_participants)
        await settle()
        assert alice.statuses == ["loading"]

        confirmed.set()
        await wait_for(lambda: alice.statuses[-1:] == ["ready"])
        assert alice.statuses[-1] == "ready"
    finally:
        await teardown_clients([alice])


async def test_request_sent_immediately_on_ready_is_not_dropped(
    hub, make_connector, make_processor, settle,
):
    """End-to-end guard: a request issued the instant ``ready`` lands arrives."""
    seen: list[DataMessage] = []
    async def on_data(msg): seen.append(msg)

    agent = make_processor(announces_readiness=True)
    agent.on_data(on_data)
    await agent.wait_until_running()
    await agent.mark_ready()

    alice = await setup_client(make_connector, "alice")
    try:
        await wait_for(lambda: alice.statuses[-1:] == ["ready"])
        assert alice.statuses[-1] == "ready"

        # No settle: this is the first thing the client does on seeing ready.
        await alice.connector.push_data(
            DataMessage("alice", "chat", 1, b"first-request"),
        )
        await wait_for(lambda: bool(seen))
        assert [m.data for m in seen] == [b"first-request"]
    finally:
        await teardown_clients([alice])


async def test_ready_waits_for_the_subscription_it_depends_on(
    hub, make_connector, make_processor, settle,
):
    """The endpoint has a confirmed subscription for the pid by the time it
    announces availability."""
    agent = make_processor(announces_readiness=True)
    await agent.wait_until_running()
    await agent.mark_ready()

    alice = await setup_client(make_connector, "alice")
    try:
        await wait_for(lambda: alice.statuses[-1:] == ["ready"])
        assert "alice" in agent.subscribed_participants
        assert await agent.wait_for_subscriptions(timeout=0.5)
    finally:
        await teardown_clients([alice])


# ── ready is the room's state, not one agent's ───────────────────────────────


async def test_one_ready_agent_does_not_mask_a_loading_peer(
    hub, make_connector, make_processor, settle,
):
    """A second agent that has not reported keeps the room at ``loading``."""
    ready_agent = make_processor(announces_readiness=True, agent_id="ready-agent")
    slow_agent  = make_processor(announces_readiness=True, agent_id="slow-agent")
    await ready_agent.wait_until_running()
    await slow_agent.wait_until_running()
    await ready_agent.mark_ready()

    alice = await setup_client(make_connector, "alice")
    try:
        await wait_for(lambda: len(alice.statuses) >= 2, timeout=0.6)
        assert alice.statuses == ["loading"]

        await slow_agent.mark_ready()
        await wait_for(lambda: alice.statuses[-1:] == ["ready"])
        assert alice.statuses[-1] == "ready"
    finally:
        await teardown_clients([alice])


async def test_ready_agent_reannouncement_cannot_clear_a_busy_peer(
    hub, make_connector, make_processor, settle,
):
    """Agent A's periodic re-announcement must not overwrite agent B's
    ``processing`` with ``ready``."""
    idle_agent = make_processor(announces_readiness=True, agent_id="idle-agent")
    busy_agent = make_processor(announces_readiness=True, agent_id="busy-agent")
    await idle_agent.wait_until_running()
    await busy_agent.wait_until_running()
    await idle_agent.mark_ready()
    await busy_agent.mark_ready()

    alice = await setup_client(make_connector, "alice")
    try:
        await wait_for(lambda: alice.statuses[-1:] == ["ready"])

        await busy_agent.set_status("processing", "alice")
        await wait_for(lambda: alice.statuses[-1:] == ["processing"])

        # This is what the 2 s re-announce loop does on the idle agent.
        for _ in range(3):
            await idle_agent.republish_statuses()
            await settle()
        assert alice.statuses[-1] == "processing"

        await busy_agent.set_status("ready", "alice")
        await wait_for(lambda: alice.statuses[-1:] == ["ready"])
        assert alice.statuses[-1] == "ready"
    finally:
        await teardown_clients([alice])


async def test_detached_agent_stops_holding_the_room_back(
    hub, make_connector, make_processor, settle,
):
    """An agent that exits is no longer counted against room availability."""
    ready_agent = make_processor(announces_readiness=True, agent_id="ready-agent")
    slow_agent  = make_processor(announces_readiness=True, agent_id="slow-agent")
    await ready_agent.wait_until_running()
    await slow_agent.wait_until_running()
    await ready_agent.mark_ready()

    alice = await setup_client(make_connector, "alice")
    try:
        await wait_for(lambda: alice.statuses == ["loading"], timeout=0.6)
        assert alice.statuses == ["loading"]

        slow_agent.stop()

        await wait_for(lambda: alice.statuses[-1:] == ["ready"])
        assert alice.statuses[-1] == "ready"
    finally:
        await teardown_clients([alice])


# ── readiness participation and scope ────────────────────────────────────────


async def test_passive_processor_does_not_hold_the_room_at_loading(
    hub, make_connector, make_processor, settle,
):
    """A processor that never reports status must not gate readiness.

    ProcessorEndpoint is the generic downstream endpoint — video-MCP,
    analytics, recorders. None of them call set_status, so counting them as
    unreported agents would pin every client at loading forever.
    """
    voice = make_processor(announces_readiness=True, agent_id="voice")
    make_processor(agent_id="video-mcp", filter=Subscribe.VIDEO)
    await voice.wait_until_running()
    await voice.mark_ready()

    alice = await setup_client(make_connector, "alice")
    try:
        await wait_for(lambda: alice.statuses[-1:] == ["ready"])
        assert alice.statuses[-1] == "ready"
    finally:
        await teardown_clients([alice])


async def test_fixed_pid_agent_says_nothing_about_an_unrelated_client(
    hub, make_connector, make_processor, settle,
):
    """An endpoint pinned to one pid must not mark another client ready.

    The subscription barrier proves the SUBSCRIBEs this endpoint issued are
    live; it cannot prove one was issued for *this* participant. Without the
    scope check, bob is told ready and his first request is dropped for want
    of a ``data.bob.*`` subscription.
    """
    seen: list[DataMessage] = []
    async def on_data(msg): seen.append(msg)

    agent = make_processor(auto_subscribe=False, announces_readiness=True,
                           agent_id="alice-only")
    agent.on_data(on_data)
    agent.subscribe("alice")
    await agent.wait_until_running()
    await agent.mark_ready()

    bob = await setup_client(make_connector, "bob")
    try:
        await settle()
        assert bob.statuses == ["loading"], f"bob got {bob.statuses}"

        await bob.connector.push_data(DataMessage("bob", "chat", 1, b"hi"))
        await wait_for(lambda: bool(seen), timeout=0.5)
        assert seen == []
    finally:
        await teardown_clients([bob])


async def test_scope_widens_when_the_agent_subscribes(
    hub, make_connector, make_processor, settle,
):
    """Subscribing to a new pid re-announces scope, so that pid can go ready."""
    agent = make_processor(auto_subscribe=False, announces_readiness=True,
                           agent_id="pinned")
    agent.subscribe("alice")
    await agent.wait_until_running()
    await agent.mark_ready()

    bob = await setup_client(make_connector, "bob")
    try:
        await settle()
        assert bob.statuses == ["loading"]

        agent.subscribe("bob")
        await agent.set_status("ready", "bob")
        await wait_for(lambda: bob.statuses[-1:] == ["ready"])
        assert bob.statuses[-1] == "ready"
    finally:
        await teardown_clients([bob])


async def test_agent_scoped_elsewhere_does_not_hold_this_client_back(
    hub, make_connector, make_processor, settle,
):
    """Bob's readiness ignores an agent pinned to alice.

    This is the aggregation half of scope: the alice-only agent will never
    report for bob, so counting it among bob's responsible agents would leave
    bob at ``loading`` forever even with a ready general agent present.
    """
    general = make_processor(announces_readiness=True, agent_id="general")
    pinned  = make_processor(auto_subscribe=False, announces_readiness=True,
                             agent_id="alice-only")
    pinned.subscribe("alice")
    await general.wait_until_running()
    await pinned.wait_until_running()
    await general.mark_ready()

    bob = await setup_client(make_connector, "bob")
    try:
        await wait_for(lambda: bob.statuses[-1:] == ["ready"])
        assert bob.statuses[-1] == "ready"
    finally:
        await teardown_clients([bob])
