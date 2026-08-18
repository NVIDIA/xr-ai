// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const eventTemplate = document.querySelector('#event-template');
const topicTemplate = document.querySelector('#topic-template');
const grid = document.querySelector('#topic-grid');
const empty = document.querySelector('#empty');
const follow = document.querySelector('#follow');
const participantFilter = document.querySelector('#participant-filter');
const status = document.querySelector('#status');
const statusDot = document.querySelector('#status-dot');
const viewerTitle = document.querySelector('#viewer-title');

const MAX_BROWSER_EVENTS = 2000;
let events = [];
let cursor = 0;

function participantName(event) {
  return event.participant_id || 'Global';
}

function presentation(event) {
  const payload = event.payload;
  const candidates = ['text', 'message', 'response', 'summary', 'caption', 'delta', 'meter_reading'];
  for (const key of candidates) {
    if (typeof payload[key] === 'string' && payload[key]) return payload[key];
  }
  const kind = payload.record_type || payload.event_type || payload.type || payload.event;
  return typeof kind === 'string' ? kind.replaceAll('_', ' ') : JSON.stringify(payload);
}

function refreshParticipantOptions() {
  const selected = participantFilter.value;
  const names = [...new Set(events.map(participantName))].sort();
  participantFilter.replaceChildren(new Option('All', '*'));
  names.forEach((name) => participantFilter.add(new Option(name, name)));
  participantFilter.value = names.includes(selected) || selected === '*' ? selected : '*';
}

function topicKey(event) {
  return `${participantName(event)}\u0000${event.topic}`;
}

function render() {
  refreshParticipantOptions();
  grid.querySelectorAll('.topic-pane').forEach((pane) => pane.remove());
  const selected = participantFilter.value;
  const visible = selected === '*' ? events : events.filter((event) => participantName(event) === selected);
  const groups = new Map();
  visible.forEach((event) => {
    const key = topicKey(event);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(event);
  });
  empty.hidden = groups.size > 0;

  [...groups.entries()].sort(([left], [right]) => left.localeCompare(right)).forEach(([, topicEvents]) => {
    const latest = topicEvents[topicEvents.length - 1];
    const pane = topicTemplate.content.firstElementChild.cloneNode(true);
    pane.querySelector('h2').textContent = latest.title || latest.topic;
    pane.querySelector('.count').textContent = String(topicEvents.length);
    const feed = pane.querySelector('.feed');
    topicEvents.forEach((event) => {
      const node = eventTemplate.content.firstElementChild.cloneNode(true);
      node.querySelector('.participant').textContent = `${participantName(event)} · ${event.source}`;
      node.querySelector('time').textContent = new Date(event.timestamp_us / 1000).toLocaleTimeString();
      node.querySelector('.primary').textContent = presentation(event);
      node.querySelector('pre').textContent = JSON.stringify(event.payload, null, 2);
      feed.append(node);
    });
    grid.append(pane);
    if (follow.checked) feed.scrollTop = feed.scrollHeight;
  });
}

document.querySelector('#clear').addEventListener('click', () => {
  events = [];
  render();
});
participantFilter.addEventListener('change', render);

async function poll() {
  try {
    const response = await fetch(`/api/events?after=${cursor}`, { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    viewerTitle.textContent = payload.title;
    document.title = payload.title;
    if (payload.reset) {
      events = [];
      render();
    }
    if (payload.events.length) {
      events.push(...payload.events);
      if (events.length > MAX_BROWSER_EVENTS) events = events.slice(-MAX_BROWSER_EVENTS);
      render();
    }
    cursor = payload.cursor;
    status.textContent = 'Live';
    statusDot.classList.add('live');
  } catch (_error) {
    status.textContent = 'Reconnecting';
    statusDot.classList.remove('live');
  } finally {
    window.setTimeout(poll, 500);
  }
}

poll();
