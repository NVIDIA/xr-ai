// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const feed = document.querySelector('#feed');
const empty = document.querySelector('#empty');
const template = document.querySelector('#event-template');
const follow = document.querySelector('#follow');
const status = document.querySelector('#status');
const statusDot = document.querySelector('#status-dot');
let lastId = 0;
let filter = 'all';

document.querySelectorAll('.filter').forEach((button) => {
  button.addEventListener('click', () => {
    filter = button.dataset.source;
    document.querySelectorAll('.filter').forEach((item) => item.classList.toggle('active', item === button));
    document.querySelectorAll('.event').forEach((item) => {
      item.hidden = filter !== 'all' && item.dataset.source !== filter;
    });
  });
});

document.querySelector('#clear').addEventListener('click', () => {
  document.querySelectorAll('.event').forEach((item) => item.remove());
  empty.hidden = false;
});

function describe(record) {
  if (record.type === 'utterance') return record.text;
  if (record.type === 'summary') return `Summary: ${record.text}`;
  if (record.type === 'observation') return record.delta;
  if (record.type === 'session') return `Session started${record.participant_id ? ` for ${record.participant_id}` : ''}.`;
  if (record.type === 'session_end') return 'Session ended.';
  return JSON.stringify(record);
}

function addEvent(event) {
  const record = event.record;
  const node = template.content.firstElementChild.cloneNode(true);
  node.dataset.source = event.source_id;
  node.classList.add(event.source_id);
  node.hidden = filter !== 'all' && filter !== event.source_id;
  node.querySelector('.source').textContent = event.source_title;
  node.querySelector('.kind').textContent = record.type || 'event';
  node.querySelector('time').textContent = record.timestamp
    ? new Date(record.timestamp).toLocaleTimeString()
    : '';
  node.querySelector('.primary').textContent = describe(record);
  const details = node.querySelector('details');
  if (record.caption) {
    node.querySelector('.caption').textContent = record.caption;
  } else {
    details.remove();
  }
  feed.append(node);
  empty.hidden = true;
}

async function poll() {
  try {
    const response = await fetch(`/api/events?after=${lastId}`, { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    payload.events.forEach((event) => {
      lastId = Math.max(lastId, event.id);
      addEvent(event);
    });
    status.textContent = 'Live';
    statusDot.classList.add('live');
    if (payload.events.length && follow.checked) window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
  } catch (error) {
    status.textContent = 'Reconnecting';
    statusDot.classList.remove('live');
  } finally {
    window.setTimeout(poll, 500);
  }
}

poll();
