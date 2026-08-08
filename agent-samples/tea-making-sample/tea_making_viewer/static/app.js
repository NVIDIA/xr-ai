// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const template = document.querySelector('#event-template');
const follow = document.querySelector('#follow');
const status = document.querySelector('#status');
const statusDot = document.querySelector('#status-dot');
const panes = Object.fromEntries([...document.querySelectorAll('.pane')].map((pane) => [
  pane.dataset.source,
  {
    feed: pane.querySelector('.feed'),
    empty: pane.querySelector('.empty'),
    count: pane.querySelector('.count'),
    total: 0,
  },
]));
let lastId = 0;

document.querySelector('#clear').addEventListener('click', () => {
  Object.values(panes).forEach((pane) => {
    pane.feed.querySelectorAll('.event').forEach((item) => item.remove());
    pane.empty.hidden = false;
    pane.total = 0;
    pane.count.textContent = '0';
  });
});

function agentDescription(record) {
  const fields = (record.updates && Object.keys(record.updates).join(', ')) || '';
  const descriptions = {
    'agent.foreground.request': `The ${record.foreground || 'foreground'} agent is handling a request.`,
    'agent.foreground.response': `The ${record.foreground || 'foreground'} agent completed the turn.`,
    'agent.foreground.retry': 'The agent is correcting invalid tool arguments.',
    'agent.observe.request': `Analyzing the latest view for ${record.step || 'the current step'}.`,
    'agent.observe.response': `Finished visual analysis for ${record.step || 'the current step'}.`,
    'agent.observe.retry': 'Correcting an invalid observation tool call.',
    'agent.observe.skipped': 'Skipped an observation that could not be repaired.',
    'agent.background.request': `${record.application || 'A background agent'} is evaluating new input.`,
    'agent.background.response': `${record.application || 'A background agent'} completed its evaluation.`,
    'desktop.route': `Routed this turn to ${record.foreground || 'the root agent'}.`,
    'rag.lookup.request': 'Searching the tea reference material.',
    'rag.lookup.response': `Reference search completed${record.latency_ms ? ` in ${record.latency_ms} milliseconds` : ''}.`,
    'step.commit': `Updated ${record.step || 'the current step'}${fields ? `: ${fields}` : ''}.`,
    'step.commit_rejected': `Rejected an unsupported update${record.reason ? `: ${record.reason}` : ''}.`,
    'step.ready': `${record.step || 'The current step'} is ready for manual advancement.`,
    'step.enter': `Entered ${record.step || 'a new tea step'}.`,
    'trigger.request': `Checking the current view for ${record.step || 'the active task'}.`,
    'trigger.response': `Current-view check completed${record.latency_ms ? ` in ${record.latency_ms} milliseconds` : ''}.`,
    'voice.complete': 'The agent finished responding to the user.',
    'workflow.reset': 'Reset the tea workflow.',
    'workflow.complete': 'Completed the tea workflow.',
  };
  return descriptions[record.event] || record.event.replaceAll('.', ' ').replaceAll('_', ' ');
}

function describe(source, record) {
  if (source === 'agent') return agentDescription(record);
  if (source === 'change_watch') {
    if (record.type === 'session') return `Watching for: ${record.watch_for}`;
    if (record.type === 'baseline') return `Baseline: ${record.caption}`;
    if (record.type === 'observation') {
      return record.important ? record.summary : 'No important change detected.';
    }
    if (record.type === 'session_end') return 'Watcher stopped.';
  }
  if (record.type === 'utterance') return record.text;
  if (record.type === 'summary') return `Summary: ${record.text}`;
  if (record.type === 'observation') return record.delta;
  if (record.type === 'session') return 'Session started.';
  if (record.type === 'session_end') return 'Session ended.';
  return JSON.stringify(record);
}

function details(source, record) {
  if (source === 'video_log' && record.caption) return record.caption;
  if (source === 'agent') {
    const value = { ...record };
    delete value.timestamp;
    return JSON.stringify(value, null, 2);
  }
  if (source === 'change_watch') return JSON.stringify(record, null, 2);
  return '';
}

function addEvent(event) {
  const pane = panes[event.source_id];
  if (!pane) return;
  const record = event.record;
  const node = template.content.firstElementChild.cloneNode(true);
  node.querySelector('.kind').textContent = record.type || record.event || 'event';
  node.querySelector('time').textContent = record.timestamp
    ? new Date(record.timestamp).toLocaleTimeString()
    : '';
  node.querySelector('.primary').textContent = describe(event.source_id, record);
  const detail = details(event.source_id, record);
  if (detail) node.querySelector('pre').textContent = detail;
  else node.querySelector('details').remove();
  pane.feed.append(node);
  pane.empty.hidden = true;
  pane.total += 1;
  pane.count.textContent = pane.total;
  if (follow.checked) pane.feed.scrollTop = pane.feed.scrollHeight;
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
  } catch (error) {
    status.textContent = 'Reconnecting';
    statusDot.classList.remove('live');
  } finally {
    window.setTimeout(poll, 500);
  }
}

poll();
