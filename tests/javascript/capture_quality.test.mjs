// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import { CaptureQualityClient, CAPTURE_QUALITY_CONTROL, prepareCaptureOutageAudio } from '../../client-samples/web/App/CaptureQuality.js';
import { cameraNetworkMetrics } from '../../client-samples/web/StreamKit/NetworkMetrics.js';

function fixture() {
  let now = 0;
  const timers = new Map();
  const sent = [];
  let warnings = 0;
  const state = { connectionState: 'connected', cameraMode: 'live', isCameraActive: true, cameraTrackId: 'device-track' };
  const client = new CaptureQualityClient({
    send: async (text, options) => sent.push([JSON.parse(text), options]), read: () => state,
    warn: () => { warnings++; }, clock: () => now,
    schedule: (callback, delay) => {
      const id = Symbol(); timers.set(id, { callback, due: now + delay }); return id;
    }, cancel: id => timers.delete(id),
  });
  return {
    client, state, sent, timers, get warnings() { return warnings; },
    advance: ms => { now += ms; },
    fire: () => {
      for (const [id, { callback, due }] of timers) {
        if (due <= now) { timers.delete(id); callback(); }
      }
    },
    control: (data = {}) => client.control(CAPTURE_QUALITY_CONTROL, new TextEncoder().encode(JSON.stringify({
      version: 1, token: 'a'.repeat(32), sequence: 1, active: true, ...data,
    }))),
  };
}

test('telemetry is opt-in and follows user camera intent without enabling a camera', async () => {
  const f = fixture();
  f.client.network({ quality: 'poor' });
  assert.equal(f.sent.length, 0);
  f.control();
  await Promise.resolve();
  f.state.cameraMode = 'on-demand';
  f.state.isCameraActive = false;
  await f.client.report();
  assert.equal(f.sent.at(-1)[0].mode, 'on-demand');
  assert.equal(f.sent.at(-1)[0].streaming, false);
  assert.equal(f.sent.at(-1)[0].quality, 'poor');
  assert.equal(f.sent.at(-1)[1].reliable, false);
  assert.equal(f.sent.at(-1)[0].sequence, 2);
  f.advance(5000);
  await f.client.report();
  assert.equal(f.sent.at(-1)[0].quality, 'unknown');
});

test('unexpected disconnect plays one local alert only for a recently active capture', () => {
  const f = fixture();
  f.control();
  f.state.connectionState = 'reconnecting';
  f.client.connection('reconnecting');
  f.advance(1999);
  f.fire();
  assert.equal(f.warnings, 0);
  f.advance(1);
  f.fire();
  assert.equal(f.warnings, 1);
  f.client.connection('disconnected');
  f.advance(10000);
  f.fire();
  assert.equal(f.warnings, 1);
});

test('brief reconnect, intentional disconnect, stopped capture, and expired lease stay quiet', () => {
  for (const action of ['recover', 'close', 'stop', 'expire']) {
    const f = fixture();
    f.control();
    if (action === 'expire') f.advance(16000);
    f.state.connectionState = 'reconnecting';
    f.client.connection('reconnecting');
    if (action === 'recover') { f.state.connectionState = 'connected'; f.client.connection('connected'); }
    if (action === 'close') f.client.close();
    if (action === 'stop') f.control({ sequence: 2, active: false });
    f.advance(2000);
    f.fire();
    assert.equal(f.warnings, 0, action);
  }
});

test('stale stop/control cannot override a newer capture or replacement worker', () => {
  const f = fixture();
  f.control({ sequence: 2 });
  f.control({ sequence: 1, active: false });
  assert.equal(f.client.active, true);
  f.control({ token: 'b'.repeat(32), sequence: 1 });
  f.control({ token: 'a'.repeat(32), sequence: 3, active: false });
  assert.equal(f.client.token, 'b'.repeat(32));
  assert.equal(f.client.active, true);
  f.control({ token: 'b'.repeat(32), sequence: 2, active: false });
  assert.equal(f.client.active, false);
});

test('sender telemetry selects a video layer and its associated upload report', () => {
  const data = [
    { id: 'low', type: 'outbound-rtp', kind: 'video', frameWidth: 320, frameHeight: 180, framesPerSecond: 15 },
    { id: 'high', type: 'outbound-rtp', kind: 'video', remoteId: 'remote-high', frameWidth: 1280,
      frameHeight: 720, framesPerSecond: 30, qualityLimitationReason: 'bandwidth' },
    { id: 'remote-high', type: 'remote-inbound-rtp', fractionLost: 0.12, roundTripTime: 0.05 },
    { id: 'downlink', type: 'inbound-rtp', kind: 'audio', jitter: 0.3 },
  ];
  const metrics = cameraNetworkMetrics(new Map(data.map(s => [s.id, s])), 'TR_camera');
  assert.deepEqual(metrics, { trackId: 'TR_camera', width: 1280, height: 720, framesPerSecond: 30,
    limitation: 'bandwidth', uplinkLossFraction: 0.12, uplinkRttMs: 50 });
  assert.equal(cameraNetworkMetrics(null, 'TR_camera').framesPerSecond, null);
});

test('invalid controls do not arm local audio', () => {
  const f = fixture();
  f.client.control(CAPTURE_QUALITY_CONTROL, new TextEncoder().encode('not json'));
  f.control({ active: 'true' });
  f.control({ token: 'short' });
  f.state.connectionState = 'disconnected';
  f.client.connection('disconnected');
  f.advance(2000);
  f.fire();
  assert.equal(f.warnings, 0);
});

test('repeated short reconnects stay quiet and a confirmed new capture can warn again', () => {
  const f = fixture();
  f.control();
  for (let i = 0; i < 4; i++) {
    f.state.connectionState = 'reconnecting';
    f.client.connection('reconnecting');
    f.advance(1500);
    f.fire();
    assert.equal(f.warnings, 0);
    f.state.connectionState = 'connected';
    f.client.connection('connected');
    f.advance(500);
    f.fire();
  }
  f.state.connectionState = 'disconnected';
  f.client.connection('disconnected');
  f.advance(2000);
  f.fire();
  assert.equal(f.warnings, 1);
  f.state.connectionState = 'connected';
  f.client.connection('connected');
  f.state.connectionState = 'disconnected';
  f.client.connection('disconnected');
  f.advance(2000);
  f.fire();
  assert.equal(f.warnings, 1, 'reconnect alone must not claim a finalized capture restarted');
  f.control({ sequence: 2, active: true });
  f.client.connection('disconnected');
  f.advance(2000);
  f.fire();
  assert.equal(f.warnings, 2);
});


test('outage feedback never waits for an unfinished audio download', async (t) => {
  const oldAudio = globalThis.AudioContext;
  const oldFetch = globalThis.fetch;
  let playCount = 0;
  let fetchCount = 0;
  let signal;
  globalThis.AudioContext = class {
    state = 'running';
    resume = async () => {};
    close = async () => {};
    decodeAudioData = async () => ({});
    createBufferSource() { return { connect() {}, start() { playCount++; } }; }
  };
  globalThis.fetch = (_url, options) => { fetchCount++; signal = options.signal; return new Promise(() => {}); };
  t.after(() => { globalThis.AudioContext = oldAudio; globalThis.fetch = oldFetch; });
  const messages = [];
  const audio = prepareCaptureOutageAudio(text => messages.push(text));
  audio.play();
  assert.equal(messages.length, 1);
  assert.equal(playCount, 0);
  assert.equal(fetchCount, 1);
  audio.close();
  assert.equal(signal.aborted, true);
  audio.play();
  assert.equal(messages.length, 1);
});

test('a preloaded outage clip plays without another network or model request', async (t) => {
  const oldAudio = globalThis.AudioContext;
  const oldFetch = globalThis.fetch;
  let playCount = 0;
  let fetchCount = 0;
  globalThis.AudioContext = class {
    state = 'running';
    resume = async () => {};
    close = async () => {};
    decodeAudioData = async () => ({});
    createBufferSource() { return { connect() {}, start() { playCount++; } }; }
  };
  globalThis.fetch = async () => { fetchCount++; return { ok: true, arrayBuffer: async () => new ArrayBuffer(4) }; };
  t.after(() => { globalThis.AudioContext = oldAudio; globalThis.fetch = oldFetch; });
  const audio = prepareCaptureOutageAudio(() => {});
  await new Promise(resolve => setImmediate(resolve));
  audio.play();
  assert.equal(playCount, 1);
  assert.equal(fetchCount, 1);
  audio.close();
});
