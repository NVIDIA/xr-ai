// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import { CaptureQualityClient, CAPTURE_QUALITY_CONTROL } from '../../client-samples/web/App/CaptureQuality.js';
import { cameraNetworkMetrics } from '../../client-samples/web/StreamKit/NetworkMetrics.js';

function fixture() {
  let now = 0;
  const sent = [];
  const state = { connectionState: 'connected', cameraMode: 'live', isCameraActive: true, cameraTrackId: 'device-track' };
  const client = new CaptureQualityClient({
    send: async (text, options) => sent.push([JSON.parse(text), options]), read: () => state,
    clock: () => now,
  });
  return {
    client, state, sent,
    advance: ms => { now += ms; },
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

test('closed client stops reporting and transport errors never affect camera state', async () => {
  const f = fixture();
  f.control();
  await Promise.resolve();
  f.client.send = async () => { throw new Error('network lost'); };
  await f.client.report();
  assert.equal(f.state.isCameraActive, true);
  f.client.close();
  const count = f.sent.length;
  f.client.network({ quality: 'poor' });
  await f.client.report();
  assert.equal(f.sent.length, count);
});

test('telemetry does not send while the connection is unavailable', async () => {
  const f = fixture();
  f.state.connectionState = 'reconnecting';
  f.control();
  assert.equal(f.sent.length, 0);
  f.state.connectionState = 'connected';
  await f.client.report();
  assert.equal(f.sent.length, 1);
});
