// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import { CaptureQualityClient, CAPTURE_QUALITY_TELEMETRY } from '../../client-samples/web/App/CaptureQuality.js';

function fixture() {
  const sent = [];
  const state = { connectionState: 'connected', cameraMode: 'live', isCameraActive: true };
  const client = new CaptureQualityClient({
    send: async (text, options) => sent.push([JSON.parse(text), options]), read: () => state,
  });
  return {
    client, state, sent,
    enable: (data = {}) => client.enable(new TextEncoder().encode(JSON.stringify({
      version: 1, token: 'a'.repeat(32), ...data,
    }))),
  };
}

test('telemetry waits for opt-in and follows the user camera mode', async () => {
  const f = fixture();
  await f.client.report({ quality: 'poor' });
  assert.equal(f.sent.length, 0);
  assert.equal(f.enable({ token: 'invalid' }), false);
  assert.equal(f.enable(), true);
  await f.client.report({ quality: 'poor' });
  assert.deepEqual(f.sent[0], [{
    version: 2, token: 'a'.repeat(32), sequence: 1, mode: 'live', streaming: true, quality: 'poor',
  }, { topic: CAPTURE_QUALITY_TELEMETRY, reliable: false }]);
  f.state.cameraMode = 'on-demand';
  f.state.isCameraActive = false;
  await f.client.report();
  assert.equal(f.sent.at(-1)[0].mode, 'on-demand');
  assert.equal(f.sent.at(-1)[0].streaming, false);
  assert.equal(f.sent.at(-1)[0].quality, 'unknown');
  assert.equal(f.sent.at(-1)[0].sequence, 2);
});

test('a replacement worker gets a new token and sequence', async () => {
  const f = fixture();
  f.enable();
  await f.client.report();
  f.enable({ token: 'b'.repeat(32) });
  await f.client.report();
  assert.equal(f.sent.at(-1)[0].token, 'b'.repeat(32));
  assert.equal(f.sent.at(-1)[0].sequence, 1);
});

test('report skips unavailable connections and recovers after transport errors', async () => {
  const f = fixture();
  f.state.connectionState = 'reconnecting';
  f.enable();
  await f.client.report();
  assert.equal(f.sent.length, 0);
  f.state.connectionState = 'connected';
  const send = f.client.send;
  f.client.send = async () => { throw new Error('network lost'); };
  await f.client.report();
  f.client.send = send;
  await f.client.report({ quality: 'good' });
  assert.equal(f.sent.length, 1);
  assert.equal(f.sent[0][0].quality, 'good');
  assert.equal(f.state.isCameraActive, true);
});
