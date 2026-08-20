// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { register } from 'node:module';
import test from 'node:test';

register('./web_camera_test_loader.mjs', import.meta.url);

const { LiveKitBackend } = await import(
  '../../client-samples/web/StreamKit/Backends/LiveKit/LiveKitBackend.js'
);

function makeMediaTrack(settings = {}) {
  return {
    stopCount: 0,
    getSettings: () => ({ width: 1280, height: 720, facingMode: 'environment', ...settings }),
    stop() {
      this.stopCount += 1;
    },
  };
}

function makePublishedTrack(mediaTrack) {
  return {
    mediaStreamTrack: mediaTrack,
    stopCount: 0,
    stop() {
      this.stopCount += 1;
      mediaTrack.stop();
    },
  };
}

function makeRoom(publishTrack, { localTracks = [], remoteTracks = [], quality = 'unknown' } = {}) {
  const handlers = new Map();
  const room = {
    state: 'disconnected',
    remoteParticipants: new Map(remoteTracks.length ? [[
      'remote',
      { trackPublications: new Map(remoteTracks.map((track, i) => [String(i), { track }])) },
    ]] : []),
    unpublishedTracks: [],
    localParticipant: {
      connectionQuality: quality,
      trackPublications: new Map(localTracks.map((track, i) => [String(i), { track }])),
      publishTrack,
      publishData: async () => {},
      unpublishTrack: async track => {
        globalThis.__livekitRoom.unpublishedTracks.push(track);
      },
    },
    on(event, handler) {
      handlers.set(event, handler);
      return this;
    },
    emit(event, ...args) {
      handlers.get(event)?.(...args);
    },
    removeAllListeners() {
      handlers.clear();
    },
    async connect() {
      this.state = 'connected';
    },
    async disconnect() {
      this.state = 'disconnected';
    },
  };
  return room;
}

async function connectedBackend(t, publishTrack, options = {}) {
  const room = makeRoom(publishTrack, options);
  globalThis.__livekitRoom = room;
  const backend = new LiveKitBackend({
    host: 'localhost',
    port: 7880,
    secure: false,
    token: 'test-token',
    tokenURL: null,
    hubIdentity: null,
  });
  t.after(() => backend.disconnect());
  options.configure?.(backend);
  await backend.connect({ identity: 'browser-test' });
  return { backend, room };
}

function installMediaDevices(getUserMedia) {
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    value: { mediaDevices: { getUserMedia } },
  });
}

class FakeClassList {
  values = new Set();

  add(value) {
    this.values.add(value);
  }

  remove(value) {
    this.values.delete(value);
  }
}

function installAppBrowser(model) {
  const previewCard = { style: {} };
  const video = {
    classList: new FakeClassList(),
    closest: () => previewCard,
    onresize: null,
    srcObject: null,
    style: {},
    videoHeight: 0,
    videoWidth: 0,
  };
  const element = () => ({
    classList: new FakeClassList(),
    style: {},
    textContent: '',
  });
  globalThis.__elements = new Map([
    ['camera-preview', video],
    ['preview-placeholder', element()],
    ['preview-live-badge', element()],
    ['agent-response-text', element()],
    ['error-toast', element()],
  ]);
  globalThis.__appBaseModel = model;
  globalThis.window = {
    addEventListener: () => {},
  };
  globalThis.MediaStream = class {
    constructor(tracks) {
      this.tracks = tracks;
    }

    getVideoTracks() {
      return this.tracks;
    }
  };
  return { previewCard, video };
}

test('publishes and previews the captured full-frame camera track', async (t) => {
  const capturedTracks = [makeMediaTrack(), makeMediaTrack()];
  let currentTrack = capturedTracks[0];
  const captureConstraints = [];
  const publishedTracks = [];
  const publishedWrappers = [];
  installMediaDevices(async constraints => {
    captureConstraints.push(constraints);
    return { getVideoTracks: () => [currentTrack] };
  });

  const { backend, room } = await connectedBackend(t, async mediaTrack => {
    publishedTracks.push(mediaTrack);
    const wrapper = makePublishedTrack(mediaTrack);
    publishedWrappers.push(wrapper);
    return { videoTrack: wrapper };
  });
  const model = {
    agentResponse: null,
    isCameraActive: false,
    selectedCameraId: 'camera-7',
    session: backend,
  };
  const { previewCard, video } = installAppBrowser(model);
  await import(`../../client-samples/web/App/app.js?test=${Date.now()}`);

  await globalThis.__appActions.startCamera();

  assert.deepEqual(captureConstraints[0], {
    audio: false,
    video: { deviceId: { exact: 'camera-7' }, resizeMode: 'none' },
  });
  assert.equal(publishedTracks[0], capturedTracks[0]);
  assert.equal(backend.cameraTrack, capturedTracks[0]);
  assert.equal(video.srcObject.getVideoTracks()[0], capturedTracks[0]);
  assert.equal(previewCard.style.aspectRatio, '1280 / 720');

  await globalThis.__appActions.stopCamera();

  assert.equal(capturedTracks[0].stopCount, 1);
  assert.equal(publishedWrappers[0].stopCount, 1);
  assert.equal(room.unpublishedTracks[0], publishedWrappers[0]);
  assert.equal(video.srcObject, null);
  assert.equal(video.onresize, null);
  assert.equal(previewCard.style.aspectRatio, '');

  currentTrack = capturedTracks[1];
  await globalThis.__appActions.startCamera();
  assert.equal(video.srcObject.getVideoTracks()[0], capturedTracks[1]);

  await globalThis.__appActions.disconnect();

  assert.equal(capturedTracks[1].stopCount, 1);
  assert.equal(publishedWrappers[1].stopCount, 1);
  assert.equal(video.srcObject, null);
  assert.equal(video.onresize, null);
  assert.equal(previewCard.style.aspectRatio, '');
});

test('stops the captured camera track when LiveKit publication fails', async (t) => {
  const mediaTrack = makeMediaTrack();
  installMediaDevices(async () => ({ getVideoTracks: () => [mediaTrack] }));
  const failure = new Error('publication failed');
  const { backend } = await connectedBackend(t, async () => {
    throw failure;
  });

  await assert.rejects(backend.startCamera({ facing: 'environment' }), failure);

  assert.equal(mediaTrack.stopCount, 1);
  assert.equal(backend.cameraTrack, null);
});

test('reports the transport-selected ICE pair and all inbound-track jitter', async (t) => {
  const report = new Map([
    ['transport', {
      id: 'transport', type: 'transport', selectedCandidatePairId: 'active-pair',
    }],
    ['stale-pair', {
      id: 'stale-pair', type: 'candidate-pair', state: 'succeeded',
      currentRoundTripTime: 0.9,
    }],
    ['active-pair', {
      id: 'active-pair', type: 'candidate-pair', nominated: false,
      currentRoundTripTime: 0.042,
    }],
    ['inbound-audio', { id: 'inbound-audio', type: 'inbound-rtp', jitter: 0.003 }],
    ['inbound-video', { id: 'inbound-video', type: 'inbound-rtp', jitter: 0.007 }],
  ]);
  const track = { getRTCStatsReport: async () => report };
  let resolveMetrics;
  const metricsPromise = new Promise(resolve => { resolveMetrics = resolve; });

  await connectedBackend(t, async () => {}, {
    quality: 'good',
    remoteTracks: [track],
    configure: backend => { backend.onNetworkMetrics = resolveMetrics; },
  });

  const metrics = await metricsPromise;
  assert.equal(metrics.quality, 'good');
  assert.equal(metrics.roundTripTimeMs, 42);
  assert.equal(metrics.receiveJitterMs, 7);
});

test('maps network quality and rejects invalid RTC measurements', async (t) => {
  const invalidReport = new Map([
    ['pair', {
      id: 'pair', type: 'candidate-pair', nominated: true,
      currentRoundTripTime: Number.POSITIVE_INFINITY,
    }],
    ['inbound', { id: 'inbound', type: 'inbound-rtp', jitter: -1 }],
  ]);
  const track = { getRTCStatsReport: async () => invalidReport };

  for (const [quality, expected] of [
    ['excellent', 'excellent'],
    ['good', 'good'],
    ['poor', 'poor'],
    ['lost', 'lost'],
    ['unknown', 'unknown'],
    ['unexpected', 'unknown'],
  ]) {
    let resolveMetrics;
    const metricsPromise = new Promise(resolve => { resolveMetrics = resolve; });
    await connectedBackend(t, async () => {}, {
      quality,
      localTracks: [track],
      configure: backend => { backend.onNetworkMetrics = resolveMetrics; },
    });
    const metrics = await metricsPromise;
    assert.equal(metrics.quality, expected);
    assert.equal(metrics.roundTripTimeMs, null);
    assert.equal(metrics.receiveJitterMs, null);
  }
});

test('stops network polling after a terminal room disconnect', async (t) => {
  let statsCalls = 0;
  const track = {
    async getRTCStatsReport() {
      statsCalls += 1;
      return new Map();
    },
  };
  const { room } = await connectedBackend(t, async () => {}, { localTracks: [track] });
  await new Promise(resolve => setTimeout(resolve, 0));
  assert.equal(statsCalls, 1);

  room.state = 'disconnected';
  room.emit('connectionStateChanged', 'disconnected');
  await new Promise(resolve => setTimeout(resolve, 1_100));

  assert.equal(statsCalls, 1);
});
