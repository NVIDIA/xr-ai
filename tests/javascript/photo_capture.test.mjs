// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import { capturePhoto } from '../../client-samples/web/StreamKit/PhotoCapture.js';

test('native photo uses maximum photo dimensions and leaves video settings alone', async () => {
  const track = { readyState: 'live', applyConstraints: () => assert.fail('video was reconfigured') };
  let photoSettings;
  globalThis.ImageCapture = class {
    constructor(actual) { assert.equal(actual, track); }
    async getPhotoCapabilities() { return { imageWidth: { max: 4000 }, imageHeight: { max: 3000 } }; }
    async takePhoto(settings) { photoSettings = settings; return new Blob(['photo'], { type: 'image/jpeg' }); }
  };
  try {
    const result = await capturePhoto(track);
    assert.deepEqual(photoSettings, { imageWidth: 4000, imageHeight: 3000 });
    assert.equal(result.mimeType, 'image/jpeg');
    assert.equal(new TextDecoder().decode(result.data), 'photo');
  } finally { delete globalThis.ImageCapture; }
});

test('unsupported native photo API fails explicitly', async () => {
  await assert.rejects(capturePhoto({ readyState: 'live' }), /does not support/);
});

test('cancellation during native exposure prevents delivering the photo', async () => {
  const controller = new AbortController();
  globalThis.ImageCapture = class {
    async getPhotoCapabilities() { return {}; }
    async takePhoto() { controller.abort(); return new Blob(['late'], { type: 'image/jpeg' }); }
  };
  try {
    await assert.rejects(capturePhoto({ readyState: 'live' }, { signal: controller.signal }), { name: 'AbortError' });
  } finally { delete globalThis.ImageCapture; }
});
