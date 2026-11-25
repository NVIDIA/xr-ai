// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Request the camera's largest supported still photograph without resizing its
 * streaming track. Requires the browser ImageCapture photo API. Devices choose
 * the actual resolution and may briefly pause video during exposure.
 * @param {MediaStreamTrack} track Active camera track.
 * @param {{signal?: AbortSignal}} options Cancellation of delivery (native
 * exposure itself cannot be aborted).
 * @returns {Promise<{data: Uint8Array, mimeType: string, name: string}>}
 */
export async function capturePhoto(track, { signal } = {}) {
  signal?.throwIfAborted();
  if (!track || track.readyState !== 'live') throw new Error('An active camera track is required');
  if (typeof globalThis.ImageCapture !== 'function') {
    throw new Error('This browser does not support native still photography');
  }
  const camera = new ImageCapture(track);
  const capabilities = await camera.getPhotoCapabilities();
  signal?.throwIfAborted();
  const settings = {};
  if (capabilities.imageWidth?.max > 0) settings.imageWidth = capabilities.imageWidth.max;
  if (capabilities.imageHeight?.max > 0) settings.imageHeight = capabilities.imageHeight.max;
  const blob = await camera.takePhoto(settings);
  const data = new Uint8Array(await blob.arrayBuffer());
  signal?.throwIfAborted();
  return { data, mimeType: blob.type, name: 'photo' };
}
