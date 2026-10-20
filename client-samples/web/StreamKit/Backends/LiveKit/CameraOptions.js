// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { VideoPresets, VideoPresets43 } from 'livekit-client';
import { VideoQualityPreference } from '../../Config/CameraConfig.js';

/** @internal SDK-only publication mapping; not part of StreamKit's exported API. */
export function cameraPublishOptions(encoding, dimensions, defaults) {
  if (!encoding) return {};
  const options = {
    videoEncoding: { maxBitrate: encoding.maxBitrateBps, maxFramerate: encoding.maxFramerate },
    ...(encoding.simulcast !== null && { simulcast: encoding.simulcast }),
    ...(encoding.qualityPreference !== null && {
      degradationPreference: {
        [VideoQualityPreference.DETAIL]: 'maintain-resolution',
        [VideoQualityPreference.MOTION]: 'maintain-framerate',
        [VideoQualityPreference.BALANCED]: 'balanced',
      }[encoding.qualityPreference],
    }),
  };
  if (!(encoding.simulcast ?? defaults.simulcast)) return options;
  const { width, height } = dimensions;
  if (!(width > 0 && height > 0)) throw new Error('Camera dimensions are not resolved');
  const aspect = Math.max(width, height) / Math.min(width, height);
  const presets = Math.abs(aspect - 16 / 9) < Math.abs(aspect - 4 / 3) ? VideoPresets : VideoPresets43;
  const layers = defaults.videoSimulcastLayers ?? [presets.h180, presets.h360];
  const budgets = layers.map(layer => layer.encoding.maxBitrate);
  const largest = Math.max(0, ...budgets);
  const reduced = largest > encoding.maxBitrateBps;
  const scaled = reduced ? budgets.map(budget => {
    if (!Number.isSafeInteger(budget) || budget <= 0 || !Number.isSafeInteger(encoding.maxBitrateBps)) {
      throw new RangeError('Simulcast bitrate limits must be representable positive integers');
    }
    return Number(BigInt(budget) * BigInt(encoding.maxBitrateBps) / BigInt(largest));
  }) : budgets;
  if (reduced && (scaled.some(budget => budget === 0) || budgets.some((budget, i) =>
    budgets.some((other, j) => budget !== other && scaled[i] === scaled[j])))) {
    throw new RangeError('Primary bitrate cannot preserve the SDK lower-layer budget ordering');
  }
  options.videoSimulcastLayers = layers.map((layer, i) => ({
    ...layer,
    encoding: {
      ...layer.encoding,
      maxBitrate: scaled[i],
    },
  }));
  return options;
}
