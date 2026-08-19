// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Transport-neutral connection quality reported by StreamKit. */
export const NetworkQuality = Object.freeze({
  UNKNOWN:   'unknown',
  EXCELLENT: 'excellent',
  GOOD:      'good',
  POOR:      'poor',
  LOST:      'lost',
});

/**
 * A once-per-second snapshot of LiveKit's native WebRTC telemetry.
 * RTT and jitter are milliseconds; either is `null` until LiveKit has a
 * corresponding RTCStats sample.
 */
export class NetworkMetrics {
  constructor({
    quality = NetworkQuality.UNKNOWN,
    roundTripTimeMs = null,
    receiveJitterMs = null,
  } = {}) {
    this.quality = quality;
    this.roundTripTimeMs = finiteOrNull(roundTripTimeMs);
    this.receiveJitterMs = finiteOrNull(receiveJitterMs);
    Object.freeze(this);
  }
}

function finiteOrNull(value) {
  return Number.isFinite(value) && value >= 0 ? value : null;
}
