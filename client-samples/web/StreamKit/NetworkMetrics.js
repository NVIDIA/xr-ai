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
 * corresponding RTCStats sample. Optional `video` contains camera-sender
 * dimensions/FPS, encoder limitation, and the associated upload receiver report;
 * missing fields remain null rather than being interpreted as zero.
 */
export class NetworkMetrics {
  constructor({
    quality = NetworkQuality.UNKNOWN,
    roundTripTimeMs = null,
    receiveJitterMs = null,
    video = null,
  } = {}) {
    this.quality = quality;
    this.roundTripTimeMs = finiteOrNull(roundTripTimeMs);
    this.receiveJitterMs = finiteOrNull(receiveJitterMs);
    this.video = video;
    Object.freeze(this);
  }
}

function finiteOrNull(value) {
  return Number.isFinite(value) && value >= 0 ? value : null;
}

/** Camera-sender statistics only; remote audio jitter is not upload loss. */
export function cameraNetworkMetrics(report, trackId) {
  const stats = report ? [...report.values()] : [];
  const streams = stats.filter(s => s.type === 'outbound-rtp' && (s.kind ?? s.mediaType) === 'video');
  // Simulcast layers are alternatives, not additive camera dimensions or FPS.
  const video = streams.sort((a, b) => (b.frameWidth ?? 0) * (b.frameHeight ?? 0)
    - (a.frameWidth ?? 0) * (a.frameHeight ?? 0))[0];
  const remote = video && stats.find(s => s.id === video.remoteId && s.type === 'remote-inbound-rtp');
  return Object.freeze({
    trackId: trackId ?? '',
    framesPerSecond: finiteOrNull(video?.framesPerSecond),
    width: Number.isInteger(video?.frameWidth) && video.frameWidth > 0 ? video.frameWidth : null,
    height: Number.isInteger(video?.frameHeight) && video.frameHeight > 0 ? video.frameHeight : null,
    limitation: ['none', 'bandwidth', 'cpu', 'other'].includes(video?.qualityLimitationReason)
      ? video.qualityLimitationReason : null,
    uplinkLossFraction: Number.isFinite(remote?.fractionLost) && remote.fractionLost >= 0 && remote.fractionLost <= 1
      ? remote.fractionLost : null,
    uplinkRttMs: finiteOrNull(remote?.roundTripTime) === null ? null : remote.roundTripTime * 1000,
  });
}
