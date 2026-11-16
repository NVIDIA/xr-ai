// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Opt-in network and video telemetry for capture setup and spoken media alerts.
 * The hub enables this per connection. It never opens a camera or changes capture
 * state. Camera intent comes from the user's app settings, not missing frames.
 */
export const CAPTURE_QUALITY_CONTROL = 'procedure.capture.quality.control';
export const CAPTURE_QUALITY_TELEMETRY = 'procedure.capture.quality.telemetry';

export class CaptureQualityClient {
  constructor({ send, read, clock = () => performance.now() }) {
    Object.assign(this, { send, read, clock });
    this.token = null;
    this.retiredTokens = new Set();
    this.sequence = 0;
    this.controlSequence = -1;
    this.active = false;
    this.closed = false;
    this.sending = false;
    this.lastMetrics = -Infinity;
    this.metrics = null;
  }

  control(topic, bytes) {
    if (topic !== CAPTURE_QUALITY_CONTROL) return false;
    if (this.closed || bytes.length > 4096) return true;
    let data;
    try { data = JSON.parse(new TextDecoder().decode(bytes)); } catch { return true; }
    if (data.version !== 1 || typeof data.token !== 'string' || !/^[a-f0-9]{32}$/.test(data.token)
        || !Number.isSafeInteger(data.sequence) || typeof data.active !== 'boolean') return true;
    // The trusted hub may replace the worker while the client stays connected.
    // Retire old leases so delayed cleanup cannot disable a newer capture.
    if (this.retiredTokens.has(data.token)) return true;
    if (this.token !== null && data.token !== this.token) {
      this.retiredTokens.add(this.token);
      if (this.retiredTokens.size > 32) this.retiredTokens.delete(this.retiredTokens.values().next().value);
      this.controlSequence = -1;
    }
    if (data.sequence <= this.controlSequence) return true;
    this.token = data.token;
    this.controlSequence = data.sequence;
    this.active = data.active;
    void this.report();
    return true;
  }

  network(metrics) {
    this.metrics = metrics;
    this.lastMetrics = this.clock();
    void this.report();
  }

  async report() {
    if (!this.token || this.closed || this.sending || this.read().connectionState !== 'connected') return;
    const state = this.read();
    const fresh = this.clock() - this.lastMetrics < 4000;
    const metrics = fresh ? this.metrics : null;
    const video = metrics?.video;
    this.sending = true;
    try {
      await this.send(JSON.stringify({
        version: 2, token: this.token, sequence: ++this.sequence,
        mode: state.cameraMode, streaming: state.cameraMode === 'live' && state.isCameraActive,
        profile: JSON.stringify([state.cameraMode, state.selectedCameraId ?? '', state.cameraTrackId ?? '']).slice(0, 256),
        track_id: video?.trackId ?? '', quality: metrics?.quality ?? 'unknown',
        rtt_ms: metrics?.roundTripTimeMs ?? null, receive_jitter_ms: metrics?.receiveJitterMs ?? null,
        sent_fps: video?.framesPerSecond ?? null, sent_width: video?.width ?? null, sent_height: video?.height ?? null,
        limitation: video?.limitation ?? null, uplink_loss_fraction: video?.uplinkLossFraction ?? null,
        uplink_rtt_ms: video?.uplinkRttMs ?? null,
      }), { topic: CAPTURE_QUALITY_TELEMETRY, reliable: false });
    } catch { /* Telemetry is best effort; it must never interrupt camera capture. */ }
    finally { this.sending = false; }
  }

  close() {
    this.closed = true;
    this.active = false;
  }
}
