// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Opt-in capture telemetry and a local warning when return audio cannot arrive.
 * The hub enables this per connection. It never opens a camera or changes capture
 * state. Camera intent comes from the user's app settings, not missing frames.
 */
export const CAPTURE_QUALITY_CONTROL = 'procedure.capture.quality.control';
export const CAPTURE_QUALITY_TELEMETRY = 'procedure.capture.quality.telemetry';

export class CaptureQualityClient {
  constructor({ send, read, warn, clock = () => performance.now(), schedule = setTimeout, cancel = clearTimeout }) {
    Object.assign(this, { send, read, warn, clock, schedule, cancel });
    this.token = null;
    this.retiredTokens = new Set();
    this.sequence = 0;
    this.controlSequence = -1;
    this.active = false;
    this.closed = false;
    this.sending = false;
    this.lastControl = -Infinity;
    this.lastMetrics = -Infinity;
    this.timer = null;
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
    this.lastControl = this.clock();
    this.active = data.active;
    if (!this.active) this.cancelWarning();
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
        version: 1, token: this.token, sequence: ++this.sequence,
        mode: state.cameraMode, streaming: state.cameraMode === 'live' && state.isCameraActive,
        profile: JSON.stringify([state.cameraMode, state.selectedCameraId ?? '', state.cameraTrackId ?? '']).slice(0, 256),
        track_id: video?.trackId ?? '', quality: metrics?.quality ?? 'unknown',
        rtt_ms: metrics?.roundTripTimeMs ?? null, receive_jitter_ms: metrics?.receiveJitterMs ?? null,
        sent_fps: video?.framesPerSecond ?? null, sent_width: video?.width ?? null, sent_height: video?.height ?? null,
        limitation: video?.limitation ?? null, uplink_loss_fraction: video?.uplinkLossFraction ?? null,
        uplink_rtt_ms: video?.uplinkRttMs ?? null,
      }), { topic: CAPTURE_QUALITY_TELEMETRY, reliable: false });
    } catch { /* A lost data channel is handled by local connection feedback. */ }
    finally { this.sending = false; }
  }

  connection(state) {
    if (state === 'connected') {
      this.cancelWarning();
      return;
    }
    if (this.closed || !this.active || this.timer !== null || this.clock() - this.lastControl > 15000) return;
    // Brief reconnects should not interrupt a demonstration. The audio is
    // already decoded locally; it needs neither TTS nor a working network.
    this.timer = this.schedule(() => {
      this.timer = null;
      if (!this.closed && this.active && this.read().connectionState !== 'connected') {
        this.active = false; // At most one outage warning until the hub confirms capture again.
        this.warn();
      }
    }, 2000);
  }

  cancelWarning() {
    if (this.timer !== null) this.cancel(this.timer);
    this.timer = null;
  }

  close() {
    this.closed = true;
    this.active = false;
    this.cancelWarning();
  }
}

/** Unlock in the Connect gesture, before asynchronous token/room setup. */
export function prepareCaptureOutageAudio(showError) {
  const text = 'Connection lost. Pause your demonstration and check the connection.';
  const fallback = { play: () => showError(text), close() {} };
  const AudioContext = globalThis.AudioContext ?? globalThis.webkitAudioContext;
  if (!AudioContext) return fallback;
  let context;
  try { context = new AudioContext(); } catch { return fallback; }
  void context.resume().catch(() => {});
  let closed = false;
  let buffer = null;
  const controller = new AbortController();
  void fetch(new URL('./assets/capture-connection-lost.wav', import.meta.url), { signal: controller.signal })
    .then(response => { if (!response.ok) throw new Error('Alert audio unavailable'); return response.arrayBuffer(); })
    .then(bytes => context.decodeAudioData(bytes))
    .then(audio => { if (!closed) buffer = audio; }).catch(() => {});
  return {
    play() {
      if (closed) return;
      showError(text);
      // Never wait for a download during an outage. Only an already decoded
      // clip is eligible; the text warning remains immediate if preloading failed.
      if (!buffer || context.state !== 'running') return;
      const source = context.createBufferSource();
      source.buffer = buffer;
      source.connect(context.destination);
      source.start();
    },
    close() {
      closed = true;
      controller.abort();
      void context.close().catch(() => {});
    },
  };
}
