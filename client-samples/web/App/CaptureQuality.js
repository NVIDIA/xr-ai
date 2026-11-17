// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const CAPTURE_QUALITY_CONTROL = 'procedure.capture.quality.control';
export const CAPTURE_QUALITY_TELEMETRY = 'procedure.capture.quality.telemetry';

/** Send live connection grades and camera intent after the worker opts in. */
export class CaptureQualityClient {
  constructor({ send, read }) {
    this.send = send;
    this.read = read;
    this.token = null;
    this.sequence = 0;
    this.sending = false;
  }

  enable(bytes) {
    if (bytes.length > 4096) return false;
    let data;
    try { data = JSON.parse(new TextDecoder().decode(bytes)); } catch { return false; }
    if (data?.version !== 1 || typeof data.token !== 'string' || !/^[a-f0-9]{32}$/.test(data.token)) return false;
    if (this.token !== data.token) {
      this.token = data.token;
      this.sequence = 0;
    }
    return true;
  }

  async report(metrics) {
    if (!this.token || this.sending) return;
    const state = this.read();
    if (state.connectionState !== 'connected') return;
    this.sending = true;
    try {
      await this.send(JSON.stringify({
        version: 2, token: this.token, sequence: ++this.sequence,
        mode: state.cameraMode, streaming: state.cameraMode === 'live' && state.isCameraActive,
        quality: metrics?.quality ?? 'unknown',
      }), { topic: CAPTURE_QUALITY_TELEMETRY, reliable: false });
    } catch { /* Telemetry is best effort and must never interrupt capture. */ }
    finally { this.sending = false; }
  }
}
