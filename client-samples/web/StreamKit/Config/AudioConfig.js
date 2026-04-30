// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @fileoverview Microphone capture configuration for a StreamKit session.
 *
 * Mirror of the Swift `AudioConfig` struct and its `MicrophoneMode` enum.
 * Web DSP options differ from Apple's AVFoundation stack — `highpassFilter`
 * and `typingNoiseDetection` are replaced by the single `echoCancellation`
 * boolean that maps to the browser's built-in AEC pipeline.
 *
 * @module StreamKit/Config/AudioConfig
 */

/**
 * Frozen enumeration of microphone processing modes.
 *
 * Maps to Swift `AudioConfig.MicrophoneMode`. Values are kept as strings so
 * they survive JSON serialisation and can be used in `MediaTrackConstraints`.
 *
 * @readonly
 * @enum {string}
 */
export const MicrophoneMode = Object.freeze({
  /**
   * Apple AUVoiceIO equivalent.
   * On the web this selects the browser's voice-isolation / noise-suppression
   * pipeline (where supported), giving the closest experience to iOS's
   * native voice-processing I/O unit.
   */
  VOICE_PROCESSING: 'voiceProcessing',

  /**
   * WebRTC software DSP: echo cancellation, AGC and noise suppression via the
   * browser's built-in WebRTC stack. Mirrors Swift `.softwareProcessing`.
   */
  SOFTWARE_PROCESSING: 'softwareProcessing',

  /**
   * Raw PCM — all browser-side DSP is disabled. Choose this when the server
   * handles audio processing, or for non-voice audio sources.
   */
  RAW: 'raw',

  /** Microphone is not captured or published. */
  DISABLED: 'disabled',
});

// ─────────────────────────────────────────────────────────────────────────────

/**
 * Configures microphone capture for a {@link StreamSession}.
 *
 * ## Presets
 * ```js
 * AudioConfig.default  // SOFTWARE_PROCESSING + echo cancellation — safe default
 * AudioConfig.raw      // no DSP — let the server handle processing
 * AudioConfig.disabled // microphone off
 * ```
 *
 * @example
 * import { AudioConfig, MicrophoneMode } from './StreamKit/Config/AudioConfig.js';
 *
 * const custom = new AudioConfig({ mode: MicrophoneMode.RAW });
 */
export class AudioConfig {
  /** @type {string} One of the {@link MicrophoneMode} values. */
  #mode;

  /**
   * Whether the browser's built-in acoustic echo cancellation is enabled.
   *
   * Replaces Swift's `highpassFilter` + `typingNoiseDetection` pair — the web
   * platform exposes echo cancellation as its primary DSP knob.
   * Only meaningful for `SOFTWARE_PROCESSING` and `VOICE_PROCESSING` modes.
   *
   * @type {boolean}
   */
  #echoCancellation;

  /**
   * @param {object}  [opts]
   * @param {string}  [opts.mode=MicrophoneMode.SOFTWARE_PROCESSING]
   * @param {boolean} [opts.echoCancellation=false]
   */
  constructor({
    mode = MicrophoneMode.SOFTWARE_PROCESSING,
    echoCancellation = false,
  } = {}) {
    this.#mode = mode;
    this.#echoCancellation = echoCancellation;
  }

  /** @returns {string} */
  get mode() { return this.#mode; }
  set mode(v) { this.#mode = v; }

  /** @returns {boolean} */
  get echoCancellation() { return this.#echoCancellation; }
  set echoCancellation(v) { this.#echoCancellation = v; }

  // -------------------------------------------------------------------------
  // Presets
  // -------------------------------------------------------------------------

  /**
   * SOFTWARE_PROCESSING with echo cancellation enabled.
   * Good all-around default for voice calls.
   *
   * @returns {AudioConfig}
   */
  static get default() {
    return new AudioConfig({ mode: MicrophoneMode.SOFTWARE_PROCESSING, echoCancellation: true });
  }

  /**
   * Raw PCM capture — no browser DSP applied.
   *
   * @returns {AudioConfig}
   */
  static get raw() {
    return new AudioConfig({ mode: MicrophoneMode.RAW, echoCancellation: false });
  }

  /**
   * Microphone disabled — nothing is captured or published.
   *
   * @returns {AudioConfig}
   */
  static get disabled() {
    return new AudioConfig({ mode: MicrophoneMode.DISABLED, echoCancellation: false });
  }
}
