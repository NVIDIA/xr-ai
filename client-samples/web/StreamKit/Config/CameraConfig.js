// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @fileoverview Camera capture configuration for a StreamKit session.
 *
 * Mirror of the Swift `CameraConfig` struct and its `Position` enum.
 * `CameraFacing` values are set to the `facingMode` strings used by the
 * browser's `getUserMedia` / `MediaTrackConstraints` API so they can be
 * forwarded directly to the WebRTC layer without translation.
 *
 * @module StreamKit/Config/CameraConfig
 */

/**
 * Frozen enumeration of camera facing directions.
 *
 * Values match the `facingMode` constraint strings defined by the
 * Media Capture and Streams specification, mirroring Swift `CameraConfig.Position`.
 *
 * @readonly
 * @enum {string}
 */
export const CameraFacing = Object.freeze({
  /** Front-facing camera (`facingMode: 'user'`). Maps to Swift `.front`. */
  FRONT: 'user',

  /** Rear-facing camera (`facingMode: 'environment'`). Maps to Swift `.back`. */
  BACK: 'environment',
});

/** Encoder tradeoff under resource or bandwidth pressure; never a quality guarantee. */
export const VideoQualityPreference = Object.freeze({
  BALANCED: 'balanced',
  /** Favor resolution over frame rate. */
  DETAIL: 'detail',
  /** Favor frame rate over resolution. */
  MOTION: 'motion',
});

/**
 * Optional publishing policy. Limits apply to each encoded stream, not aggregate
 * network traffic. Capture format is unchanged. Backends apply supported settings.
 * The detail preset disables multiple-resolution publication and favors resolution
 * over FPS; it does not guarantee minimum resolution or frame delivery.
 * Defaults for an explicitly supplied policy are 3 Mbps and 30 FPS.
 * Restart the camera to apply changes; these are not live setters.
 */
export class CameraEncodingConfig {
  /**
   * @param {object} [opts]
   * @param {number} [opts.maxBitrateBps=3000000] Positive integer, bits per second.
   * @param {number} [opts.maxFramerate=30] Positive integer, frames per second.
   * @param {string|null} [opts.qualityPreference=null] Null preserves backend default.
   * @param {boolean|null} [opts.simulcast=null] Publish multiple resolutions when supported.
   */
  constructor({ maxBitrateBps = 3_000_000, maxFramerate = 30,
    qualityPreference = null, simulcast = null } = {}) {
    for (const [name, value] of Object.entries({ maxBitrateBps, maxFramerate })) {
      if (!Number.isSafeInteger(value) || value <= 0) {
        throw new RangeError(`${name} must be a positive integer`);
      }
    }
    if (qualityPreference !== null && !Object.values(VideoQualityPreference).includes(qualityPreference)) {
      throw new TypeError('Unknown video quality preference');
    }
    if (simulcast !== null && typeof simulcast !== 'boolean') {
      throw new TypeError('simulcast must be a boolean or null');
    }
    this.maxBitrateBps = maxBitrateBps;
    this.maxFramerate = maxFramerate;
    this.qualityPreference = qualityPreference;
    this.simulcast = simulcast;
    Object.freeze(this);
  }

  /** @returns {CameraEncodingConfig} Favor resolution and disable simulcast. */
  static get detail() {
    return new CameraEncodingConfig({ qualityPreference: VideoQualityPreference.DETAIL, simulcast: false });
  }
  /** @returns {CameraEncodingConfig} Favor frame rate; leave simulcast at backend default. */
  static get motion() {
    return new CameraEncodingConfig({ qualityPreference: VideoQualityPreference.MOTION });
  }
  /** @returns {CameraEncodingConfig} Allow both resolution and frame rate to adapt. */
  static get balanced() {
    return new CameraEncodingConfig({ qualityPreference: VideoQualityPreference.BALANCED });
  }
}

// ─────────────────────────────────────────────────────────────────────────────

/**
 * Configures camera capture for a {@link StreamSession}.
 *
 * Capture resolution and frame-rate are intentionally not exposed here: the browser
 * selects a supported native format without forcing an aspect ratio, matching
 * the behaviour of the Swift SDK on iOS and visionOS.
 *
 * ## Presets
 * ```js
 * CameraConfig.default  // enabled, front-facing
 * CameraConfig.disabled // camera off
 * CameraConfig.rear     // enabled, rear-facing
 * ```
 *
 * @example
 * import { CameraConfig, CameraFacing } from './StreamKit/Config/CameraConfig.js';
 *
 * const custom = new CameraConfig({ enabled: true, facing: CameraFacing.BACK });
 */
export class CameraConfig {
  /** @type {boolean} */
  #enabled;

  /** @type {string} One of the {@link CameraFacing} values. */
  #facing;

  /**
   * Specific device ID from `navigator.mediaDevices.enumerateDevices()`.
   * When set, takes precedence over `facing`.
   *
   * @type {string | null}
   */
  #deviceId;

  /** @type {CameraEncodingConfig|null} Optional publish-side policy. */
  #encoding;

  /**
   * @param {object}       [opts]
   * @param {boolean}      [opts.enabled=true]
   * @param {string}       [opts.facing=CameraFacing.FRONT]
   * @param {string|null}  [opts.deviceId=null]
   * @param {CameraEncodingConfig|null} [opts.encoding=null] Null preserves backend defaults.
   */
  constructor({ enabled = true, facing = CameraFacing.FRONT, deviceId = null, encoding = null } = {}) {
    this.#enabled  = enabled;
    this.#facing   = facing;
    this.#deviceId = deviceId;
    this.encoding = encoding;
  }

  /** @returns {boolean} Whether the camera should be captured and streamed. */
  get enabled() { return this.#enabled; }
  set enabled(v) { this.#enabled = v; }

  /**
   * Which camera to use (`'user'` or `'environment'`).
   * This value can be passed directly to `getUserMedia` as `facingMode`.
   *
   * @returns {string}
   */
  get facing() { return this.#facing; }
  set facing(v) { this.#facing = v; }

  /**
   * Specific device ID. When set, overrides `facing`.
   *
   * @returns {string | null}
   */
  get deviceId() { return this.#deviceId; }
  set deviceId(v) { this.#deviceId = v; }

  /** @returns {CameraEncodingConfig|null} Publish-side policy for the next camera start. */
  get encoding() { return this.#encoding; }
  set encoding(v) { this.#encoding = v == null ? null : new CameraEncodingConfig(v); }

  // -------------------------------------------------------------------------
  // Presets
  // -------------------------------------------------------------------------

  /**
   * Camera enabled, front-facing. Equivalent to Swift `CameraConfig.default`.
   *
   * @returns {CameraConfig}
   */
  static get default() {
    return new CameraConfig({ enabled: true, facing: CameraFacing.FRONT });
  }

  /**
   * Camera disabled — nothing is captured or published.
   *
   * @returns {CameraConfig}
   */
  static get disabled() {
    return new CameraConfig({ enabled: false, facing: CameraFacing.FRONT });
  }

  /**
   * Camera enabled, rear-facing. Equivalent to Swift `CameraConfig.rear`.
   *
   * @returns {CameraConfig}
   */
  static get rear() {
    return new CameraConfig({ enabled: true, facing: CameraFacing.BACK });
  }
}
