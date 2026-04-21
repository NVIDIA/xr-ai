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

// ─────────────────────────────────────────────────────────────────────────────

/**
 * Configures camera capture for a {@link StreamSession}.
 *
 * Resolution and frame-rate are intentionally not exposed here: the LiveKit
 * JS SDK and the browser negotiate the best supported format automatically,
 * matching the behaviour of the Swift SDK on iOS and visionOS.
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

  /**
   * @param {object}       [opts]
   * @param {boolean}      [opts.enabled=true]
   * @param {string}       [opts.facing=CameraFacing.FRONT]
   * @param {string|null}  [opts.deviceId=null]
   */
  constructor({ enabled = true, facing = CameraFacing.FRONT, deviceId = null } = {}) {
    this.#enabled  = enabled;
    this.#facing   = facing;
    this.#deviceId = deviceId;
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
