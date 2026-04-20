/**
 * @fileoverview Generic session configuration for a StreamKit connection.
 *
 * Mirror of the Swift `SessionConfig` struct. Network endpoint details
 * (host, port, token) are backend-specific and live in {@link LiveKitConfig}.
 * `SessionConfig` captures only the cross-backend concerns: participant
 * identity and media settings.
 *
 * @module StreamKit/Config/SessionConfig
 */

import { AudioConfig } from './AudioConfig.js';
import { CameraConfig } from './CameraConfig.js';

// ─────────────────────────────────────────────────────────────────────────────

/**
 * Configuration passed to `StreamSession.connect(config)`.
 *
 * @example
 * import { SessionConfig } from './StreamKit/Config/SessionConfig.js';
 *
 * // Use the default preset:
 * await session.connect(SessionConfig.default);
 *
 * // Or build a custom config:
 * import { AudioConfig } from './StreamKit/Config/AudioConfig.js';
 * import { CameraConfig } from './StreamKit/Config/CameraConfig.js';
 *
 * const config = new SessionConfig({
 *   audio: AudioConfig.raw,
 *   camera: CameraConfig.disabled,
 *   identity: 'hololens-pilot-1',
 * });
 */
export class SessionConfig {
  /** @type {AudioConfig} */
  #audio;

  /** @type {CameraConfig} */
  #camera;

  /** @type {string} */
  #identity;

  /**
   * @param {object}       [opts]
   * @param {AudioConfig}  [opts.audio]    - Defaults to {@link AudioConfig.default}.
   * @param {CameraConfig} [opts.camera]   - Defaults to {@link CameraConfig.default}.
   * @param {string}       [opts.identity] - Defaults to `"participant-<6-digit random>"`.
   */
  constructor({
    audio = AudioConfig.default,
    camera = CameraConfig.default,
    identity = `participant-${String(Math.floor(100_000 + Math.random() * 900_000))}`,
  } = {}) {
    this.#audio = audio;
    this.#camera = camera;
    this.#identity = identity;
  }

  /** @returns {AudioConfig} Microphone capture settings. */
  get audio() { return this.#audio; }
  set audio(v) { this.#audio = v; }

  /** @returns {CameraConfig} Camera capture settings. */
  get camera() { return this.#camera; }
  set camera(v) { this.#camera = v; }

  /**
   * A unique identity for this participant in the LiveKit room.
   * Must be non-empty and unique within the room.
   *
   * @returns {string}
   */
  get identity() { return this.#identity; }
  set identity(v) { this.#identity = v; }

  // -------------------------------------------------------------------------
  // Preset
  // -------------------------------------------------------------------------

  /**
   * Default session: software-processed audio, front camera, random identity.
   * Equivalent to Swift `SessionConfig.default`.
   *
   * @returns {SessionConfig}
   */
  static get default() {
    return new SessionConfig();
  }
}
