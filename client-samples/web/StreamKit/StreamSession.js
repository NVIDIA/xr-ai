/**
 * @fileoverview Primary public API for StreamKit sessions.
 *
 * `StreamSession` is the single entry-point for application code. It delegates
 * all network operations to a {@link StreamingBackend} implementation, keeping
 * the call-site free of transport-specific details.
 *
 * Mirror of the Swift `StreamSession` class. Because `BackendConfiguration
 * .makeBackend()` is async (dynamic import), construction is split into a
 * synchronous constructor (accepts a pre-created backend) and a static async
 * factory `StreamSession.create(backendConfig)` that resolves the backend and
 * then constructs the session — matching the Swift initialiser flow without
 * requiring an async constructor (which JavaScript does not support).
 *
 * @module StreamKit/StreamSession
 *
 * @example
 * import { StreamSession } from './StreamKit/StreamSession.js';
 * import { BackendConfiguration, LiveKitConfig } from './StreamKit/Config/BackendConfiguration.js';
 * import { SessionConfig } from './StreamKit/Config/SessionConfig.js';
 *
 * const session = await StreamSession.create(
 *   BackendConfiguration.liveKit(new LiveKitConfig({ host: '192.168.1.100', token: myJwt }))
 * );
 *
 * session.onConnectionStateChanged = (state) => console.log('state:', state);
 * session.onDataReceived = (data) => console.log('data:', data);
 *
 * await session.connect(SessionConfig.default);
 */

import { ConnectionState } from './ConnectionState.js';

// ─────────────────────────────────────────────────────────────────────────────

export class StreamSession {
  // ── Private fields ──────────────────────────────────────────────────────────

  /** @type {import('./Backends/StreamingBackend.js').StreamingBackend} */
  #backend;

  /** @type {string} Current connection state; one of {@link ConnectionState}. */
  #connectionState = ConnectionState.DISCONNECTED;

  // ── Public event hooks ──────────────────────────────────────────────────────

  /**
   * Called whenever the connection lifecycle state changes.
   *
   * The argument is one of the {@link ConnectionState} constant values.
   *
   * @type {((state: string) => void) | null}
   */
  onConnectionStateChanged = null;

  /**
   * Called whenever binary data is received from the remote end.
   *
   * @type {((data: Uint8Array) => void) | null}
   */
  onDataReceived = null;

  // ── Constructor ─────────────────────────────────────────────────────────────

  /**
   * Create a `StreamSession` wrapping an already-instantiated backend.
   *
   * Prefer the {@link StreamSession.create} static factory when starting from
   * a `BackendConfiguration`, as it handles the async dynamic import for you.
   *
   * @param {import('./Backends/StreamingBackend.js').StreamingBackend} backend
   *   A concrete backend instance implementing the `StreamingBackend` interface.
   */
  constructor(backend) {
    this.#backend = backend;
    this.#wireCallbacks();
  }

  // ── Static factory ──────────────────────────────────────────────────────────

  /**
   * Async factory — resolves the backend from a `BackendConfiguration` and
   * returns a fully wired `StreamSession`.
   *
   * This is the recommended way to create a session when you have a
   * `BackendConfiguration` rather than a pre-constructed backend instance.
   *
   * @param {import('./Config/BackendConfiguration.js').BackendConfiguration} backendConfig
   * @returns {Promise<StreamSession>}
   *
   * @example
   * const session = await StreamSession.create(
   *   BackendConfiguration.liveKit(new LiveKitConfig({ host: '192.168.1.100', token: jwt }))
   * );
   */
  static async create(backendConfig) {
    const backend = await backendConfig.makeBackend();
    return new StreamSession(backend);
  }

  // ── Public getters ──────────────────────────────────────────────────────────

  /**
   * The current connection lifecycle state.
   *
   * One of the {@link ConnectionState} enum values. Starts as
   * `ConnectionState.DISCONNECTED` and is updated automatically as the
   * underlying backend fires state-change events.
   *
   * @returns {string}
   */
  get connectionState() {
    return this.#connectionState;
  }

  // ── Public API ──────────────────────────────────────────────────────────────

  /**
   * Establishes a session using the provided configuration.
   *
   * Delegates to the backend's `connect()` method. The backend fires
   * `onConnectionStateChanged` as the handshake progresses; this session
   * forwards those events to the caller's own `onConnectionStateChanged`
   * callback.
   *
   * @param {import('./Config/SessionConfig.js').SessionConfig} config
   * @returns {Promise<void>}
   * @throws {import('./StreamError.js').StreamError}
   */
  async connect(config) {
    await this.#backend.connect(config);
  }

  /**
   * Cleanly disconnects from the session and releases all resources.
   *
   * Safe to call at any time, including before `connect()`.
   *
   * @returns {Promise<void>}
   */
  async disconnect() {
    await this.#backend.disconnect();
  }

  /**
   * Begins capturing the local camera and streaming it to remote participants.
   *
   * @returns {Promise<void>}
   * @throws {import('./StreamError.js').StreamError} `cameraRequiresConnection`
   *   if called while not connected.
   */
  async startCamera() {
    await this.#backend.startCamera();
  }

  /**
   * Stops camera capture and removes the published video track.
   *
   * Resolves silently if the camera was never started.
   *
   * @returns {Promise<void>}
   */
  async stopCamera() {
    await this.#backend.stopCamera();
  }

  /**
   * Sends binary (or string) data to remote participants.
   *
   * Forwarded directly to the backend's `send()`. Keep individual messages
   * under the transport MTU (15 KB for LiveKit's WebRTC data channel).
   *
   * @param {string | ArrayBuffer | Uint8Array} data
   * @param {object}  [options]
   * @param {boolean} [options.reliable=true]
   *   `true` for ordered, guaranteed delivery.
   *   `false` for low-latency best-effort delivery.
   * @returns {Promise<void>}
   * @throws {import('./StreamError.js').StreamError} `notConnected` if not connected.
   */
  async send(data, options) {
    await this.#backend.send(data, options);
  }

  // ── Private helpers ─────────────────────────────────────────────────────────

  /**
   * Subscribes to the backend's event hooks and forwards them to this
   * session's own public callbacks. Must be called once, immediately after
   * the backend is set.
   */
  #wireCallbacks() {
    this.#backend.onConnectionStateChanged = (state) => {
      this.#connectionState = state;
      this.onConnectionStateChanged?.(state);
    };

    this.#backend.onDataReceived = (data) => {
      this.onDataReceived?.(data);
    };
  }
}
