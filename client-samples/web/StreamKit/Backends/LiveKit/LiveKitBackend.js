/**
 * @fileoverview LiveKit WebRTC backend for StreamKit.
 *
 * Implements the {@link StreamingBackend} interface using the LiveKit JS SDK v2.
 * The SDK is resolved via an import map in the host page's `index.html`; no
 * build step is required.
 *
 * Mirror of the Swift `LiveKitBackend` class. Three deliberate platform
 * adaptations apply — see `.claudelearnings/decisions/web-streamkit-api-design.md`.
 *
 * @module StreamKit/Backends/LiveKit/LiveKitBackend
 */

import {
  Room,
  RoomEvent,
  Track,
  createLocalVideoTrack,
  createLocalAudioTrack,
} from 'livekit-client';

import { ConnectionState } from '../../ConnectionState.js';
import { StreamError } from '../../StreamError.js';
import { MicrophoneMode } from '../../Config/AudioConfig.js';

// ─────────────────────────────────────────────────────────────────────────────

/**
 * Maps a LiveKit `ConnectionState` string to the StreamKit {@link ConnectionState}
 * enum value.
 *
 * LiveKit v2 fires `RoomEvent.ConnectionStateChanged` with one of the literal
 * strings `'connected'`, `'connecting'`, `'disconnected'`, or `'reconnecting'`.
 * Those happen to match our own enum values one-for-one, but we map explicitly
 * here rather than relying on that coincidence, so a future SDK bump won't
 * silently produce wrong states.
 *
 * @param {string} lkState - Raw LiveKit ConnectionState string.
 * @returns {string} A {@link ConnectionState} value.
 */
function mapState(lkState) {
  switch (lkState) {
    case 'connected':     return ConnectionState.CONNECTED;
    case 'connecting':    return ConnectionState.CONNECTING;
    case 'reconnecting':  return ConnectionState.RECONNECTING;
    case 'disconnected':
    default:              return ConnectionState.DISCONNECTED;
  }
}

// ─────────────────────────────────────────────────────────────────────────────

/**
 * Implements {@link StreamingBackend} over LiveKit WebRTC.
 *
 * @implements {import('../StreamingBackend.js').StreamingBackend}
 *
 * @example
 * // Consumers should not construct this directly — use BackendConfiguration:
 * import { BackendConfiguration, LiveKitConfig } from '../../Config/BackendConfiguration.js';
 *
 * const session = await StreamSession.create(
 *   BackendConfiguration.liveKit(new LiveKitConfig({ host: '192.168.1.100', token: jwt }))
 * );
 */
export class LiveKitBackend {
  // ── Private fields ──────────────────────────────────────────────────────────

  /** @type {import('../../Config/BackendConfiguration.js').LiveKitConfig} */
  #config;

  /** @type {Room | null} */
  #room = null;

  /** @type {import('livekit-client').LocalVideoTrack | null} */
  #videoTrack = null;

  /** @type {import('livekit-client').LocalAudioTrack | null} */
  #audioTrack = null;

  /** @type {import('../../Config/SessionConfig.js').SessionConfig | null} */
  #sessionConfig = null;

  // ── Public event hooks ──────────────────────────────────────────────────────

  /**
   * Called whenever the connection lifecycle state changes.
   * `StreamSession` assigns this before calling `connect()`.
   *
   * @type {((state: string) => void) | null}
   */
  onConnectionStateChanged = null;

  /**
   * Called whenever binary data is received from remote participants.
   * `StreamSession` assigns this before calling `connect()`.
   *
   * @type {((data: Uint8Array) => void) | null}
   */
  onDataReceived = null;

  // ── Constructor ─────────────────────────────────────────────────────────────

  /**
   * @param {import('../../Config/BackendConfiguration.js').LiveKitConfig} config
   */
  constructor(config) {
    this.#config = config;
  }

  // ── Public API ──────────────────────────────────────────────────────────────

  /**
   * Establishes a LiveKit room connection.
   *
   * Tears down any existing session first, validates the host, builds the
   * WebSocket URL, acquires a JWT (from `config.token` or the token endpoint),
   * creates and wires up a `Room`, then connects. Audio is published
   * automatically unless the session config disables the microphone.
   *
   * @param {import('../../Config/SessionConfig.js').SessionConfig} sessionConfig
   * @returns {Promise<void>}
   * @throws {StreamError} `invalidHost` | `missingToken` | `tokenFetchFailed`
   */
  async connect(sessionConfig) {
    // Tear down any existing session cleanly before starting a new one.
    await this.#tearDown();

    this.#sessionConfig = sessionConfig;

    // ── Validate host & build WebSocket URL ───────────────────────────────────
    const { host, port, secure } = this.#config;
    if (!host || typeof host !== 'string' || host.trim() === '') {
      throw StreamError.invalidHost(host);
    }

    // Reject strings that already contain a scheme or look structurally wrong.
    let cleanHost;
    try {
      // Temporarily add a scheme so URL can parse the authority part.
      const probe = new URL(`ws://${host}`);
      cleanHost = probe.hostname;
      if (!cleanHost) throw new Error('empty hostname');
    } catch {
      throw StreamError.invalidHost(host);
    }

    const scheme = secure ? 'wss' : 'ws';
    const wsURL = `${scheme}://${cleanHost}:${port}`;

    // ── Resolve JWT ───────────────────────────────────────────────────────────
    let token;
    if (this.#config.token) {
      token = this.#config.token;
    } else if (this.#config.tokenURL) {
      token = await this.#fetchToken(this.#config.tokenURL, sessionConfig.identity);
    } else {
      throw StreamError.missingToken();
    }

    // ── Create Room and wire events ───────────────────────────────────────────
    const room = new Room();
    this.#room = room;

    room.on(RoomEvent.ConnectionStateChanged, (lkState) => {
      this.onConnectionStateChanged?.(mapState(lkState));
    });

    room.on(RoomEvent.DataReceived, (payload /*, participant, kind, topic */) => {
      // payload is already a Uint8Array per the LiveKit v2 API.
      this.onDataReceived?.(payload);
    });

    // ── Connect ───────────────────────────────────────────────────────────────
    await room.connect(wsURL, token);

    // ── Publish audio (unless disabled) ──────────────────────────────────────
    const audioMode = sessionConfig.audio?.mode;
    if (audioMode !== MicrophoneMode.DISABLED) {
      await this.#publishAudio(sessionConfig);
    }
  }

  /**
   * Disconnects from the room and releases all resources.
   *
   * Safe to call at any time, including before `connect()`.
   *
   * @returns {Promise<void>}
   */
  async disconnect() {
    await this.#tearDown();
  }

  /**
   * Captures the local camera and publishes it to the room.
   *
   * The `facingMode` is taken from `sessionConfig.camera.facing`, which is
   * already a browser `getUserMedia` constraint value (`'user'`/`'environment'`).
   *
   * @returns {Promise<void>}
   * @throws {StreamError} `cameraRequiresConnection` if not connected.
   */
  async startCamera() {
    if (!this.#room || this.#room.state !== 'connected') {
      throw StreamError.cameraRequiresConnection();
    }

    // Stop any previous video track first.
    await this.stopCamera();

    const facingMode = this.#sessionConfig?.camera?.facing ?? 'user';
    const track = await createLocalVideoTrack({ facingMode });
    this.#videoTrack = track;

    await this.#room.localParticipant.publishTrack(track, {
      source: Track.Source.Camera,
    });
  }

  /**
   * Unpublishes and stops the local video track.
   *
   * Resolves silently if the camera was never started.
   *
   * @returns {Promise<void>}
   */
  async stopCamera() {
    if (!this.#videoTrack) return;

    if (this.#room) {
      await this.#room.localParticipant.unpublishTrack(this.#videoTrack);
    }
    this.#videoTrack.stop();
    this.#videoTrack = null;
  }

  /**
   * Sends binary data to remote participants via the LiveKit data channel.
   *
   * Accepts a `string`, `ArrayBuffer`, or `Uint8Array`. Strings are encoded
   * to UTF-8. Keep individual messages under 15 KB (LiveKit's WebRTC MTU)
   * to avoid fragmentation.
   *
   * @param {string | ArrayBuffer | Uint8Array} data
   * @param {object}  [opts]
   * @param {boolean} [opts.reliable=true]
   * @returns {Promise<void>}
   * @throws {StreamError} `notConnected` if not connected.
   */
  async send(data, { reliable = true } = {}) {
    if (!this.#room || this.#room.state !== 'connected') {
      throw StreamError.notConnected();
    }

    let bytes;
    if (typeof data === 'string') {
      bytes = new TextEncoder().encode(data);
    } else if (data instanceof Uint8Array) {
      bytes = data;
    } else {
      // ArrayBuffer or any other BufferSource.
      bytes = new Uint8Array(data);
    }

    await this.#room.localParticipant.publishData(bytes, { reliable });
  }

  // ── Private helpers ─────────────────────────────────────────────────────────

  /**
   * Disconnects the room, stops all tracks, and nulls out all state.
   *
   * @returns {Promise<void>}
   */
  async #tearDown() {
    if (this.#room) {
      // Remove all listeners before disconnecting to prevent stale callbacks.
      this.#room.removeAllListeners();
      await this.#room.disconnect();
      this.#room = null;
    }

    if (this.#videoTrack) {
      this.#videoTrack.stop();
      this.#videoTrack = null;
    }

    if (this.#audioTrack) {
      this.#audioTrack.stop();
      this.#audioTrack = null;
    }

    this.#sessionConfig = null;
  }

  /**
   * Creates and publishes a local audio track based on the session's
   * `AudioConfig`. Called automatically from `connect()`.
   *
   * @param {import('../../Config/SessionConfig.js').SessionConfig} sessionConfig
   * @returns {Promise<void>}
   */
  async #publishAudio(sessionConfig) {
    const { mode, echoCancellation } = sessionConfig.audio;

    // Derive WebRTC constraint booleans from MicrophoneMode.
    // VOICE_PROCESSING: enable the full browser voice-isolation pipeline.
    // SOFTWARE_PROCESSING: standard WebRTC DSP — use echoCancellation setting.
    // RAW: disable all DSP so the server can handle processing.
    const isVoice = mode === MicrophoneMode.VOICE_PROCESSING;
    const isSoftware = mode === MicrophoneMode.SOFTWARE_PROCESSING;

    const track = await createLocalAudioTrack({
      echoCancellation: isVoice ? true : (isSoftware ? echoCancellation : false),
      noiseSuppression: isVoice || isSoftware,
      autoGainControl:  isVoice || isSoftware,
    });

    this.#audioTrack = track;

    await this.#room.localParticipant.publishTrack(track, {
      source: Track.Source.Microphone,
    });
  }

  /**
   * Fetches a LiveKit JWT from a token endpoint.
   *
   * The endpoint is called as:
   * `GET <tokenURL>?identity=<identity>`
   *
   * The server must return either a plain JWT string or a JSON object with a
   * `"token"` key:  `{ "token": "eyJ…" }`.
   *
   * @param {string} tokenURL  - Base URL of the token endpoint.
   * @param {string} identity  - Participant identity to embed in the token.
   * @returns {Promise<string>} Resolved JWT string.
   * @throws {StreamError} `tokenFetchFailed` on any network or parse error.
   */
  async #fetchToken(tokenURL, identity) {
    let response;
    try {
      const url = new URL(tokenURL);
      url.searchParams.set('identity', identity);
      response = await fetch(url.toString());
    } catch (err) {
      throw StreamError.tokenFetchFailed(tokenURL);
    }

    if (!response.ok) {
      throw StreamError.tokenFetchFailed(tokenURL);
    }

    const text = await response.text();

    // Try JSON first: { "token": "eyJ…" }
    try {
      const json = JSON.parse(text);
      if (json && typeof json.token === 'string' && json.token.length > 0) {
        return json.token;
      }
    } catch {
      // Not JSON — fall through to plain-string check.
    }

    // Plain JWT string.
    const trimmed = text.trim();
    if (trimmed.length > 0) {
      return trimmed;
    }

    throw StreamError.tokenFetchFailed(tokenURL);
  }
}
