// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @fileoverview StreamingBackend interface contract.
 *
 * This is the single seam between `StreamSession` and any networking
 * technology. The library ships with a LiveKit implementation (`LiveKitBackend`);
 * to use a different transport (proprietary streaming SDK, custom WebRTC, etc.)
 * implement this interface and pass your instance to `StreamSession`.
 *
 * There is no runtime code in this file — JavaScript has no formal interfaces.
 * The contract is enforced by documentation and duck-typing.
 *
 * Mirror of the Swift `StreamingBackend` protocol.
 *
 * @module StreamKit/Backends/StreamingBackend
 */

// ─────────────────────────────────────────────────────────────────────────────
// Interface definition (JSDoc only)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @interface StreamingBackend
 *
 * The contract that every networking backend must satisfy.
 *
 * `StreamSession` delegates all network operations to an object implementing
 * this interface, so call-sites never depend on a specific transport.
 *
 * ## Implementing a custom backend
 *
 * ```js
 * import { ConnectionState } from '../ConnectionState.js';
 *
 * class MyBackend {
 *   // ── Event hooks ─────────────────────────────────────────────────────────
 *   // StreamSession assigns these before calling connect().
 *   // Fire them from any context; StreamSession re-dispatches as needed.
 *
 *   /** @type {((state: string) => void) | null} *\/
 *   onConnectionStateChanged = null;
 *
 *   /** @type {((data: ArrayBuffer | Uint8Array) => void) | null} *\/
 *   onDataReceived = null;
 *
 *   // ── Lifecycle ───────────────────────────────────────────────────────────
 *
 *   async connect(sessionConfig) {
 *     // establish connection using sessionConfig.identity, etc.
 *     this.onConnectionStateChanged?.(ConnectionState.CONNECTED);
 *   }
 *
 *   async disconnect() { /* release resources *\/ }
 *
 *   // ── Media ───────────────────────────────────────────────────────────────
 *
 *   async startCamera() { /* begin capture & publish *\/ }
 *   async stopCamera()  { /* stop capture & unpublish *\/ }
 *
 *   // ── Data channel ────────────────────────────────────────────────────────
 *
 *   async send(data, { reliable = true } = {}) {
 *     // publish data; reliable → ordered/guaranteed, !reliable → best-effort
 *   }
 * }
 *
 * // Then:
 * import { StreamSession } from '../StreamSession.js';
 * const session = new StreamSession(new MyBackend());
 * ```
 */

// ─────────────────────────────────────────────────────────────────────────────
// Property / method stubs (JSDoc typedef style)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Fired when the connection lifecycle state changes.
 *
 * `StreamSession` sets this property before calling `connect()`.
 * Fire it from any execution context; the session re-dispatches as needed.
 *
 * @name StreamingBackend#onConnectionStateChanged
 * @type {((state: import('../ConnectionState.js').ConnectionState) => void) | null}
 */

/**
 * Fired when binary data arrives from the remote end.
 *
 * `StreamSession` sets this property before calling `connect()`.
 *
 * @name StreamingBackend#onDataReceived
 * @type {((topic: string, data: Uint8Array) => void) | null}
 */

/**
 * Establish a connection using the provided session configuration.
 *
 * Network endpoint details (host, port, token) are supplied at construction
 * time via the backend-specific config object (e.g. `LiveKitConfig`).
 *
 * @function
 * @name StreamingBackend#connect
 * @param {import('../Config/SessionConfig.js').SessionConfig} sessionConfig
 *   Room/participant metadata and media settings.
 * @returns {Promise<void>}
 * @throws {import('../StreamError.js').StreamError}
 */

/**
 * Cleanly disconnect and release all resources (tracks, sockets, etc.).
 *
 * Must resolve even if the session was never connected.
 *
 * @function
 * @name StreamingBackend#disconnect
 * @returns {Promise<void>}
 */

/**
 * Begin capturing the local camera and streaming it to remote participants.
 *
 * Must throw `StreamError.cameraRequiresConnection()` if called while not
 * connected.
 *
 * @function
 * @name StreamingBackend#startCamera
 * @returns {Promise<void>}
 * @throws {import('../StreamError.js').StreamError}
 */

/**
 * Stop camera capture and remove the published video track.
 *
 * Must resolve silently if the camera was never started.
 *
 * @function
 * @name StreamingBackend#stopCamera
 * @returns {Promise<void>}
 * @throws {import('../StreamError.js').StreamError}
 */

/**
 * Send binary data to remote participants via the transport's data channel.
 *
 * Keep individual messages under the transport's MTU (15 KB for LiveKit's
 * WebRTC data channel) to avoid fragmentation.
 *
 * @function
 * @name StreamingBackend#send
 * @param {ArrayBuffer | Uint8Array} data - Payload bytes.
 * @param {object}  [opts]
 * @param {boolean} [opts.reliable=true]
 *   `true` for ordered, guaranteed delivery.
 *   `false` for low-latency best-effort delivery.
 * @returns {Promise<void>}
 * @throws {import('../StreamError.js').StreamError}
 */
