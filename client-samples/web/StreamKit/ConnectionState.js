// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @fileoverview Connection lifecycle state reported by {@link StreamSession}.
 *
 * Mirror of the Swift `ConnectionState` enum. The object is deeply frozen so
 * consumers can use strict equality (`===`) against these constants without
 * worrying about accidental mutation.
 *
 * @module StreamKit/ConnectionState
 */

/**
 * Frozen enumeration of connection lifecycle states.
 *
 * @example
 * import { ConnectionState } from './StreamKit/ConnectionState.js';
 *
 * session.onConnectionStateChanged = (state) => {
 *   if (state === ConnectionState.CONNECTED) {
 *     console.log('ready');
 *   }
 * };
 *
 * @readonly
 * @enum {string}
 */
export const ConnectionState = Object.freeze({
  /** No active connection; the initial state and the state after disconnect. */
  DISCONNECTED: 'disconnected',

  /** Handshake and authentication in progress. */
  CONNECTING: 'connecting',

  /** Fully connected and ready to send/receive. */
  CONNECTED: 'connected',

  /** Lost connection; the backend is attempting to re-establish it. */
  RECONNECTING: 'reconnecting',
});
