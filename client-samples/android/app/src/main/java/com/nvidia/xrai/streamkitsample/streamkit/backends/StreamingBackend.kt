// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.xrai.streamkitsample.streamkit.backends

import com.nvidia.xrai.streamkitsample.streamkit.ConnectionState
import com.nvidia.xrai.streamkitsample.streamkit.config.AudioConfig
import com.nvidia.xrai.streamkitsample.streamkit.config.CameraConfig
import com.nvidia.xrai.streamkitsample.streamkit.config.SessionConfig

/**
 * The contract that every networking backend must satisfy.
 *
 * [StreamSession][com.nvidia.xrai.streamkitsample.streamkit.StreamSession] delegates
 * all network operations to an implementation of this interface, keeping
 * call-sites free of transport-specific details.
 *
 * Mirror of Swift `StreamingBackend` protocol and web `StreamingBackend` interface.
 *
 * ## Implementing a custom backend
 *
 * ```kotlin
 * class MyBackend : StreamingBackend {
 *     override var onConnectionStateChanged: ((ConnectionState) -> Unit)? = null
 *     override var onDataReceived: ((topic: String, data: ByteArray) -> Unit)? = null
 *     override var onAgentStatus: ((status: String) -> Unit)? = null
 *
 *     override suspend fun connect(sessionConfig: SessionConfig) {
 *         // establish connection; call onConnectionStateChanged(CONNECTED) when done
 *     }
 *     override suspend fun disconnect() { /* release resources */ }
 *     override suspend fun startAudio(config: AudioConfig) { /* begin capture & publish */ }
 *     override suspend fun stopAudio() { /* stop & unpublish */ }
 *     override suspend fun startCamera(config: CameraConfig) { /* begin capture & publish */ }
 *     override suspend fun stopCamera() { /* stop & unpublish */ }
 *     override suspend fun send(data: ByteArray, reliable: Boolean) { /* publish data */ }
 * }
 * ```
 */
interface StreamingBackend {

    // ── Event hooks assigned by StreamSession before connect() ────────────────

    /** Fired when the connection lifecycle state changes. */
    var onConnectionStateChanged: ((ConnectionState) -> Unit)?

    /**
     * Fired when binary data arrives from remote participants.
     * `topic` identifies the logical channel (empty string when unset).
     */
    var onDataReceived: ((topic: String, data: ByteArray) -> Unit)?

    /**
     * Fired when an agent publishes a status update on the reserved
     * `_agent.status` channel. Never fires for that topic on [onDataReceived].
     *
     * Common values: `"idle"`, `"processing"`.
     */
    var onAgentStatus: ((status: String) -> Unit)?

    // ── Lifecycle ──────────────────────────────────────────────────────────────

    /**
     * Establishes a connection. Suspends until the connection is ready.
     *
     * Does **not** start audio or camera — call [startAudio] and [startCamera]
     * explicitly after [connect] returns.
     *
     * @throws [com.nvidia.xrai.streamkitsample.streamkit.StreamError]
     */
    suspend fun connect(sessionConfig: SessionConfig)

    /**
     * Cleanly disconnects and releases all resources (tracks, sockets, etc.).
     * Must complete even if [connect] was never called.
     */
    suspend fun disconnect()

    // ── Media ──────────────────────────────────────────────────────────────────

    /**
     * Begins microphone capture and publishes an audio track.
     * @throws [com.nvidia.xrai.streamkitsample.streamkit.StreamError.NotConnected]
     */
    suspend fun startAudio(config: AudioConfig)

    /** Stops microphone capture and unpublishes the audio track. */
    suspend fun stopAudio()

    /**
     * Begins camera capture and publishes a video track.
     * @throws [com.nvidia.xrai.streamkitsample.streamkit.StreamError.CameraRequiresConnection]
     */
    suspend fun startCamera(config: CameraConfig)

    /** Stops camera capture and unpublishes the video track. */
    suspend fun stopCamera()

    // ── Data channel ──────────────────────────────────────────────────────────

    /**
     * Sends binary data to remote participants via the transport's data channel.
     *
     * Keep individual messages under 15 KB (LiveKit's WebRTC data-channel MTU).
     *
     * @param data    Payload bytes.
     * @param reliable `true` for ordered, guaranteed delivery (default).
     *                 `false` for low-latency best-effort delivery.
     * @throws [com.nvidia.xrai.streamkitsample.streamkit.StreamError.NotConnected]
     */
    suspend fun send(data: ByteArray, reliable: Boolean = true)
}
