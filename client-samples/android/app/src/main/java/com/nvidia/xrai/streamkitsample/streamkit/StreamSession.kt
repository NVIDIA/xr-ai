// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.xrai.streamkitsample.streamkit

import android.content.Context
import com.nvidia.xrai.streamkitsample.streamkit.backends.StreamingBackend
import com.nvidia.xrai.streamkitsample.streamkit.config.AudioConfig
import com.nvidia.xrai.streamkitsample.streamkit.config.BackendConfiguration
import com.nvidia.xrai.streamkitsample.streamkit.config.CameraConfig
import com.nvidia.xrai.streamkitsample.streamkit.config.SessionConfig

/**
 * Transport-agnostic streaming session — the single public entry-point of StreamKit.
 *
 * [StreamSession] wraps any [StreamingBackend] with a clean, coroutine-friendly API.
 * All operations are suspend functions; call them from a ViewModel coroutine scope.
 *
 * Mirror of Swift `StreamSession` and web `StreamSession`. The Android version
 * requires a [Context] to instantiate the LiveKit backend (used as applicationContext
 * internally — no memory leak).
 *
 * ## Lifecycle
 *
 * ```kotlin
 * // 1. Connect — WebRTC peer connection + data channel only
 * session.connect(SessionConfig(identity = "android-1"))
 *
 * // 2. Start media explicitly once connected
 * session.startAudio(AudioConfig.DEFAULT)
 * session.startCamera(CameraConfig.DEFAULT)
 *
 * // 3. Send / receive data
 * session.onDataReceived = { topic, data -> … }
 * session.send("hello".toByteArray())
 *
 * // 4. Stop media / disconnect
 * session.stopAudio()
 * session.stopCamera()
 * session.disconnect()
 * ```
 */
class StreamSession(private val backend: StreamingBackend) {

    /**
     * Creates a session backed by a [BackendConfiguration].
     * The [context] is used to instantiate the backend (stored as applicationContext).
     */
    constructor(backendConfig: BackendConfiguration, context: Context) : this(
        when (backendConfig) {
            is BackendConfiguration.LiveKit -> backendConfig.makeBackend(context)
        }
    )

    // ── Callbacks (wire before calling connect) ────────────────────────────────

    /**
     * Called when the connection lifecycle state changes.
     * Common pattern: update a ViewModel StateFlow from here.
     */
    var onConnectionStateChanged: ((ConnectionState) -> Unit)? = null

    /**
     * Called when data is received from remote participants.
     * `topic` identifies the logical channel; `data` is the raw payload.
     */
    var onDataReceived: ((topic: String, data: ByteArray) -> Unit)? = null

    /**
     * Called when an agent publishes a status update.
     * Common values: `"idle"`, `"processing"`.
     */
    var onAgentStatus: ((status: String) -> Unit)? = null

    init {
        wireCallbacks()
    }

    // ── Connection ─────────────────────────────────────────────────────────────

    /**
     * Establishes a WebRTC peer connection and data channel.
     * Does **not** start audio or camera — call [startAudio] and [startCamera]
     * explicitly once connected.
     *
     * @throws [StreamError]
     */
    suspend fun connect(config: SessionConfig = SessionConfig.DEFAULT) {
        backend.connect(config)
    }

    /**
     * Disconnects and releases all resources.
     * Safe to call at any time, including before [connect].
     */
    suspend fun disconnect() {
        backend.disconnect()
    }

    // ── Audio ──────────────────────────────────────────────────────────────────

    /**
     * Starts microphone capture and publishes an audio track.
     * @throws [StreamError.NotConnected]
     */
    suspend fun startAudio(config: AudioConfig = AudioConfig.DEFAULT) {
        backend.startAudio(config)
    }

    /**
     * Stops microphone capture and unpublishes the audio track.
     */
    suspend fun stopAudio() {
        backend.stopAudio()
    }

    // ── Camera ─────────────────────────────────────────────────────────────────

    /**
     * Starts camera capture and publishes a video track.
     * @throws [StreamError.CameraRequiresConnection]
     */
    suspend fun startCamera(config: CameraConfig = CameraConfig.DEFAULT) {
        backend.startCamera(config)
    }

    /**
     * Stops camera capture and unpublishes the video track.
     */
    suspend fun stopCamera() {
        backend.stopCamera()
    }

    // ── Data channel ──────────────────────────────────────────────────────────

    /**
     * Sends binary data to remote participants.
     *
     * @param data     Payload. Keep individual messages ≤ 15 KB on most transports.
     * @param reliable Ordered + guaranteed delivery when `true` (default).
     * @throws [StreamError.NotConnected]
     */
    suspend fun send(data: ByteArray, reliable: Boolean = true) {
        backend.send(data, reliable)
    }

    // ── Private ────────────────────────────────────────────────────────────────

    private fun wireCallbacks() {
        backend.onConnectionStateChanged = { state -> onConnectionStateChanged?.invoke(state) }
        backend.onDataReceived = { topic, data -> onDataReceived?.invoke(topic, data) }
        backend.onAgentStatus = { status -> onAgentStatus?.invoke(status) }
    }
}
