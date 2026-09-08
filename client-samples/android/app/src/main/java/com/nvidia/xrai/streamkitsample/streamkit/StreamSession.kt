// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.xrai.streamkitsample.streamkit

import android.content.Context
import android.util.Log
import com.nvidia.xrai.streamkitsample.streamkit.backends.StreamingBackend
import com.nvidia.xrai.streamkitsample.streamkit.backends.livekit.LiveKitBackend
import com.nvidia.xrai.streamkitsample.streamkit.config.AudioConfig
import com.nvidia.xrai.streamkitsample.streamkit.config.BackendConfiguration
import com.nvidia.xrai.streamkitsample.streamkit.config.CameraConfig
import com.nvidia.xrai.streamkitsample.streamkit.config.SessionConfig
import io.livekit.android.renderer.TextureViewRenderer
import io.livekit.android.room.track.LocalVideoTrack
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import org.json.JSONObject
import java.nio.ByteBuffer

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

    private val captureScope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private val captureJobs = mutableMapOf<String, Job>()

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

    /** Called about once per second with LiveKit-native network telemetry. */
    var onNetworkMetrics: ((metrics: NetworkMetrics) -> Unit)? = null

    /** Opt-in handler invoked when the remote agent asks this client for a still image. */
    var onImageCaptureRequested: (suspend (ImageCaptureRequest) -> CapturedImage)? = null

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
        captureJobs.values.forEach { it.cancel() }
        captureJobs.clear()
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

    /**
     * Pushes a single externally-sourced I420 video frame to the published
     * video track. The track is created lazily on the first call and reused
     * for subsequent frames; [stopCamera] tears it down.
     *
     * Used by clients that inject frames from their own pipeline (external
     * camera adapters, screen capture, synthetic frame sources). Mirror of
     * iOS `StreamSession.injectVideoFrame(_: CMSampleBuffer)`.
     *
     * @param i420         Read-only buffer containing Y, U, V planes back-to-back.
     * @param width        Y-plane pixel width (must be even).
     * @param height       Y-plane pixel height (must be even).
     * @param timestampUs  Source-side presentation timestamp, microseconds.
     *                     Note: not honored downstream — the WebRTC frame
     *                     timestamp is overridden internally with the local
     *                     monotonic clock to keep the encoder's PTS stable.
     * @throws [StreamError.NotConnected]
     */
    suspend fun injectVideoFrame(
        i420: ByteBuffer,
        width: Int,
        height: Int,
        timestampUs: Long,
    ) {
        backend.injectVideoFrame(i420, width, height, timestampUs)
    }

    // ── Local preview ─────────────────────────────────────────────────────────

    /**
     * The currently published local camera track, if any.
     * Used by `CameraPreviewView`; app code typically does not access this
     * directly.  Returns null when the camera is stopped or the active
     * backend is not LiveKit-based.
     */
    val localCameraTrack: LocalVideoTrack?
        get() = (backend as? LiveKitBackend)?.localCameraTrack

    /**
     * Initialises a [TextureViewRenderer] with the active room's EGL context.
     * Required once per renderer instance before frames can be drawn.
     * No-op when the active backend is not LiveKit-based.
     */
    fun initVideoRenderer(view: TextureViewRenderer) {
        (backend as? LiveKitBackend)?.initVideoRenderer(view)
    }

    /** Capture the next frame from the active local camera as a JPEG still. */
    suspend fun captureCurrentCameraImage(): CapturedImage {
        val track = localCameraTrack ?: throw IllegalStateException("Camera is not active.")
        return CapturedImage(track.captureJpeg())
    }

    /** Capture a local still without publishing video when the camera is off. */
    suspend fun captureImage(config: CameraConfig = CameraConfig.DEFAULT): CapturedImage =
        backend.captureImage(config)

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
        backend.onDataReceived = { topic, data ->
            when (topic) {
                "camera.capture.request" -> handleCaptureRequest(data)
                "camera.capture.cancel" -> handleCaptureCancel(data)
                else -> onDataReceived?.invoke(topic, data)
            }
        }
        backend.onAgentStatus = { status -> onAgentStatus?.invoke(status) }
        backend.onNetworkMetrics = { metrics -> onNetworkMetrics?.invoke(metrics) }
    }

    private fun handleCaptureRequest(data: ByteArray) {
        val request = runCatching { JSONObject(String(data, Charsets.UTF_8)) }.getOrNull() ?: return
        if (request.optInt("version") != 1) return
        val requestId = request.optString("request_id")
        if (requestId.isBlank()) return
        val handler = onImageCaptureRequested ?: return
        captureJobs.remove(requestId)?.cancel()
        captureJobs[requestId] = captureScope.launch {
            try {
                val image = handler(
                    ImageCaptureRequest(requestId, request.optLong("timeout_ms"))
                )
                require(image.data.isNotEmpty()) { "image capture returned no data" }
                require(image.mimeType in setOf("image/jpeg", "image/png", "image/webp")) {
                    "image capture returned an unsupported media type"
                }
                backend.sendImage(image.data, requestId, image.mimeType, image.name)
            } catch (_: CancellationException) {
                // Cancellation is the normal outcome for a superseded request.
            } catch (error: Exception) {
                Log.w("StreamSession", "Image capture request failed", error)
            } finally {
                captureJobs.remove(requestId)
            }
        }
    }

    private fun handleCaptureCancel(data: ByteArray) {
        val requestId = runCatching {
            JSONObject(String(data, Charsets.UTF_8)).optString("request_id")
        }.getOrNull() ?: return
        captureJobs.remove(requestId)?.cancel()
    }
}
