// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.xrai.streamkitsample.streamkit.backends.livekit

import android.content.Context
import com.nvidia.xrai.streamkitsample.streamkit.ConnectionState
import com.nvidia.xrai.streamkitsample.streamkit.StreamError
import com.nvidia.xrai.streamkitsample.streamkit.backends.StreamingBackend
import com.nvidia.xrai.streamkitsample.streamkit.config.AudioConfig
import com.nvidia.xrai.streamkitsample.streamkit.config.BackendConfiguration
import com.nvidia.xrai.streamkitsample.streamkit.config.CameraConfig
import com.nvidia.xrai.streamkitsample.streamkit.config.LiveKitConfig
import com.nvidia.xrai.streamkitsample.streamkit.config.SessionConfig
import io.livekit.android.LiveKit
import io.livekit.android.events.RoomEvent
import io.livekit.android.events.collect
import io.livekit.android.renderer.TextureViewRenderer
import io.livekit.android.room.Room
import io.livekit.android.room.track.CameraPosition
import io.livekit.android.room.track.DataPublishReliability
import io.livekit.android.room.track.LocalVideoTrack
import io.livekit.android.room.track.LocalVideoTrackOptions
import io.livekit.android.room.track.Track
import io.livekit.android.room.track.VideoCaptureParameter
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.net.URL
import java.net.URLEncoder
import java.nio.ByteBuffer

/**
 * [StreamingBackend] implementation using the LiveKit Android SDK.
 *
 * Consumers should not construct this directly — create it via
 * [BackendConfiguration.LiveKit] passed to [StreamSession][com.nvidia.xrai.streamkitsample.streamkit.StreamSession].
 *
 * Mirror of the Swift `LiveKitBackend` and web `LiveKitBackend`.
 *
 * ## Remote audio
 * LiveKit Android renders remote audio tracks automatically once subscribed —
 * no explicit attachment step is required (unlike the web SDK).
 *
 * ## Connection-state model
 * LiveKit Android `Room.connect()` is a suspend function that returns when the
 * room is CONNECTED (or throws). We emit CONNECTING before calling it and
 * CONNECTED after it returns. RECONNECTING / DISCONNECTED come through
 * `room.events` asynchronously.
 */
internal class LiveKitBackend(
    private val config: LiveKitConfig,
    private val appContext: Context,
) : StreamingBackend {

    // ── Event hooks ────────────────────────────────────────────────────────────

    override var onConnectionStateChanged: ((ConnectionState) -> Unit)? = null
    override var onDataReceived: ((topic: String, data: ByteArray) -> Unit)? = null
    override var onAgentStatus: ((status: String) -> Unit)? = null

    // ── Public local preview accessor ─────────────────────────────────────────

    /**
     * Currently published local camera track, or null when stopped.  Used by
     * `CameraPreviewView` to render the outgoing stream locally; app code
     * goes through that composable rather than touching this directly.
     */
    val localCameraTrack: LocalVideoTrack?
        get() = room?.localParticipant
            ?.getTrackPublication(Track.Source.CAMERA)?.track as? LocalVideoTrack

    /** Initialises a [TextureViewRenderer] with the connected room's EGL
     *  context so it can sink frames from the local camera track. No-op when
     *  the room is not connected. */
    fun initVideoRenderer(view: TextureViewRenderer) {
        room?.initVideoRenderer(view)
    }

    // ── Private state ──────────────────────────────────────────────────────────

    private var room: Room? = null
    private var isConnected = false

    /** Coroutine scope active for the lifetime of one connection. */
    private var connectionScope: CoroutineScope? = null

    // ── Injected video state (external frame source) ───────────────────────────

    /** Custom capturer that lets us push externally-sourced frames into LiveKit. */
    @Volatile private var injectedCapturer: InjectedVideoCapturer? = null
    private var injectedVideoTrack: LocalVideoTrack? = null
    private val injectedVideoMutex = Mutex()

    // ── StreamingBackend: connect / disconnect ─────────────────────────────────

    override suspend fun connect(sessionConfig: SessionConfig) {
        tearDown()

        if (config.host.isBlank()) throw StreamError.InvalidHost(config.host)

        val wsUrl = "wss://${config.host}:${config.port}"

        val token = when {
            !config.token.isNullOrBlank() -> config.token
            !config.tokenURL.isNullOrBlank() -> fetchToken(config.tokenURL, sessionConfig.identity)
            else -> throw StreamError.MissingToken
        }

        val newRoom = LiveKit.create(appContext)
        room = newRoom

        val scope = CoroutineScope(Dispatchers.Main + SupervisorJob())
        connectionScope = scope

        // Collect all room events for state changes and data.
        // Must be launched before room.connect() so we don't miss RECONNECTING /
        // DISCONNECTED events that arrive immediately after a connect.
        scope.launch {
            newRoom.events.collect { event ->
                handleEvent(event)
            }
        }

        // Emit CONNECTING before the async handshake.
        onConnectionStateChanged?.invoke(ConnectionState.CONNECTING)

        // room.connect() suspends until the room is fully connected or throws.
        newRoom.connect(wsUrl, token)

        // Successfully connected.
        isConnected = true
        onConnectionStateChanged?.invoke(ConnectionState.CONNECTED)
    }

    override suspend fun disconnect() {
        tearDown()
    }

    // ── StreamingBackend: audio ────────────────────────────────────────────────

    override suspend fun startAudio(config: AudioConfig) {
        if (!isConnected) throw StreamError.NotConnected
        val enabled = config.mode != AudioConfig.MicrophoneMode.DISABLED
        room?.localParticipant?.setMicrophoneEnabled(enabled)
    }

    override suspend fun stopAudio() {
        room?.localParticipant?.setMicrophoneEnabled(false)
    }

    // ── StreamingBackend: camera ───────────────────────────────────────────────

    override suspend fun startCamera(config: CameraConfig) {
        if (!isConnected) throw StreamError.CameraRequiresConnection

        val lp = room?.localParticipant ?: return
        val position = when (config.facing) {
            CameraConfig.CameraFacing.FRONT -> CameraPosition.FRONT
            CameraConfig.CameraFacing.BACK  -> CameraPosition.BACK
        }
        // setCameraEnabled() reads the current videoTrackCaptureDefaults when it
        // creates the track. Apply both deviceId (if pinned) and position there
        // before flipping it on. Per LocalVideoTrackOptions docs, deviceId is
        // preferred — position is only the fallback when deviceId is null or
        // not found.
        lp.videoTrackCaptureDefaults = lp.videoTrackCaptureDefaults.copy(
            deviceId = config.deviceId,
            position = position,
        )
        lp.setCameraEnabled(true)
    }

    override suspend fun stopCamera() {
        injectedVideoMutex.withLock {
            val track = injectedVideoTrack
            if (track != null) {
                // unpublishTrack with default stopOnUnpublish=true already
                // calls track.stop() + track.dispose() — calling them again
                // races MediaCodec's event handler against its torn-down
                // thread ("Handler on a dead thread" IllegalStateException).
                room?.localParticipant?.unpublishTrack(track)
                injectedVideoTrack = null
                injectedCapturer = null
                return
            }
        }
        room?.localParticipant?.setCameraEnabled(false)
    }

    // ── StreamingBackend: injected video frames ───────────────────────────────

    override suspend fun injectVideoFrame(
        i420: ByteBuffer,
        width: Int,
        height: Int,
        timestampUs: Long,
    ) {
        if (!isConnected) throw StreamError.NotConnected

        // Fast path — track already exists. Take the same mutex that
        // stopCamera()/tearDown() use when they clear and dispose the
        // capturer, so a concurrent teardown cannot dispose the native
        // capturer between the null-check and pushI420Frame (use-after-dispose).
        injectedVideoMutex.withLock {
            injectedCapturer?.let {
                it.pushI420Frame(i420, width, height, timestampUs)
                return
            }
        }
        // Slow path — first frame: create + publish track under the mutex.
        val capturer = injectedVideoMutex.withLock {
            injectedCapturer?.let { return@withLock it }

            val lp = room?.localParticipant ?: throw StreamError.NotConnected
            val newCapturer = InjectedVideoCapturer()
            val track = lp.createVideoTrack(
                name = "injected-video",
                capturer = newCapturer,
                options = LocalVideoTrackOptions(
                    isScreencast = false,
                    captureParams = VideoCaptureParameter(width, height, 30),
                ),
            )
            track.startCapture()
            lp.publishVideoTrack(track)
            injectedVideoTrack = track
            injectedCapturer = newCapturer
            newCapturer
        }
        capturer.pushI420Frame(i420, width, height, timestampUs)
    }

    // ── StreamingBackend: data channel ─────────────────────────────────────────

    override suspend fun send(data: ByteArray, reliable: Boolean) {
        if (!isConnected) throw StreamError.NotConnected
        // DataPublishOptions carries reliability and optional topic.
        // The hub reads the data channel payload directly — no topic encoding needed.
        room?.localParticipant?.publishData(
            data,
            if (reliable) DataPublishReliability.RELIABLE else DataPublishReliability.LOSSY,
        )
    }

    // ── Event dispatcher ───────────────────────────────────────────────────────

    private fun handleEvent(event: RoomEvent) {
        when (event) {
            is RoomEvent.Reconnecting -> {
                onConnectionStateChanged?.invoke(ConnectionState.RECONNECTING)
            }
            is RoomEvent.Reconnected -> {
                isConnected = true
                onConnectionStateChanged?.invoke(ConnectionState.CONNECTED)
            }
            is RoomEvent.Disconnected -> {
                isConnected = false
                onConnectionStateChanged?.invoke(ConnectionState.DISCONNECTED)
            }
            is RoomEvent.DataReceived -> handleData(event)
            else -> {}
        }
    }

    private fun handleData(event: RoomEvent.DataReceived) {
        val topic = event.topic ?: ""
        val data  = event.data

        // Intercept the reserved agent-status topic — never forward to onDataReceived.
        if (topic == AGENT_STATUS_TOPIC) {
            try {
                val json   = JSONObject(String(data, Charsets.UTF_8))
                val status = json.optString("status")
                if (status.isNotEmpty()) onAgentStatus?.invoke(status)
            } catch (_: Exception) {
                // Malformed payload — ignore.
            }
            return
        }

        onDataReceived?.invoke(topic, data)
    }

    // ── Teardown ──────────────────────────────────────────────────────────────

    private suspend fun tearDown() {
        isConnected = false
        connectionScope?.cancel()
        connectionScope = null
        // Drop any injected video track before disconnecting the room. The
        // room teardown itself unpublishes tracks (which stops + disposes
        // them); avoid a manual stop()/dispose() here so we don't race the
        // MediaCodec event handler against its own torn-down thread.
        injectedVideoMutex.withLock {
            val track = injectedVideoTrack
            if (track != null) {
                runCatching { room?.localParticipant?.unpublishTrack(track) }
            }
            injectedVideoTrack = null
            injectedCapturer = null
        }
        room?.disconnect()
        room = null
        // Emit DISCONNECTED only when we initiated the teardown — a network-driven
        // disconnect will have already fired via RoomEvent.Disconnected.
        onConnectionStateChanged?.invoke(ConnectionState.DISCONNECTED)
    }

    // ── Token fetch ───────────────────────────────────────────────────────────

    /**
     * Fetches a LiveKit JWT from a token endpoint.
     *
     * Called as `GET <tokenURL>?identity=<identity>`. The endpoint must return
     * either a plain JWT string or `{ "token": "eyJ…" }`.
     *
     * Mirrors the web [fetchToken] and Swift [fetchToken] implementations exactly.
     */
    private suspend fun fetchToken(tokenURL: String, identity: String): String =
        withContext(Dispatchers.IO) {
            val encodedIdentity = URLEncoder.encode(identity, "UTF-8")
            val separator       = if (tokenURL.contains('?')) '&' else '?'
            val urlStr          = "$tokenURL${separator}identity=$encodedIdentity"

            val conn = URL(urlStr).openConnection().apply {
                connectTimeout = 5_000
                readTimeout    = 5_000
                (this as? java.net.HttpURLConnection)?.requestMethod = "GET"
            } as java.net.HttpURLConnection

            try {
                val code = conn.responseCode
                val stream = if (code in 200..299) conn.inputStream else conn.errorStream
                val body = stream?.bufferedReader()?.use { it.readText() } ?: ""

                if (code !in 200..299) {
                    throw StreamError.TokenFetchFailed(
                        urlStr,
                        "HTTP $code${if (body.isNotBlank()) ": ${body.take(200)}" else ""}",
                    )
                }

                // JSON form: { "token": "eyJ…" }
                try {
                    val t = JSONObject(body).optString("token")
                    if (t.isNotEmpty()) return@withContext t
                } catch (_: Exception) { /* not JSON, fall through */ }

                // Plain JWT form. Reject HTML/JSON-shaped bodies that didn't have a token.
                val trimmed = body.trim()
                if (trimmed.isNotEmpty() && !trimmed.startsWith("{") && !trimmed.startsWith("<")) {
                    return@withContext trimmed
                }

                throw StreamError.TokenFetchFailed(
                    urlStr,
                    "Unexpected response body: ${body.take(200)}",
                )
            } catch (e: StreamError) {
                throw e
            } catch (e: Exception) {
                // Surface the underlying failure (DNS, ConnectException, SSL, …) — the
                // generic "Token fetch failed" was useless for debugging.
                throw StreamError.TokenFetchFailed(
                    urlStr,
                    "${e.javaClass.simpleName}: ${e.message ?: "no message"}",
                )
            } finally {
                conn.disconnect()
            }
        }

    // ── Constants ──────────────────────────────────────────────────────────────

    companion object {
        /** Reserved LiveKit topic for internal agent status messages. Matches web and Swift. */
        private const val AGENT_STATUS_TOPIC = "_agent.status"
    }
}
