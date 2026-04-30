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
import io.livekit.android.LiveKitOverrides
import io.livekit.android.events.RoomEvent
import io.livekit.android.events.collect
import io.livekit.android.room.Room
import io.livekit.android.room.track.CameraPosition
import io.livekit.android.room.track.DataPublishReliability
import io.livekit.android.room.track.LocalVideoTrackOptions
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import javax.net.ssl.HttpsURLConnection

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

    // ── Private state ──────────────────────────────────────────────────────────

    private var room: Room? = null
    private var isConnected = false

    /** Coroutine scope active for the lifetime of one connection. */
    private var connectionScope: CoroutineScope? = null

    // ── StreamingBackend: connect / disconnect ─────────────────────────────────

    override suspend fun connect(sessionConfig: SessionConfig) {
        tearDown()

        if (config.host.isBlank()) throw StreamError.InvalidHost(config.host)

        // LiveKit signaling on port 7880 is plain ws:// in the xr-ai reference
        // deployment — TLS only terminates at the web/token server. The
        // `secure` flag therefore controls the token URL scheme only, not this.
        val wsUrl = "ws://${config.host}:${config.port}"

        val token = when {
            !config.token.isNullOrBlank() -> config.token
            !config.tokenURL.isNullOrBlank() -> fetchToken(config.tokenURL, sessionConfig.identity)
            else -> throw StreamError.MissingToken
        }

        // Trust all certs — the hub uses a self-signed cert by default and we
        // don't want to require users to manually install a CA. See TrustAllCerts.
        val newRoom = LiveKit.create(
            appContext,
            overrides = LiveKitOverrides(okHttpClient = TrustAllCerts.okHttpClient()),
        )
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
        room?.localParticipant?.setCameraEnabled(false)
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

            val conn = (URL(urlStr).openConnection() as HttpURLConnection).apply {
                connectTimeout = 5_000
                readTimeout    = 5_000
                requestMethod  = "GET"
                if (this is HttpsURLConnection) {
                    // Match the WebSocket's trust-all behavior so the same self-signed
                    // dev cert works for both the token endpoint and the LiveKit socket.
                    sslSocketFactory  = TrustAllCerts.socketFactory
                    hostnameVerifier  = TrustAllCerts.permissiveHostnameVerifier
                }
            }

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
