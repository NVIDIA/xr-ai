package com.nvidia.xrai.streamkitsample

import android.app.Application
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.nvidia.xrai.streamkitsample.streamkit.ConnectionState
import com.nvidia.xrai.streamkitsample.streamkit.StreamSession
import com.nvidia.xrai.streamkitsample.streamkit.config.AudioConfig
import com.nvidia.xrai.streamkitsample.streamkit.config.BackendConfiguration
import com.nvidia.xrai.streamkitsample.streamkit.config.CameraConfig
import com.nvidia.xrai.streamkitsample.streamkit.config.LiveKitConfig
import com.nvidia.xrai.streamkitsample.streamkit.config.SessionConfig
import kotlinx.coroutines.launch
import java.util.UUID

// ─────────────────────────────────────────────────────────────────────────────

/** A message received from the agent or other remote participants. */
data class ReceivedMessage(
    val id: String = UUID.randomUUID().toString(),
    val text: String,
    val timestamp: Long = System.currentTimeMillis(),
)

// ─────────────────────────────────────────────────────────────────────────────

/**
 * Observable state shared across the sample app.
 *
 * Mirrors AppModel.swift and app.js field-for-field. All mutable state is
 * exposed as Compose [androidx.compose.runtime.State] so the UI recomposes
 * automatically.
 */
class AppViewModel(application: Application) : AndroidViewModel(application) {

    // ── Connection settings ────────────────────────────────────────────────────

    var host by mutableStateOf("192.168.1.100")
    var port by mutableStateOf("7880")
    /**
     * When true, the default token URL uses `https://`. Has no effect on the
     * LiveKit signaling socket — that is always plain `ws://` in the reference
     * deployment.
     */
    var secure by mutableStateOf(false)
    /** Pre-signed JWT token (alternative to tokenServerURL). */
    var tokenInput by mutableStateOf("")
    /** Token server URL. Defaults to http(s)://<host>:8080/token when blank. */
    var tokenServerURL by mutableStateOf("")
    var identity by mutableStateOf("android-client")

    // ── Audio settings ─────────────────────────────────────────────────────────

    var audioMode by mutableStateOf(AudioConfig.MicrophoneMode.VOICE_PROCESSING)

    // ── Camera settings ────────────────────────────────────────────────────────

    /** All cameras visible to Camera2 on this device. Populated at construction. */
    val availableCameras: List<CameraInfo> = enumerateCameras(application.applicationContext)

    /**
     * Currently selected Camera2 id. Defaults to the first back-facing camera
     * if any, else the first camera, else null (device has no camera).
     */
    var selectedCameraId by mutableStateOf(
        availableCameras.firstOrNull { it.facing == CameraConfig.CameraFacing.BACK }?.id
            ?: availableCameras.firstOrNull()?.id
    )

    // ── Live state ─────────────────────────────────────────────────────────────

    var connectionState by mutableStateOf(ConnectionState.DISCONNECTED)
        private set
    var agentStatus by mutableStateOf<String?>(null)
        private set
    var isAudioActive by mutableStateOf(false)
        private set
    var isCameraActive by mutableStateOf(false)
        private set
    var isConnecting by mutableStateOf(false)
        private set
    val receivedMessages = mutableStateListOf<ReceivedMessage>()
    var lastError by mutableStateOf<String?>(null)
        private set
    /** When true, ``clientControl`` startCamera/stopCamera messages from the
     *  agent are honoured.  When false (default — always-on), they are ignored
     *  and the camera button is the sole control. */
    var cameraOnDemand by mutableStateOf(false)

    // ── Private ────────────────────────────────────────────────────────────────

    private var session: StreamSession? = null

    // ── Connect / disconnect ──────────────────────────────────────────────────

    fun connect() {
        if (isConnecting || connectionState != ConnectionState.DISCONNECTED) return
        viewModelScope.launch {
            isConnecting = true
            lastError = null
            receivedMessages.clear()

            try {
                val portNumber = port.toIntOrNull() ?: 7880
                val trimmedToken = tokenInput.trim()
                val tokenScheme = if (secure) "https" else "http"
                val resolvedTokenURL = tokenServerURL.trim().ifEmpty {
                    // Default mirrors AppModel.swift: http(s)://<host>:8080/token
                    "$tokenScheme://$host:8080/token"
                }

                val lkConfig = LiveKitConfig(
                    host = host,
                    port = portNumber,
                    secure = secure,
                    token = trimmedToken.ifEmpty { null },
                    tokenURL = resolvedTokenURL,
                )

                val newSession = StreamSession(
                    BackendConfiguration.LiveKit(lkConfig),
                    getApplication(),
                )

                // Wire callbacks before connecting — same ordering as iOS/web.
                newSession.onConnectionStateChanged = { state ->
                    connectionState = state
                    if (state == ConnectionState.DISCONNECTED) {
                        isAudioActive = false
                        isCameraActive = false
                        agentStatus = null
                    }
                }
                newSession.onAgentStatus = { status ->
                    agentStatus = status
                }
                newSession.onDataReceived = { topic, data ->
                    if (topic == "clientControl") {
                        // Camera on demand: intercept clientControl signals from the agent.
                        // In always-on mode (cameraOnDemand = false) they are silently ignored.
                        // Never surface in the received messages list.
                        if (cameraOnDemand) {
                            try {
                                val json = org.json.JSONObject(String(data, Charsets.UTF_8))
                                when (json.optString("action")) {
                                    "startCamera" -> if (!isCameraActive) startCamera()
                                    "stopCamera"  -> if (isCameraActive) stopCamera()
                                }
                            } catch (_: Exception) { /* malformed — ignore */ }
                        }
                    } else {
                        val body = try {
                            String(data, Charsets.UTF_8)
                        } catch (_: Exception) {
                            "[${data.size} bytes binary]"
                        }
                        val text = if (topic.isEmpty()) body else "[$topic] $body"
                        receivedMessages.add(0, ReceivedMessage(text = text))
                    }
                }

                session = newSession
                newSession.connect(SessionConfig(identity = identity))

            } catch (e: Exception) {
                lastError = e.message ?: "Connection failed"
                session?.disconnect()
                session = null
                connectionState = ConnectionState.DISCONNECTED
            } finally {
                isConnecting = false
            }
        }
    }

    fun disconnect() {
        viewModelScope.launch {
            session?.disconnect()
            session = null
            connectionState = ConnectionState.DISCONNECTED
            agentStatus = null
            isAudioActive = false
            isCameraActive = false
        }
    }

    // ── Audio ──────────────────────────────────────────────────────────────────

    fun startAudio() {
        viewModelScope.launch {
            try {
                session?.startAudio(AudioConfig(mode = audioMode))
                isAudioActive = true
            } catch (e: Exception) {
                lastError = e.message
            }
        }
    }

    fun stopAudio() {
        viewModelScope.launch {
            try {
                session?.stopAudio()
            } catch (e: Exception) {
                lastError = e.message
            }
            isAudioActive = false
        }
    }

    // ── Camera ─────────────────────────────────────────────────────────────────

    fun startCamera() {
        viewModelScope.launch {
            try {
                val info = availableCameras.firstOrNull { it.id == selectedCameraId }
                val facing = info?.facing ?: CameraConfig.CameraFacing.BACK
                session?.startCamera(CameraConfig(deviceId = selectedCameraId, facing = facing))
                isCameraActive = true
            } catch (e: Exception) {
                lastError = e.message
            }
        }
    }

    fun stopCamera() {
        viewModelScope.launch {
            try {
                session?.stopCamera()
            } catch (e: Exception) {
                lastError = e.message
            }
            isCameraActive = false
        }
    }

    // ── Data channel ──────────────────────────────────────────────────────────

    fun sendPing() {
        // In on-demand mode, start the camera now so it warms up in parallel
        // with the ping's round-trip and agent processing.
        if (cameraOnDemand && !isCameraActive) startCamera()
        viewModelScope.launch {
            try {
                session?.send("ping".toByteArray(Charsets.UTF_8))
            } catch (e: Exception) {
                lastError = e.message
            }
        }
    }

    fun sendCustom(text: String) {
        if (text.isBlank()) return
        viewModelScope.launch {
            try {
                session?.send(text.toByteArray(Charsets.UTF_8))
            } catch (e: Exception) {
                lastError = e.message
            }
        }
    }

    // ── Error ──────────────────────────────────────────────────────────────────

    fun clearError() {
        lastError = null
    }
}
