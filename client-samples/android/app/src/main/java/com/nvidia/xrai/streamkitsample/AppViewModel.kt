// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.UUID

// ─────────────────────────────────────────────────────────────────────────────

/** Sentinel camera id for the synthetic "Virtual Camera" provider, which feeds
 *  generated I420 frames through `StreamSession.injectVideoFrame` instead of a
 *  physical Camera2 device. */
const val VIRTUAL_CAMERA_ID = "__virtual_camera__"

/** Synthetic-camera frame interval (~30 fps). */
private const val VIRTUAL_CAMERA_FRAME_MS = 33L

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
    var port by mutableStateOf("8080")
    /** Pre-signed JWT token (alternative to tokenServerURL). */
    var tokenInput by mutableStateOf("")
    /** Token server URL. Defaults to https://<host>:<port>/token when blank. */
    var tokenServerURL by mutableStateOf("")
    var identity by mutableStateOf("android-client")

    // ── Audio settings ─────────────────────────────────────────────────────────

    var audioMode by mutableStateOf(AudioConfig.MicrophoneMode.VOICE_PROCESSING)

    // ── Camera settings ────────────────────────────────────────────────────────

    /** All cameras visible to Camera2 on this device. Populated at construction. */
    val availableCameras: List<CameraInfo> = enumerateCameras(application.applicationContext)

    /**
     * Cameras offered in the selector: the physical Camera2 devices plus the
     * synthetic "Virtual Camera" provider (always available, even on a device
     * or emulator with no real camera), which demonstrates
     * `StreamSession.injectVideoFrame`.
     */
    val selectableCameras: List<CameraInfo> =
        availableCameras + CameraInfo(VIRTUAL_CAMERA_ID, "Virtual Camera (synthetic)", null)

    /**
     * Currently selected camera id. Defaults to the first back-facing camera if
     * any, else the first physical camera, else the virtual camera (so a
     * camera-less device still has a working selection).
     */
    var selectedCameraId by mutableStateOf(
        availableCameras.firstOrNull { it.facing == CameraConfig.CameraFacing.BACK }?.id
            ?: availableCameras.firstOrNull()?.id
            ?: VIRTUAL_CAMERA_ID
    )

    // ── Live state ─────────────────────────────────────────────────────────────

    var connectionState by mutableStateOf(ConnectionState.DISCONNECTED)
        private set
    var agentStatus by mutableStateOf<String?>(null)
        private set
    /** Latest final-reply text from the agent. Drives the Agent panel;
     *  null shows the "Waiting for agent…" placeholder. Mirrors web. */
    var agentResponse by mutableStateOf<String?>(null)
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

    // ── Private ────────────────────────────────────────────────────────────────

    /** Topics carrying the agent's final text reply. Routed to [agentResponse];
     *  never appended to [receivedMessages]. Mirrors the web client. */
    private val agentReplyTopics = setOf("agent.response", "vlm.response")

    /**
     * Active session, exposed for `CameraPreviewView` to render the local
     * camera track.  `null` between connects.  Mutated only from
     * [viewModelScope] coroutines.
     */
    var session by mutableStateOf<StreamSession?>(null)
        private set

    /** Running synthetic-camera frame loop, if the Virtual Camera is active. */
    private var syntheticJob: Job? = null

    // ── Connect / disconnect ──────────────────────────────────────────────────

    fun connect() {
        if (isConnecting || connectionState != ConnectionState.DISCONNECTED) return
        viewModelScope.launch {
            isConnecting = true
            lastError = null
            receivedMessages.clear()

            try {
                val portNumber = port.toIntOrNull() ?: 8080
                val trimmedToken = tokenInput.trim()
                val resolvedTokenURL = tokenServerURL.trim().ifEmpty {
                    "https://$host:$portNumber/token"
                }

                val lkConfig = LiveKitConfig(
                    host = host,
                    port = portNumber,
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
                        // Stop the synthetic feed so it doesn't keep calling
                        // injectVideoFrame on a torn-down session.
                        syntheticJob?.cancel()
                        syntheticJob = null
                        isAudioActive = false
                        isCameraActive = false
                        agentStatus = null
                        agentResponse = null
                    }
                }
                newSession.onAgentStatus = { status ->
                    agentStatus = status
                }
                newSession.onDataReceived = { topic, data ->
                    when {
                        topic in agentReplyTopics -> {
                            // Final agent reply text: drive the Agent panel and
                            // never surface in Received. Matches web's
                            // AGENT_REPLY_TOPICS interceptor.
                            agentResponse = try {
                                String(data, Charsets.UTF_8)
                            } catch (_: Exception) {
                                ""
                            }
                        }
                        topic == "clientControl" -> {
                            // Always-on streaming: clientControl signals from the
                            // agent are silently dropped and never surfaced in the
                            // received messages list.
                        }
                        else -> {
                            val body = try {
                                String(data, Charsets.UTF_8)
                            } catch (_: Exception) {
                                "[${data.size} bytes binary]"
                            }
                            val text = if (topic.isEmpty()) body else "[$topic] $body"
                            receivedMessages.add(0, ReceivedMessage(text = text))
                        }
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
            syntheticJob?.cancelAndJoin()
            syntheticJob = null
            session?.disconnect()
            session = null
            connectionState = ConnectionState.DISCONNECTED
            agentStatus = null
            agentResponse = null
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
        if (selectedCameraId == VIRTUAL_CAMERA_ID) {
            startVirtualCamera()
            return
        }
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

    /**
     * Drives the synthetic "Virtual Camera": a coroutine that generates I420
     * frames and feeds them through the public `injectVideoFrame` API at ~30 fps.
     * No physical camera or CAMERA permission is involved.
     */
    private fun startVirtualCamera() {
        if (connectionState != ConnectionState.CONNECTED) {
            lastError = "Connect before starting the virtual camera."
            return
        }
        if (syntheticJob?.isActive == true) return
        syntheticJob = viewModelScope.launch(Dispatchers.Default) {
            val source = SyntheticCameraSource()
            var frame = 0
            try {
                // Inject the first frame and await it: this lazily publishes the
                // injected camera track. Only AFTER it exists do we flip
                // isCameraActive, so the preview card composes when
                // session.localCameraTrack is already non-null. The getter is
                // not Compose-observable, so a track that appears after the
                // card composes would never be picked up — mirrors the real
                // camera, where startCamera() publishes before we mark active.
                session?.injectVideoFrame(
                    source.renderFrame(frame++), source.width, source.height,
                    System.nanoTime() / 1_000,
                )
                withContext(Dispatchers.Main) { isCameraActive = true }
                while (isActive) {
                    session?.injectVideoFrame(
                        source.renderFrame(frame++), source.width, source.height,
                        System.nanoTime() / 1_000,
                    )
                    delay(VIRTUAL_CAMERA_FRAME_MS)
                }
            } catch (e: CancellationException) {
                throw e // normal stop — let the coroutine cancel
            } catch (e: Exception) {
                // A mid-loop failure may have already published the injected
                // track (first frame succeeded, a later one threw). Unpublish
                // it so we don't leave a live track lingering until the next
                // stopCamera()/disconnect().
                runCatching { session?.stopCamera() }
                withContext(Dispatchers.Main) {
                    lastError = e.message
                    isCameraActive = false
                }
            }
        }
    }

    fun stopCamera() {
        viewModelScope.launch {
            // Stop the synthetic feed BEFORE unpublishing the track, so a
            // trailing frame can't lazily republish the injected track after
            // stopCamera() tore it down.
            syntheticJob?.cancelAndJoin()
            syntheticJob = null
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
