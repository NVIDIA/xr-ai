// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.xrai.streamkitsample.streamkit.backends.livekit

import android.content.Context
import com.nvidia.xrai.streamkitsample.streamkit.ConnectionState
import com.nvidia.xrai.streamkitsample.streamkit.CapturedImage
import com.nvidia.xrai.streamkitsample.streamkit.FileSendOptions
import com.nvidia.xrai.streamkitsample.streamkit.FileTransferInfo
import com.nvidia.xrai.streamkitsample.streamkit.NetworkMetrics
import com.nvidia.xrai.streamkitsample.streamkit.NetworkQuality
import com.nvidia.xrai.streamkitsample.streamkit.StreamError
import com.nvidia.xrai.streamkitsample.streamkit.captureJpeg
import com.nvidia.xrai.streamkitsample.streamkit.backends.StreamingBackend
import com.nvidia.xrai.streamkitsample.streamkit.config.AudioConfig
import com.nvidia.xrai.streamkitsample.streamkit.config.BackendConfiguration
import com.nvidia.xrai.streamkitsample.streamkit.config.CameraConfig
import com.nvidia.xrai.streamkitsample.streamkit.config.CameraEncodingConfig
import com.nvidia.xrai.streamkitsample.streamkit.config.VideoQualityPreference
import com.nvidia.xrai.streamkitsample.streamkit.config.LiveKitConfig
import com.nvidia.xrai.streamkitsample.streamkit.config.SessionConfig
import io.livekit.android.ConnectOptions
import io.livekit.android.LiveKit
import io.livekit.android.events.RoomEvent
import io.livekit.android.events.collect
import io.livekit.android.renderer.TextureViewRenderer
import io.livekit.android.room.Room
import io.livekit.android.room.participant.ConnectionQuality
import io.livekit.android.room.participant.Participant
import io.livekit.android.room.participant.LocalParticipant
import io.livekit.android.room.participant.VideoTrackPublishOptions
import io.livekit.android.room.participant.VideoTrackPublishDefaults
import io.livekit.android.room.track.CameraPosition
import io.livekit.android.room.track.DataPublishReliability
import io.livekit.android.room.track.LocalVideoTrack
import io.livekit.android.room.track.LocalVideoTrackOptions
import io.livekit.android.room.track.RemoteTrackPublication
import io.livekit.android.room.track.Track
import io.livekit.android.room.track.VideoCaptureParameter
import io.livekit.android.room.track.VideoEncoding
import io.livekit.android.util.LKLog
import livekit.org.webrtc.RtpParameters
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.ensureActive
import kotlin.coroutines.coroutineContext
import kotlin.coroutines.CoroutineContext
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.net.URL
import java.net.URLEncoder
import java.nio.ByteBuffer
import java.util.concurrent.atomic.AtomicLong
import kotlin.coroutines.resume

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
    private val createRoom: (Context) -> Room = { LiveKit.create(it) },
    private val eventContext: CoroutineContext = Dispatchers.Main,
) : StreamingBackend {

    // ── Event hooks ────────────────────────────────────────────────────────────

    override var onConnectionStateChanged: ((ConnectionState) -> Unit)? = null
    override var onDataReceived: ((topic: String, data: ByteArray) -> Unit)? = null
    override var onAgentStatus: ((status: String) -> Unit)? = null
    override var onNetworkMetrics: ((metrics: NetworkMetrics) -> Unit)? = null

    // ── Public local preview accessor ─────────────────────────────────────────

    /**
     * Actively capturing local camera track, or null when stopped. Used by
     * `CameraPreviewView` to render the outgoing stream locally; app code
     * goes through that composable rather than touching this directly.
     */
    val localCameraTrack: LocalVideoTrack?
        get() {
            val publication = room?.localParticipant
                ?.getTrackPublication(Track.Source.CAMERA) ?: return null
            if (publication.muted) return null
            return publication.track as? LocalVideoTrack
        }

    /** Initialises a [TextureViewRenderer] with the connected room's EGL
     *  context so it can sink frames from the local camera track. No-op when
     *  the room is not connected. */
    fun initVideoRenderer(view: TextureViewRenderer) {
        room?.initVideoRenderer(view)
    }

    // ── Private state ──────────────────────────────────────────────────────────

    @Volatile private var room: Room? = null
    @Volatile private var isConnected = false
    private val connectionGeneration = AtomicLong()
    private val byteStreamWriter = LiveKitByteStreamWriter(
        snapshotConnection = {
            room?.takeIf { isConnected }?.let {
                ByteStreamConnection(it, connectionGeneration.get())
            }
        },
        isConnectionActive = { connection ->
            isConnected &&
                room === connection.room &&
                connectionGeneration.get() == connection.generation
        },
    )

    /** Coroutine scope active for the lifetime of one connection. */
    @Volatile private var connectionScope: CoroutineScope? = null
    private class RoomLifetime(val room: Room) {
        val terminal = CompletableDeferred<Unit>()
        @Volatile var connectEntered = false
    }
    @Volatile private var roomLifetime: RoomLifetime? = null
    private data class RoomCloseFailure(val lifetime: RoomLifetime, val error: Throwable)
    private enum class TeardownPurpose { DISCONNECT, RECONNECT }
    private var failedRoomClose: RoomCloseFailure? = null

    // ── Injected video state (external frame source) ───────────────────────────

    /** Custom capturer that lets us push externally-sourced frames into LiveKit. */
    @Volatile private var injectedCapturer: InjectedVideoCapturer? = null
    private var injectedVideoTrack: LocalVideoTrack? = null
    /** Only explicit injected allocations belong to StreamKit. */
    private class CameraAllocation(
        val track: LocalVideoTrack,
        val participant: LocalParticipant,
        val capturer: InjectedVideoCapturer,
        val lifetime: RoomLifetime,
    ) {
        val room: Room get() = lifetime.room
        var disposalFailure: Throwable? = null
    }
    @Volatile private var ownedCamera: CameraAllocation? = null
    private val ownedCameraTrack: LocalVideoTrack? get() = ownedCamera?.track
    // Device capture and injected frames share the SDK's CAMERA publication.
    private val cameraMutex = Mutex()
    private var cameraPublishBaseline: VideoTrackPublishDefaults? = null

    // ── StreamingBackend: connect / disconnect ─────────────────────────────────

    override suspend fun connect(sessionConfig: SessionConfig) {
        tearDown(TeardownPurpose.RECONNECT)
        val generation = connectionGeneration.get()

        if (config.host.isBlank()) throw StreamError.InvalidHost(config.host)

        val wsUrl = "wss://${config.host}:${config.port}"

        val token = when {
            !config.token.isNullOrBlank() -> config.token
            !config.tokenURL.isNullOrBlank() -> fetchToken(config.tokenURL, sessionConfig.identity)
            else -> throw StreamError.MissingToken
        }
        if (connectionGeneration.get() != generation) throw StreamError.NotConnected

        val newRoom = createRoom(appContext)
        val lifetime = RoomLifetime(newRoom)
        roomLifetime = lifetime
        room = newRoom
        cameraPublishBaseline = newRoom.localParticipant.videoTrackPublishDefaults

        val scope = CoroutineScope(eventContext + SupervisorJob())
        connectionScope = scope

        // Collect all room events for state changes and data.
        // Must be launched before room.connect() so we don't miss RECONNECTING /
        // DISCONNECTED events that arrive immediately after a connect.
        scope.launch(start = CoroutineStart.UNDISPATCHED) {
            newRoom.events.collect { event ->
                if (event is RoomEvent.Disconnected) {
                    lifetime.terminal.complete(Unit)
                    try { handleEvent(newRoom, event) } finally { scope.cancel() }
                }
            }
        }
        // Application event delivery keeps its existing error policy, but a
        // throwing callback cannot remove the independent custody observer.
        scope.launch(start = CoroutineStart.UNDISPATCHED) {
            newRoom.events.collect { event ->
                if (event !is RoomEvent.Disconnected) handleEvent(newRoom, event)
            }
        }

        // Emit CONNECTING before the async handshake.
        try {
            onConnectionStateChanged?.invoke(ConnectionState.CONNECTING)
            coroutineContext.ensureActive()
        } catch (error: Throwable) {
            // The SDK was never entered: it cannot owe a terminal event and
            // no application camera allocation was admitted for this room.
            if (room === newRoom) room = null
            if (roomLifetime === lifetime) roomLifetime = null
            if (connectionScope === scope) connectionScope = null
            scope.cancel()
            throw error
        }

        // room.connect() suspends until the room is fully connected or throws.
        // Audio isolation: when hubIdentity is set, disable auto-subscribe and
        // subscribe only to the hub participant's tracks (below + in
        // handleTrackPublished), so a client never receives another
        // participant's microphone.
        lifetime.connectEntered = true
        newRoom.connect(
            wsUrl,
            token,
            ConnectOptions(autoSubscribe = config.hubIdentity == null),
        )
        if (room !== newRoom || connectionGeneration.get() != generation) {
            withContext(NonCancellable) { newRoom.disconnect() }
            throw StreamError.NotConnected
        }

        // Subscribe to any hub tracks already published before we connected.
        config.hubIdentity?.let { hub ->
            newRoom.remoteParticipants.values
                .filter { it.identity?.value == hub }
                .forEach { p ->
                    p.trackPublications.values.forEach { pub ->
                        (pub as? RemoteTrackPublication)?.setSubscribed(true)
                    }
                }
        }

        // Successfully connected.
        isConnected = true
        onConnectionStateChanged?.invoke(ConnectionState.CONNECTED)
        scope.launch(Dispatchers.Default) {
            while (isActive && room === newRoom) {
                if (isConnected) {
                    val metrics = collectNetworkMetrics(newRoom)
                    if (isConnected && room === newRoom) {
                        withContext(Dispatchers.Main.immediate) {
                            if (isConnected && room === newRoom) {
                                onNetworkMetrics?.invoke(metrics)
                            }
                        }
                    }
                }
                delay(1_000)
            }
        }
    }

    override suspend fun disconnect() {
        tearDown(TeardownPurpose.DISCONNECT)
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

    private fun videoPublishDefaults(encoding: CameraEncodingConfig?): VideoTrackPublishDefaults {
        val current = room?.localParticipant?.videoTrackPublishDefaults ?: VideoTrackPublishDefaults()
        val baseline = cameraPublishBaseline ?: current
        val preference = when (encoding?.qualityPreference) {
            VideoQualityPreference.DETAIL -> RtpParameters.DegradationPreference.MAINTAIN_RESOLUTION
            VideoQualityPreference.MOTION -> RtpParameters.DegradationPreference.MAINTAIN_FRAMERATE
            VideoQualityPreference.BALANCED -> RtpParameters.DegradationPreference.BALANCED
            null -> baseline.degradationPreference
        }
        return current.copy(
            videoEncoding = encoding?.let { VideoEncoding(it.maxBitrateBps, it.maxFramerate) } ?: baseline.videoEncoding,
            simulcast = encoding?.simulcast ?: baseline.simulcast,
            simulcastLayers = baseline.simulcastLayers,
            degradationPreference = preference,
        )
    }

    override suspend fun startCamera(config: CameraConfig) = cameraMutex.withLock {
        if (!isConnected) throw StreamError.CameraRequiresConnection

        val lp = room?.localParticipant ?: throw StreamError.CameraRequiresConnection
        ownedCameraTrack?.let { track ->
            if (lp.getTrackPublication(Track.Source.CAMERA)?.track !== track) stopCameraLocked()
        }
        var publishDefaults = videoPublishDefaults(config.encoding)
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
        if (config.encoding != null && publishDefaults.simulcast) {
            // The public cached-track accessor resolves the same supported
            // hardware format without starting a second capture allocation.
            // The SDK retains this track through mute/unpublish/disconnect.
            val dimensions = lp.getOrCreateDefaultVideoTrack().dimensions
            publishDefaults = cameraLayerDefaults(
                config.encoding, publishDefaults, dimensions.width, dimensions.height,
            )
        }
        // Compare effective options, not the unscaled baseline, so an unchanged
        // policy does not interrupt a retained publication.
        if (injectedVideoTrack != null || lp.videoTrackPublishDefaults != publishDefaults) {
            val previous = lp.getTrackPublication(Track.Source.CAMERA)?.track
            val wasOwned = ownedCameraTrack != null
            stopCameraLocked()
            if (!wasOwned) previous?.let { lp.unpublishTrack(it) }
            lp.videoTrackPublishDefaults = publishDefaults
        }
        check(lp.setCameraEnabled(true)) { "Camera publication failed" }
        Unit
    }

    override suspend fun stopCamera() = cameraMutex.withLock { stopCameraLocked() }

    /** Caller owns cameraMutex; never re-enter the public stop operation. */
    private suspend fun stopCameraLocked() {
        // Full reconnect temporarily moves publications into an SDK-private
        // republish pool. Registry absence cannot authorize disposing its tracks.
        if (room != null && (!isConnected || room?.state != Room.State.CONNECTED)) throw StreamError.NotConnected
        val allocation = ownedCamera
        if (allocation != null) {
            releaseOwnedCamera(allocation.track)
            injectedVideoTrack = null
            injectedCapturer = null
            return
        }
        room?.localParticipant?.setCameraEnabled(false)
    }

    private fun releaseOwnedCamera(track: LocalVideoTrack) {
        val allocation = ownedCamera?.takeIf { it.track === track } ?: return
        val lp = allocation.participant
        allocation.disposalFailure?.let { throw it }
        if (allocation.capturer.disposed) {
            // SDK invoked this exact capturer's terminal callback. Relinquish
            // SDK-owned cleanup; it does not acknowledge private source success.
            ownedCamera = null
            return
        }
        check(allocation.room.state == Room.State.CONNECTED || allocation.lifetime.terminal.isCompleted) {
            "SDK camera cleanup has not returned capturer custody"
        }
        var failure: Throwable? = null
        if (lp.getTrackPublication(Track.Source.CAMERA)?.track === track) {
            // A failed unpublish can still leave SDK custody intact. Do not
            // dispose a track the SDK may subsequently republish or release.
            lp.unpublishTrack(track)
        }
        try { track.stopCapture() } catch (error: Throwable) { if (failure == null) failure = error }
        try { track.stop() } catch (error: Throwable) { if (failure == null) failure = error }
        try { track.dispose() } catch (error: Throwable) {
            // Native dispose may have partially released the graph. Retain the
            // actual failure and close admission; blind disposal retry is unsafe.
            allocation.disposalFailure = error
            if (failure != null) error.addSuppressed(failure)
            throw error
        }
        if (ownedCameraTrack === track) ownedCamera = null
        failure?.let { throw it }
    }

    override suspend fun captureImage(config: CameraConfig): CapturedImage {
        if (!isConnected) throw StreamError.NotConnected
        localCameraTrack?.let { return CapturedImage(it.captureJpeg()) }
        val participant = room?.localParticipant ?: throw StreamError.NotConnected
        val position = when (config.facing) {
            CameraConfig.CameraFacing.FRONT -> CameraPosition.FRONT
            CameraConfig.CameraFacing.BACK -> CameraPosition.BACK
        }
        val track = participant.createVideoTrack(
            name = "still-capture",
            options = participant.videoTrackCaptureDefaults.copy(
                deviceId = config.deviceId,
                position = position,
            ),
        )
        return try {
            track.startCapture()
            CapturedImage(track.captureJpeg())
        } finally {
            track.stop()
            track.dispose()
        }
    }

    // ── StreamingBackend: injected video frames ───────────────────────────────

    override suspend fun injectVideoFrame(
        i420: ByteBuffer,
        width: Int,
        height: Int,
        timestampUs: Long,
    ) = injectVideoFrame(i420, width, height, timestampUs, null)

    override suspend fun injectVideoFrame(
        i420: ByteBuffer,
        width: Int,
        height: Int,
        timestampUs: Long,
        encoding: CameraEncodingConfig?,
    ) {
        cameraMutex.withLock {
            // Recheck after waiting: teardown closes admission before acquiring
            // the lock, so a queued frame cannot recreate a disconnected track.
            if (!isConnected) throw StreamError.NotConnected
            val currentRoom = room ?: throw StreamError.NotConnected
            val lifetime = roomLifetime?.takeIf { it.room === currentRoom } ?: throw StreamError.NotConnected
            val lp = currentRoom.localParticipant
            if (injectedVideoTrack != null &&
                lp.getTrackPublication(Track.Source.CAMERA)?.track !== injectedVideoTrack) {
                stopCameraLocked()
            }
            injectedCapturer?.let {
                it.pushI420Frame(i420, width, height, timestampUs)
                return
            }
            // Lookup and remove the existing CAMERA publication inside this
            // transaction, before any suspension. First injection preempts the
            // device camera; failed publication leaves no camera published.
            ownedCameraTrack?.let { track ->
                if (lp.getTrackPublication(Track.Source.CAMERA)?.track !== track) releaseOwnedCamera(track)
            }
            lp.getTrackPublication(Track.Source.CAMERA)?.track?.let {
                if (it === ownedCameraTrack) releaseOwnedCamera(it as LocalVideoTrack)
                else lp.unpublishTrack(it)
            }
            val newCapturer = InjectedVideoCapturer()
            val track = lp.createVideoTrack(
                name = "injected-video",
                capturer = newCapturer,
                options = LocalVideoTrackOptions(
                    isScreencast = false,
                    captureParams = VideoCaptureParameter(width, height, 30),
                ),
            )
            val allocation = CameraAllocation(track, lp, newCapturer, lifetime)
            ownedCamera = allocation
            try {
                track.startCapture()
                // Publish as CAMERA for both the preview and the hub.
                check(lp.publishVideoTrack(track, VideoTrackPublishOptions(
                    base = cameraLayerDefaults(encoding, videoPublishDefaults(encoding), width, height),
                    source = Track.Source.CAMERA,
                ))) { "Injected camera publication failed" }
                if (!isConnected || room !== currentRoom) throw StreamError.NotConnected
                injectedVideoTrack = track
                injectedCapturer = newCapturer
                // First delivery has the same lifetime protection as cached
                // pushes: stop/switch cannot dispose the capturer in between.
                newCapturer.pushI420Frame(i420, width, height, timestampUs)
            } catch (error: Throwable) {
                injectedVideoTrack = null
                injectedCapturer = null
                // Publish can fail/cancel before the SDK owns the track. Its
                // unpublish operation is a no-op for an unpublished track.
                releaseOwnedCamera(track)
                throw error
            }
        }
    }

    // ── StreamingBackend: data channel ─────────────────────────────────────────

    override suspend fun send(data: ByteArray, reliable: Boolean) {
        if (!isConnected) throw StreamError.NotConnected
        // Address outbound data to the hub participant only so it is not
        // broadcast to — and surfaced by — other participants sharing the room.
        // Clients only ever talk to the hub. null hubIdentity → whole-room broadcast.
        val identities = config.hubIdentity?.let { listOf(Participant.Identity(it)) }
        room?.localParticipant?.publishData(
            data,
            if (reliable) DataPublishReliability.RELIABLE else DataPublishReliability.LOSSY,
            identities = identities,
        )
    }

    internal suspend fun sendByteStream(
        data: ByteArray,
        topic: String,
        attributes: Map<String, String>,
        mimeType: String,
        name: String,
    ): String {
        if (!isConnected) throw StreamError.NotConnected
        return byteStreamWriter.sendBytes(
            data,
            makeByteStreamWireOptions(
                topic = topic,
                attributes = attributes,
                mimeType = mimeType,
                name = name,
                totalSize = data.size.toLong(),
            ),
        )
    }

    private fun makeByteStreamWireOptions(
        topic: String,
        attributes: Map<String, String>,
        mimeType: String,
        name: String,
        totalSize: Long,
    ) = ByteStreamWireOptions(
        topic = topic,
        attributes = attributes,
        destinationIdentities = config.hubIdentity
            ?.let { listOf(Participant.Identity(it)) }
            .orEmpty(),
        mimeType = mimeType,
        name = name,
        totalSize = totalSize,
    )

    override suspend fun sendBytes(
        data: ByteArray,
        options: FileSendOptions,
    ): FileTransferInfo {
        val effective = makeFileOptions(options, data.size.toLong(), null)
        if (!isConnected || room == null) throw StreamError.NotConnected
        val streamId = try {
            byteStreamWriter.sendBytes(data, effective.wire)
        } catch (_: ByteStreamConnectionChanged) {
            throw StreamError.FileTransferIncomplete
        }
        return effective.toTransferInfo(streamId)
    }

    override suspend fun sendFile(file: File, options: FileSendOptions): FileTransferInfo {
        var effective = makeFileOptions(options, 0, file.name)
        if (!isConnected || room == null) throw StreamError.NotConnected
        if (!file.isFile) throw StreamError.InvalidFileMetadata("file is not readable")
        val size = file.length()
        effective = effective.copy(
            size = size,
            wire = effective.wire.copy(totalSize = size),
        )
        val streamId = try {
            byteStreamWriter.sendFile(file, effective.wire)
        } catch (_: ByteStreamConnectionChanged) {
            throw StreamError.FileTransferIncomplete
        }
        return effective.toTransferInfo(streamId)
    }

    private data class EffectiveFileOptions(
        val topic: String,
        val name: String,
        val mimeType: String,
        val size: Long,
        val wire: ByteStreamWireOptions,
    ) {
        fun toTransferInfo(streamId: String) = FileTransferInfo(
            id = streamId,
            topic = topic,
            name = name,
            mimeType = mimeType,
            size = size,
        )
    }

    private fun makeFileOptions(
        options: FileSendOptions,
        size: Long,
        defaultName: String?,
    ): EffectiveFileOptions {
        val topic = validateFileField(options.topic, "topic", FILE_MAX_FIELD_BYTES)
        if (topic.startsWith(FILE_RESERVED_PREFIX)) {
            throw StreamError.InvalidFileMetadata("topic uses the reserved _streamkit. prefix")
        }
        val name = validateFileField(options.name ?: defaultName, "name", FILE_MAX_FIELD_BYTES)
        val mimeType = validateFileField(
            options.mimeType?.takeUnless(String::isEmpty) ?: "application/octet-stream",
            "MIME type",
            FILE_MAX_FIELD_BYTES,
        )
        val attributes = options.attributes.toMutableMap()
        if (attributes.size > FILE_MAX_ATTRIBUTES) {
            throw StreamError.InvalidFileMetadata("attributes exceed 32 application entries")
        }
        var attributeBytes = 0
        attributes.forEach { (key, value) ->
            validateFileField(key, "attribute key", FILE_MAX_ATTRIBUTE_KEY_BYTES)
            validateFileField(
                value,
                "attribute value",
                FILE_MAX_ATTRIBUTE_VALUE_BYTES,
                allowsEmpty = true,
            )
            if (key.startsWith(FILE_RESERVED_PREFIX)) {
                throw StreamError.InvalidFileMetadata("attribute $key is reserved")
            }
            attributeBytes += key.toByteArray(Charsets.UTF_8).size
            attributeBytes += value.toByteArray(Charsets.UTF_8).size
        }
        if (attributeBytes > FILE_MAX_ATTRIBUTES_BYTES) {
            throw StreamError.InvalidFileMetadata("attributes exceed 8192 UTF-8 bytes")
        }
        attributes[FILE_TOPIC_ATTRIBUTE] = topic
        return EffectiveFileOptions(
            topic = topic,
            name = name,
            mimeType = mimeType,
            size = size,
            wire = makeByteStreamWireOptions(
                topic = FILE_STREAM_TOPIC,
                attributes = attributes,
                mimeType = mimeType,
                name = name,
                totalSize = size,
            ),
        )
    }

    private fun validateFileField(
        value: String?,
        name: String,
        maxBytes: Int,
        allowsEmpty: Boolean = false,
    ): String {
        if (value == null || (!allowsEmpty && value.isEmpty())) {
            throw StreamError.InvalidFileMetadata("$name must be nonempty")
        }
        if ('\u0000' in value) {
            throw StreamError.InvalidFileMetadata("$name cannot contain NUL")
        }
        if (value.toByteArray(Charsets.UTF_8).size > maxBytes) {
            throw StreamError.InvalidFileMetadata("$name exceeds $maxBytes UTF-8 bytes")
        }
        return value
    }
    // ── Event dispatcher ───────────────────────────────────────────────────────

    private suspend fun handleEvent(eventRoom: Room, event: RoomEvent) {
        if (event is RoomEvent.Disconnected) {
            roomLifetime?.takeIf { it.room === eventRoom }?.terminal?.complete(Unit)
            ownedCamera?.takeIf { it.room === eventRoom }?.lifetime?.terminal?.complete(Unit)
            // An old room's SDK cleanup can finish after transport admission
            // has moved on. Reclaim only that room's actual retained allocation.
            if (room !== eventRoom) {
                withContext(NonCancellable) {
                    cameraMutex.withLock {
                        if (failedRoomClose?.lifetime?.room === eventRoom) failedRoomClose = null
                        ownedCamera?.takeIf { it.room === eventRoom }?.let {
                            // isolation boundary: retain the failed owner; this observer has no caller to report to.
                            try { releaseOwnedCamera(it.track) }
                            catch (error: Throwable) { LKLog.e(error) { "Retained camera cleanup failed" } }
                        }
                    }
                }
                return
            }
        }
        if (room !== eventRoom) return
        when (event) {
            is RoomEvent.Reconnecting -> {
                connectionGeneration.incrementAndGet()
                isConnected = false
                onConnectionStateChanged?.invoke(ConnectionState.RECONNECTING)
            }
            is RoomEvent.Reconnected -> {
                isConnected = true
                onConnectionStateChanged?.invoke(ConnectionState.CONNECTED)
            }
            is RoomEvent.Disconnected -> {
                val generation = connectionGeneration.incrementAndGet()
                val oldScope = connectionScope
                isConnected = false
                room = null
                // Published tracks were released by SDK cleanup. Unpublished
                // allocations were not in its registry and still belong to us.
                withContext(NonCancellable) {
                    cameraMutex.withLock {
                        val allocation = ownedCamera
                        if (allocation?.participant === eventRoom.localParticipant) {
                            // isolation boundary: terminal notification must survive retained camera cleanup failure.
                            try { releaseOwnedCamera(allocation.track) }
                            catch (error: Throwable) { LKLog.e(error) { "Camera cleanup after terminal disconnect failed" } }
                            if (injectedVideoTrack === allocation.track) {
                                injectedVideoTrack = null
                                injectedCapturer = null
                            }
                        }
                        if (room == null) cameraPublishBaseline = null
                    }
                }
                oldScope?.cancel()
                if (connectionScope === oldScope) connectionScope = null
                if (room == null && connectionGeneration.get() == generation) {
                    onConnectionStateChanged?.invoke(ConnectionState.DISCONNECTED)
                }
            }
            is RoomEvent.DataReceived -> handleData(event)
            is RoomEvent.TrackPublished -> {
                // Audio isolation: subscribe only to the hub participant's tracks.
                if (config.hubIdentity != null &&
                    event.participant.identity?.value == config.hubIdentity
                ) {
                    (event.publication as? RemoteTrackPublication)?.setSubscribed(true)
                }
            }
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

        // Isolation: only surface data from the hub/agent, never from peer
        // participants. null hubIdentity keeps the legacy accept-from-anyone behaviour.
        if (config.hubIdentity != null && event.participant?.identity?.value != config.hubIdentity) {
            return
        }

        onDataReceived?.invoke(topic, data)
    }

    private suspend fun collectNetworkMetrics(room: Room): NetworkMetrics {
        val quality = room.localParticipant.connectionQuality.toStreamKitQuality()
        val hasMediaTracks = room.localParticipant.trackPublications.values.any { it.track != null } ||
            room.remoteParticipants.values.any { participant ->
                participant.trackPublications.values.any { it.track != null }
            }
        if (!hasMediaTracks) return NetworkMetrics(quality = quality)

        val publisher = runCatching { publisherStats(room) }.getOrNull()
        val subscriber = runCatching { subscriberStats(room) }.getOrNull()
        val allStats = listOfNotNull(publisher, subscriber)
            .flatMap { it.statsMap.values }

        val selectedPairIds = allStats.asSequence()
            .filter { it.type == "transport" }
            .mapNotNull { it.members["selectedCandidatePairId"] as? String }
            .toSet()
        val candidatePairs = allStats.filter { it.type == "candidate-pair" }
        val selectedPairs = candidatePairs.filter { it.id in selectedPairIds }
        val activePairs = selectedPairs.ifEmpty {
            candidatePairs.filter { stat ->
                stat.members["nominated"] == true || stat.members["selected"] == true
            }
        }
        val roundTripTimeMs = activePairs.asSequence()
            .mapNotNull { (it.members["currentRoundTripTime"] as? Number)?.toDouble() }
            .filter { it.isFinite() && it >= 0.0 }
            .maxOrNull()
            ?.times(1_000)
        val receiveJitterMs = subscriber?.statsMap?.values
            ?.asSequence()
            ?.filter { it.type == "inbound-rtp" }
            ?.mapNotNull { (it.members["jitter"] as? Number)?.toDouble() }
            ?.filter { it.isFinite() && it >= 0.0 }
            ?.maxOrNull()
            ?.times(1_000)

        return NetworkMetrics(
            quality = quality,
            roundTripTimeMs = roundTripTimeMs,
            receiveJitterMs = receiveJitterMs,
        )
    }

    private suspend fun publisherStats(room: Room) = suspendCancellableCoroutine { continuation ->
        room.getPublisherRTCStats { report ->
            if (continuation.isActive) continuation.resume(report)
        }
    }

    private suspend fun subscriberStats(room: Room) = suspendCancellableCoroutine { continuation ->
        room.getSubscriberRTCStats { report ->
            if (continuation.isActive) continuation.resume(report)
        }
    }

    private fun ConnectionQuality.toStreamKitQuality() = when (this) {
        ConnectionQuality.EXCELLENT -> NetworkQuality.EXCELLENT
        ConnectionQuality.GOOD -> NetworkQuality.GOOD
        ConnectionQuality.POOR -> NetworkQuality.POOR
        ConnectionQuality.LOST -> NetworkQuality.LOST
        ConnectionQuality.UNKNOWN -> NetworkQuality.UNKNOWN
    }

    // ── Teardown ──────────────────────────────────────────────────────────────

    private suspend fun tearDown(purpose: TeardownPurpose) {
        val generation = connectionGeneration.incrementAndGet()
        val closingLifetime = roomLifetime ?: failedRoomClose?.lifetime
        val closingRoom = closingLifetime?.room ?: room
        val oldScope = connectionScope
        isConnected = false
        if (room === closingRoom) room = null
        if (roomLifetime === closingLifetime) roomLifetime = null
        // Keep the already-armed observer until actual terminal receipt. SDK
        // DISCONNECTED state and disconnect()'s early return precede completion.
        if (connectionScope === oldScope) connectionScope = null
        var failure: Throwable? = null
        var cameraFailure: Throwable? = null
        withContext(NonCancellable) {
            var allocation: CameraAllocation? = null
            cameraMutex.withLock {
                allocation = ownedCamera?.takeIf {
                    it.room === closingRoom || (closingRoom == null && room == null)
                }
                // Close first: SDK cleanup releases current publications and
                // its reconnect pool. Historical publication is not ownership.
                try {
                    val previousFailure = failedRoomClose?.takeIf { it.lifetime.room === closingRoom }
                    if (previousFailure != null && closingRoom?.state == Room.State.DISCONNECTED) throw previousFailure.error
                    closingRoom?.disconnect()
                    if (failedRoomClose?.lifetime === closingLifetime) failedRoomClose = null
                } catch (error: Throwable) {
                    if (closingLifetime != null) failedRoomClose = RoomCloseFailure(closingLifetime, error)
                    if (failure == null) failure = error else failure!!.addSuppressed(error)
                } finally {
                    if (room === closingRoom || room == null) cameraPublishBaseline = null
                }
            }
            // An observer is not a producer promise: SDK entry can fail or
            // cancel without a terminal event. Unknown capture remains owned;
            // a later actual receipt reconciles it independently.
            val retained = allocation
            cameraMutex.withLock {
                if (retained != null) {
                    try { releaseOwnedCamera(retained.track) }
                    catch (error: Throwable) { cameraFailure = error }
                    if (injectedVideoTrack === retained.track) {
                        injectedVideoTrack = null
                        injectedCapturer = null
                    }
                }
            }
            if (closingLifetime?.connectEntered != true || closingLifetime.terminal.isCompleted ||
                (retained == null && failure == null)) oldScope?.cancel()
            // Exact room detachment and generation own this notification; the
            // independent receipt observer may still be awaiting SDK completion.
            if (room == null && connectionGeneration.get() == generation) {
                onConnectionStateChanged?.invoke(ConnectionState.DISCONNECTED)
            }
        }
        failure?.let { throw it }
        cameraFailure?.let {
            // isolation boundary: reconnect admits transport, not replacement camera ownership.
            if (purpose == TeardownPurpose.RECONNECT) LKLog.e(it) { "Camera remains unavailable; reconnecting transport independently" }
            else throw it
        }
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
        private const val FILE_STREAM_TOPIC = "_streamkit.file"
        private const val FILE_TOPIC_ATTRIBUTE = "_streamkit.topic"
        private const val FILE_RESERVED_PREFIX = "_streamkit."
        private const val FILE_MAX_FIELD_BYTES = 255
        private const val FILE_MAX_ATTRIBUTE_KEY_BYTES = 128
        private const val FILE_MAX_ATTRIBUTE_VALUE_BYTES = 1_024
        private const val FILE_MAX_ATTRIBUTES = 32
        private const val FILE_MAX_ATTRIBUTES_BYTES = 8 * 1_024
    }
}
