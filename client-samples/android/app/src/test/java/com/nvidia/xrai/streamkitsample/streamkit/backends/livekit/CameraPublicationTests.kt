// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.xrai.streamkitsample.streamkit.backends.livekit

import android.content.Context
import com.nvidia.xrai.streamkitsample.streamkit.config.CameraConfig
import com.nvidia.xrai.streamkitsample.streamkit.config.CameraEncodingConfig
import com.nvidia.xrai.streamkitsample.streamkit.config.LiveKitConfig
import com.nvidia.xrai.streamkitsample.streamkit.config.SessionConfig
import com.nvidia.xrai.streamkitsample.streamkit.ConnectionState
import com.nvidia.xrai.streamkitsample.streamkit.StreamError
import io.livekit.android.events.RoomEvent
import io.livekit.android.events.DisconnectReason
import io.livekit.android.room.Room
import io.livekit.android.room.participant.LocalParticipant
import io.livekit.android.room.participant.VideoTrackPublishDefaults
import io.livekit.android.room.participant.VideoTrackPublishOptions
import io.livekit.android.room.track.LocalTrackPublication
import io.livekit.android.room.track.LocalVideoTrack
import io.livekit.android.room.track.LocalVideoTrackOptions
import io.livekit.android.room.track.Track
import io.livekit.android.room.track.CustomVideoPreset
import io.livekit.android.room.track.VideoPreset169
import io.livekit.android.room.track.VideoEncoding
import io.livekit.android.room.track.CameraPosition
import livekit.org.webrtc.RtpParameters
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test
import org.mockito.Mockito
import java.nio.ByteBuffer
import kotlin.coroutines.Continuation
import kotlin.coroutines.intrinsics.COROUTINE_SUSPENDED
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.coroutines.suspendCoroutine
import kotlinx.coroutines.CancellationException

/** Runs the real backend transaction; mocks only the SDK/native boundary.
 * Continuations force the original interleaving without sleeps or a device.
 */
class CameraPublicationTests {
    private class Fixture {
        val participant = Mockito.mock(LocalParticipant::class.java)
        val room = Mockito.mock(Room::class.java)
        val track = Mockito.mock(LocalVideoTrack::class.java)
        val deviceTrack = Mockito.mock(LocalVideoTrack::class.java)
        val publication = Mockito.mock(LocalTrackPublication::class.java)
        val backend = LiveKitBackend(LiveKitConfig(host = "test"), Mockito.mock(Context::class.java))
        var defaults = VideoTrackPublishDefaults(simulcast = false, videoCodec = "h264")
        var captureDefaults = LocalVideoTrackOptions()
        var active: LocalTrackPublication? = null
        var capturer: InjectedVideoCapturer? = null
        var onDisconnect: () -> Unit = {}
        val lifetime: Any = backend.javaClass.declaredClasses.single { it.simpleName == "RoomLifetime" }
            .getDeclaredConstructor(Room::class.java).apply { isAccessible = true }.newInstance(room)

        init {
            Mockito.`when`(room.localParticipant).thenReturn(participant)
            Mockito.`when`(room.state).thenReturn(Room.State.CONNECTED)
            Mockito.doAnswer {
                if (active?.track === track) {
                    track.dispose()
                    // SDK calls the exact private capturer's dispose callback.
                    capturer?.let { value -> Mockito.`when`(value.disposed).thenReturn(true) }
                    active = null
                }
                Mockito.`when`(room.state).thenReturn(Room.State.DISCONNECTED)
                onDisconnect()
                null
            }.`when`(room).disconnect()
            Mockito.`when`(participant.videoTrackCaptureDefaults).thenAnswer { captureDefaults }
            Mockito.doAnswer { captureDefaults = it.getArgument(0); null }
                .`when`(participant).videoTrackCaptureDefaults =
                    Mockito.any(LocalVideoTrackOptions::class.java) ?: captureDefaults
            Mockito.`when`(participant.videoTrackPublishDefaults).thenAnswer { defaults }
            Mockito.doAnswer { defaults = it.getArgument(0); null }
                .`when`(participant).videoTrackPublishDefaults =
                    Mockito.any(VideoTrackPublishDefaults::class.java) ?: defaults
            Mockito.`when`(participant.getTrackPublication(Track.Source.CAMERA)).thenAnswer { active }
            Mockito.`when`(publication.track).thenReturn(track)
            Mockito.`when`(participant.getOrCreateDefaultVideoTrack()).thenReturn(deviceTrack)
            Mockito.`when`(deviceTrack.dimensions).thenReturn(Track.Dimensions(1280, 960))
            setField("room", room)
            setField("roomLifetime", lifetime)
            setField("isConnected", true)
            setField("cameraPublishBaseline", defaults)
        }

        fun setField(name: String, value: Any?) {
            backend.javaClass.getDeclaredField(name).apply { isAccessible = true }.set(backend, value)
        }

        suspend fun terminalEvent() = event(RoomEvent.Disconnected(room, null, DisconnectReason.SERVER_SHUTDOWN))

        suspend fun event(event: RoomEvent) = suspendCoroutine<Unit> { continuation ->
            if (event is RoomEvent.Disconnected) {
                @Suppress("UNCHECKED_CAST")
                val terminal = lifetime.javaClass.getDeclaredField("terminal").apply { isAccessible = true }
                    .get(lifetime) as kotlinx.coroutines.CompletableDeferred<Unit>
                terminal.complete(Unit)
            }
            val method = backend.javaClass.getDeclaredMethod(
                "handleEvent", Room::class.java, RoomEvent::class.java, Continuation::class.java,
            ).apply { isAccessible = true }
            val result = method.invoke(backend, room, event, continuation)
            if (result !== COROUTINE_SUSPENDED) continuation.resume(Unit)
        }

        suspend fun prepareDeviceCamera() {
            Mockito.doAnswer { true }.`when`(participant).setCameraEnabled(false)
            Mockito.doAnswer {
                // The SDK caches the device track even across unpublish.
                if (active == null) {
                    val devicePublication = Mockito.mock(LocalTrackPublication::class.java)
                    Mockito.`when`(devicePublication.track).thenReturn(deviceTrack)
                    active = devicePublication
                }
                true
            }.`when`(participant).setCameraEnabled(true)
            Mockito.doAnswer {
                if (active?.track === it.getArgument<Track>(0)) active = null
                null
            }.`when`(participant).unpublishTrack(
                Mockito.any(Track::class.java) ?: track, Mockito.anyBoolean(),
            )
        }

        suspend fun prepareInjection() {
            val scope = CoroutineScope(currentCoroutineContext())
            onDisconnect = { scope.launch(start = CoroutineStart.UNDISPATCHED) { terminalEvent() }; Unit }
            // Match native arguments without constructing a WebRTC capturer.
            // Java Mockito matchers register correctly but return null. Give
            // Kotlin's non-null call boundary a value after registration.
            val capturerArgument = Mockito.mock(livekit.org.webrtc.VideoCapturer::class.java)
            val captureArgument = LocalVideoTrackOptions()
            val publishArgument = VideoTrackPublishOptions(base = defaults)
            Mockito.doAnswer {
                capturer = it.getArgument(1)
                track
            }.`when`(participant).createVideoTrack(
                Mockito.anyString(), Mockito.any(livekit.org.webrtc.VideoCapturer::class.java) ?: capturerArgument,
                Mockito.any(LocalVideoTrackOptions::class.java) ?: captureArgument, Mockito.isNull(),
            )
            Mockito.doAnswer { active = publication; true }.`when`(participant).publishVideoTrack(
                Mockito.eq(track) ?: track,
                Mockito.any(VideoTrackPublishOptions::class.java) ?: publishArgument, Mockito.isNull(),
            )
            Mockito.doAnswer { active = null; null }.`when`(participant).unpublishTrack(track)
        }

        fun policy(encoding: CameraEncodingConfig?): VideoTrackPublishDefaults =
            backend.javaClass.getDeclaredMethod("videoPublishDefaults", CameraEncodingConfig::class.java)
                .apply { isAccessible = true }.invoke(backend, encoding) as VideoTrackPublishDefaults
    }

    @Test fun injectionCannotPublishInsideSuspendedDeviceSwitch() = runBlocking {
        val f = Fixture()
        f.prepareDeviceCamera()
        f.prepareInjection()
        var stopped: Continuation<Boolean>? = null
        Mockito.doAnswer {
            @Suppress("UNCHECKED_CAST")
            stopped = it.rawArguments.last() as Continuation<Boolean>
            COROUTINE_SUSPENDED
        }.`when`(f.participant).setCameraEnabled(false)

        Mockito.mockConstruction(InjectedVideoCapturer::class.java).use { capturers ->
            val start = launch(start = CoroutineStart.UNDISPATCHED) {
                f.backend.startCamera(CameraConfig(encoding = CameraEncodingConfig.DETAIL))
            }
            fun releaseStop() {
                val pending = stopped
                stopped = null
                pending?.resume(true)
            }
            try {
                assertNotNull(stopped)
                val frame = launch(start = CoroutineStart.UNDISPATCHED) {
                    f.backend.injectVideoFrame(ByteBuffer.allocate(6), 2, 2, 0)
                }
                // First injection must wait until the device switch completes.
                assertTrue(capturers.constructed().isEmpty())
                assertFalse(frame.isCompleted)
                releaseStop()
                start.join()
                frame.join()
                assertSame(f.publication, f.active)
                assertEquals(1, capturers.constructed().size)
                Mockito.verify(f.participant, Mockito.never()).unpublishTrack(f.track)
            } finally {
                // Raw SDK continuations are not cancelled with the test's Job.
                // Assertion failure must still release the suspended child.
                releaseStop()
            }
        }
    }

    @Test fun retainedDeviceCameraRepublishesOnlyWhenPolicyChanges() = runBlocking {
        val f = Fixture()
        f.prepareDeviceCamera()
        f.backend.startCamera(CameraConfig(encoding = CameraEncodingConfig.DETAIL))
        val retained = f.active!!
        val previousTrack = retained.track!!
        assertEquals(f.policy(CameraEncodingConfig.DETAIL), f.defaults)

        f.backend.startCamera(CameraConfig(encoding = CameraEncodingConfig.DETAIL))
        assertSame(retained, f.active)
        Mockito.verify(f.participant, Mockito.never()).unpublishTrack(previousTrack)

        f.backend.startCamera(CameraConfig())
        assertNotSame(retained, f.active)
        Mockito.verify(f.participant, Mockito.times(1)).unpublishTrack(previousTrack)
        Mockito.verify(previousTrack, Mockito.never()).dispose()
        assertEquals(f.policy(null), f.defaults)
        assertEquals("h264", f.defaults.videoCodec)
        assertFalse(f.defaults.simulcast)
    }

    @Test fun firstInjectionPreemptsDeviceAndFailedPublicationDoesNotRestoreIt() = runBlocking {
        val f = Fixture()
        f.prepareInjection()
        f.prepareDeviceCamera()
        f.backend.startCamera(CameraConfig())
        val device = f.active!!.track!!
        val publishArgument = VideoTrackPublishOptions(base = f.defaults)
        Mockito.doAnswer { false }.`when`(f.participant).publishVideoTrack(
            Mockito.eq(f.track) ?: f.track,
            Mockito.any(VideoTrackPublishOptions::class.java) ?: publishArgument, Mockito.isNull(),
        )
        Mockito.mockConstruction(InjectedVideoCapturer::class.java).use {
            try {
                f.backend.injectVideoFrame(ByteBuffer.allocate(6), 2, 2, 0)
                fail("A rejected publication must fail the first injection")
            } catch (_: IllegalStateException) {
                assertNull(f.active)
                Mockito.verify(f.participant).unpublishTrack(device)
                Mockito.verify(f.track).stop()
                Mockito.verify(f.track).dispose()
                Mockito.verify(f.track).stopCapture()
            }
        }
    }

    @Test fun initialPushOwnsCapturerUntilDeliveryCompletes() = runBlocking {
        val f = Fixture()
        f.prepareInjection()
        var stop: Job? = null
        val frameArgument = ByteBuffer.allocate(0)
        Mockito.mockConstruction(InjectedVideoCapturer::class.java) { capturer, _ ->
            Mockito.doAnswer {
                stop = launch(start = CoroutineStart.UNDISPATCHED) { f.backend.stopCamera() }
                assertFalse(stop!!.isCompleted)
                assertSame(f.publication, f.active)
                null
            }.`when`(capturer).pushI420Frame(Mockito.any(ByteBuffer::class.java) ?: frameArgument, Mockito.anyInt(),
                                           Mockito.anyInt(), Mockito.anyLong())
        }.use {
            f.backend.injectVideoFrame(ByteBuffer.allocate(6), 2, 2, 0)
            stop!!.join()
            assertNull(f.active)
            Mockito.verify(f.participant).unpublishTrack(f.track)
            Mockito.verify(f.track).dispose()
        }
    }

    @Test fun policyOverlayPreservesBaselineAndUnownedDefaults() {
        val f = Fixture()
        val baseline = f.defaults
        val detail = f.policy(CameraEncodingConfig.DETAIL)
        assertFalse(detail.simulcast)
        assertEquals("h264", detail.videoCodec)
        f.defaults = detail.copy(videoCodec = "vp9", scalabilityMode = "L3T3_KEY")
        val reset = f.policy(null)
        assertEquals(baseline.videoEncoding, reset.videoEncoding)
        assertEquals(baseline.simulcast, reset.simulcast)
        assertEquals(baseline.degradationPreference, reset.degradationPreference)
        assertEquals("vp9", reset.videoCodec)
        assertEquals("L3T3_KEY", reset.scalabilityMode)
        assertTrue(f.policy(CameraEncodingConfig(simulcast = true)).simulcast)
        assertFalse(f.policy(CameraEncodingConfig(simulcast = false)).simulcast)
        assertFalse(f.policy(CameraEncodingConfig.MOTION).simulcast)
    }

    @Test fun SDKComputedEncodingsKeepTheirGeometryAndCoherentLimits() {
        val f = Fixture()
        val policy = CameraEncodingConfig(maxBitrateBps = 50_000)
        val defaults = VideoTrackPublishDefaults(videoEncoding = VideoEncoding(50_000, 30), simulcast = true)
        val bounded = cameraLayerDefaults(policy, defaults, 1280, 720)
        val compute = LocalParticipant::class.java.getDeclaredMethod(
            "computeVideoEncodings", Boolean::class.javaPrimitiveType, Track.Dimensions::class.java,
            VideoTrackPublishOptions::class.java,
        ).apply { isAccessible = true }
        @Suppress("UNCHECKED_CAST")
        fun actual(options: VideoTrackPublishDefaults) = compute.invoke(
            f.participant, false, Track.Dimensions(1280, 720), VideoTrackPublishOptions(base = options),
        ) as List<RtpParameters.Encoding>
        val before = actual(defaults)
        val after = actual(bounded)
        assertTrue(before.any { it.maxBitrateBps!! > 50_000 })
        assertTrue(after.all { it.maxBitrateBps!! <= 50_000 })
        assertEquals(before.map { it.rid to it.scaleResolutionDownBy }, after.map { it.rid to it.scaleResolutionDownBy })
        assertEquals(before.map { it.maxFramerate }, after.map { it.maxFramerate })
        assertNull(defaults.simulcastLayers)
        val custom = listOf(CustomVideoPreset(VideoPreset169.H180.capture, VideoEncoding(20_000, 12)), VideoPreset169.H360)
        val customDefaults = defaults.copy(simulcastLayers = custom)
        val limited = cameraLayerDefaults(policy, customDefaults, 1280, 720)
        assertEquals(listOf(2_222, 50_000), limited.simulcastLayers!!.map { it.encoding.maxBitrate })
        assertEquals(custom.map { it.capture }, limited.simulcastLayers!!.map { it.capture })
        assertEquals(custom.map { it.encoding.maxFps }, limited.simulcastLayers!!.map { it.encoding.maxFps })
        assertSame(defaults, cameraLayerDefaults(null, defaults, 0, 0))
        val disabled = defaults.copy(simulcast = false)
        assertSame(disabled, cameraLayerDefaults(policy, disabled, 0, 0))
        assertThrows(IllegalArgumentException::class.java) {
            cameraLayerDefaults(CameraEncodingConfig(maxBitrateBps = 1), defaults, 1280, 720)
        }
        val large = defaults.copy(simulcastLayers = listOf(
            CustomVideoPreset(VideoPreset169.H180.capture, VideoEncoding(Int.MAX_VALUE - 1, 20)),
            CustomVideoPreset(VideoPreset169.H360.capture, VideoEncoding(Int.MAX_VALUE, 20)),
        ))
        assertEquals(listOf(49_999, 50_000), cameraLayerDefaults(policy, large, 1280, 720)
            .simulcastLayers!!.map { it.encoding.maxBitrate })
    }

    @Test fun devicePolicyUsesActualSDKDimensionsAndExistingCaptureSelection() = runBlocking {
        val f = Fixture()
        f.prepareDeviceCamera()
        f.defaults = f.defaults.copy(simulcast = true)
        f.setField("cameraPublishBaseline", f.defaults)
        f.backend.startCamera(CameraConfig(deviceId = "pinned", facing = CameraConfig.CameraFacing.BACK,
                                          encoding = CameraEncodingConfig(maxBitrateBps = 50_000)))
        assertSame(f.deviceTrack, f.active!!.track)
        assertEquals("pinned", f.captureDefaults.deviceId)
        assertEquals(CameraPosition.BACK, f.captureDefaults.position)
        assertTrue(f.defaults.simulcastLayers!!.all { it.encoding.maxBitrate <= 50_000 })
        assertEquals(listOf(240, 480), f.defaults.simulcastLayers!!.map { it.capture.width })
        f.backend.startCamera(CameraConfig(encoding = CameraEncodingConfig(maxBitrateBps = 3_000_000)))
        assertTrue(f.defaults.simulcastLayers!!.any { it.encoding.maxBitrate > 50_000 })
        f.backend.startCamera(CameraConfig())
        assertNull(f.defaults.simulcastLayers)
        Mockito.verify(f.deviceTrack, Mockito.never()).dispose()
    }

    @Test fun cancelledFirstPublicationCleansItsOwnedAllocationBeforeReleasingCamera() = runBlocking {
        val f = Fixture()
        f.prepareInjection()
        val options = VideoTrackPublishOptions(base = f.defaults)
        var pending: Continuation<Boolean>? = null
        Mockito.doAnswer {
            @Suppress("UNCHECKED_CAST")
            pending = it.rawArguments.last() as Continuation<Boolean>
            COROUTINE_SUSPENDED
        }.`when`(f.participant).publishVideoTrack(Mockito.eq(f.track) ?: f.track,
                                                Mockito.any(VideoTrackPublishOptions::class.java) ?: options, Mockito.isNull())
        Mockito.mockConstruction(InjectedVideoCapturer::class.java).use {
            val frame = launch(start = CoroutineStart.UNDISPATCHED) {
                f.backend.injectVideoFrame(ByteBuffer.allocate(6), 2, 2, 0)
            }
            try {
                assertNotNull(pending)
                frame.cancel()
                val stop = launch(start = CoroutineStart.UNDISPATCHED) { f.backend.stopCamera() }
                assertFalse(stop.isCompleted)
                val completed = pending!!
                pending = null
                completed.resumeWithException(CancellationException())
                frame.join()
                stop.join()
                Mockito.verify(f.track).stop()
                Mockito.verify(f.track).dispose()
                Mockito.verify(f.track).stopCapture()
                assertNull(f.active)
            } finally {
                val unfinished = pending
                pending = null
                unfinished?.resumeWithException(CancellationException())
            }
        }
    }

    @Test fun cachedDeviceSurvivesPolicyResetAndPublicationRejectionIsReported() = runBlocking {
        val f = Fixture()
        f.prepareDeviceCamera()
        f.defaults = f.defaults.copy(simulcast = true)
        f.setField("cameraPublishBaseline", f.defaults)
        f.backend.startCamera(CameraConfig(encoding = CameraEncodingConfig(maxBitrateBps = 50_000)))
        f.backend.startCamera(CameraConfig())
        assertSame(f.deviceTrack, f.active!!.track)
        f.backend.stopCamera()
        Mockito.verify(f.deviceTrack, Mockito.never()).dispose()
        Mockito.doReturn(false).`when`(f.participant).setCameraEnabled(true)
        try { f.backend.startCamera(CameraConfig()); fail("Rejected SDK publication must be reported") }
        catch (_: IllegalStateException) {}
    }

    @Test fun failedFirstDeliveryDisposesAlreadyPublishedOwnedTrack() = runBlocking {
        val f = Fixture()
        f.prepareInjection()
        val frameArgument = ByteBuffer.allocate(0)
        Mockito.mockConstruction(InjectedVideoCapturer::class.java) { capturer, _ ->
            Mockito.doThrow(IllegalStateException("delivery failed")).`when`(capturer).pushI420Frame(
                Mockito.any(ByteBuffer::class.java) ?: frameArgument,
                Mockito.anyInt(), Mockito.anyInt(), Mockito.anyLong(),
            )
        }.use {
            try {
                f.backend.injectVideoFrame(ByteBuffer.allocate(6), 2, 2, 0)
                fail("First delivery must report its error")
            } catch (_: IllegalStateException) {
                assertNull(f.active)
                Mockito.verify(f.participant).unpublishTrack(f.track)
                Mockito.verify(f.track, Mockito.times(1)).dispose()
            }
        }
    }

    @Test fun partialDisposalRetainsFailureAndRejectsUnsafeRetry() = runBlocking {
        val f = Fixture()
        f.prepareInjection()
        Mockito.mockConstruction(InjectedVideoCapturer::class.java).use {
            f.backend.injectVideoFrame(ByteBuffer.allocate(6), 2, 2, 0)
            Mockito.doThrow(IllegalStateException("dispose failed")).doNothing().`when`(f.track).dispose()
            try {
                f.backend.stopCamera()
                fail("Cleanup failure must remain visible")
            } catch (_: IllegalStateException) {
                assertNull(f.active)
            }
            try { f.backend.stopCamera(); fail("Partial native cleanup cannot be blindly retried") }
            catch (_: IllegalStateException) {}
            Mockito.verify(f.participant, Mockito.times(1)).unpublishTrack(f.track)
            Mockito.verify(f.track, Mockito.times(1)).dispose()
            try { f.backend.startCamera(CameraConfig()); fail("Replacement must not discard failed owner") }
            catch (_: IllegalStateException) {}
        }
    }

    @Test fun terminalDisconnectDoesNotDisposeSDKReleasedCameraInNextSession() = runBlocking {
        val old = Fixture()
        old.prepareInjection()
        Mockito.mockConstruction(InjectedVideoCapturer::class.java) { capturer, _ ->
            Mockito.`when`(capturer.disposed).thenCallRealMethod()
            Mockito.doCallRealMethod().`when`(capturer).dispose()
        }.use { capturers ->
            old.backend.injectVideoFrame(ByteBuffer.allocate(6), 2, 2, 0)
            // Model the pinned SDK boundary: cleanup removes and disposes the
            // publication BEFORE delivering the terminal room event.
            old.participant.unpublishTrack(old.track)
            old.track.dispose()
            capturers.constructed().first().dispose()
            Mockito.doThrow(IllegalStateException("already disposed")).`when`(old.track).dispose()
            old.terminalEvent()
            old.backend.stopCamera()
            try {
                old.backend.connect(SessionConfig.DEFAULT)
                fail("This fixture intentionally has no connection token")
            } catch (_: StreamError.MissingToken) { /* Teardown reached credential validation. */ }
            val next = Fixture()
            next.prepareInjection()
            old.setField("room", next.room)
            old.setField("roomLifetime", next.lifetime)
            old.setField("isConnected", true)
            old.setField("cameraPublishBaseline", next.defaults)
            old.backend.injectVideoFrame(ByteBuffer.allocate(6), 2, 2, 1)
            assertSame(next.publication, next.active)
            Mockito.verify(old.track, Mockito.times(1)).dispose()
            old.backend.stopCamera()
            Mockito.verify(next.track, Mockito.times(1)).dispose()
        }
    }

    @Test fun disconnectClosesRoomAndReportsFailedUnpublishedCleanup() = runBlocking {
        val f = Fixture()
        f.prepareInjection()
        val states = mutableListOf<ConnectionState>()
        f.backend.onConnectionStateChanged = { states.add(it) }
        Mockito.mockConstruction(InjectedVideoCapturer::class.java).use {
            f.backend.injectVideoFrame(ByteBuffer.allocate(6), 2, 2, 0)
            Mockito.doThrow(IllegalStateException("dispose failed")).doThrow(IllegalStateException("retry failed"))
                .doNothing().`when`(f.track).dispose()
            try { f.backend.stopCamera(); fail("Expected failed explicit release") }
            catch (_: IllegalStateException) {}
            try { f.backend.disconnect(); fail("Cleanup error must remain visible") }
            catch (_: IllegalStateException) {}
            Mockito.verify(f.room).disconnect()
            assertEquals(listOf(ConnectionState.DISCONNECTED), states)
            try { f.backend.injectVideoFrame(ByteBuffer.allocate(6), 2, 2, 1); fail("Closed admission") }
            catch (_: StreamError.NotConnected) {}
            try { f.backend.stopCamera(); fail("Retained partial native failure remains visible") }
            catch (_: IllegalStateException) {}
            Mockito.verify(f.track, Mockito.times(1)).dispose()
            Mockito.verify(f.participant, Mockito.times(1)).unpublishTrack(f.track)
            try { f.backend.connect(SessionConfig.DEFAULT); fail("Fixture has no token") }
            catch (_: StreamError.MissingToken) { /* Camera quarantine must not block transport setup. */ }
            Mockito.verify(f.track, Mockito.times(1)).dispose()
        }
    }

    @Test fun terminalDisconnectDoesNotForgetUnpublishedAllocation() = runBlocking {
        val f = Fixture()
        f.prepareInjection()
        val options = VideoTrackPublishOptions(base = f.defaults)
        Mockito.doAnswer { false }.`when`(f.participant).publishVideoTrack(
            Mockito.eq(f.track) ?: f.track,
            Mockito.any(VideoTrackPublishOptions::class.java) ?: options, Mockito.isNull(),
        )
        Mockito.doThrow(IllegalStateException("dispose failed")).doNothing().`when`(f.track).dispose()
        Mockito.mockConstruction(InjectedVideoCapturer::class.java).use {
            try { f.backend.injectVideoFrame(ByteBuffer.allocate(6), 2, 2, 0); fail("Publication rejected") }
            catch (_: IllegalStateException) {}
            // SDK cleanup cannot see this allocation: it was never published.
            f.terminalEvent()
            Mockito.verify(f.track, Mockito.times(1)).dispose()
            Mockito.verify(f.participant, Mockito.never()).unpublishTrack(f.track)
            try { f.backend.stopCamera(); fail("Retained partial native failure remains visible") }
            catch (_: IllegalStateException) {}
            Mockito.verify(f.track, Mockito.times(1)).dispose()
        }
    }

    @Test fun injectionUnpublishesButNeverDisposesSDKCachedDevice() = runBlocking {
        val f = Fixture()
        f.prepareDeviceCamera()
        f.prepareInjection()
        f.defaults = f.defaults.copy(simulcast = true)
        f.setField("cameraPublishBaseline", f.defaults)
        val captureArgument = LocalVideoTrackOptions()
        f.backend.startCamera(CameraConfig(encoding = CameraEncodingConfig(maxBitrateBps = 50_000)))
        val next = Mockito.mock(LocalVideoTrack::class.java)
        val nextPublication = Mockito.mock(LocalTrackPublication::class.java)
        Mockito.`when`(nextPublication.track).thenReturn(next)
        val capturerArgument = Mockito.mock(livekit.org.webrtc.VideoCapturer::class.java)
        Mockito.doAnswer {
            Mockito.verify(f.participant).unpublishTrack(f.deviceTrack)
            Mockito.verify(f.deviceTrack, Mockito.never()).dispose()
            next
        }.`when`(f.participant).createVideoTrack(
            Mockito.anyString(), Mockito.any(livekit.org.webrtc.VideoCapturer::class.java) ?: capturerArgument,
            Mockito.any(LocalVideoTrackOptions::class.java) ?: captureArgument, Mockito.isNull(),
        )
        val publishArgument = VideoTrackPublishOptions(base = f.defaults)
        Mockito.doAnswer { f.active = nextPublication; true }.`when`(f.participant).publishVideoTrack(
            Mockito.eq(next) ?: next, Mockito.any(VideoTrackPublishOptions::class.java) ?: publishArgument, Mockito.isNull(),
        )
        Mockito.mockConstruction(InjectedVideoCapturer::class.java).use {
            f.backend.injectVideoFrame(ByteBuffer.allocate(6), 2, 2, 1)
            assertSame(nextPublication, f.active)
            Mockito.verify(f.participant, Mockito.times(1)).unpublishTrack(f.deviceTrack)
            Mockito.verify(next, Mockito.never()).dispose()
        }
    }

    @Test fun serverUnpublishThenDisconnectStillDisposesExplicitAllocation() = runBlocking {
        val f = Fixture()
        f.prepareInjection()
        Mockito.mockConstruction(InjectedVideoCapturer::class.java).use {
            f.backend.injectVideoFrame(ByteBuffer.allocate(6), 2, 2, 0)
            // Server unpublish removes the SDK registry entry and stops, but
            // does not dispose this explicitly allocated native stack.
            f.participant.unpublishTrack(f.track)
            assertNull(f.active)
            f.event(RoomEvent.TrackUnpublished(f.room, f.publication, f.participant))
            f.backend.disconnect()
            Mockito.verify(f.room).disconnect()
            Mockito.verify(f.track, Mockito.times(1)).dispose()
            Mockito.verify(f.participant, Mockito.times(1)).unpublishTrack(f.track)
        }
    }

    @Test fun reconnectKeepsSDKRepublishCustodyUntilNormalStopCanAcknowledge() = runBlocking {
        val f = Fixture()
        f.prepareInjection()
        Mockito.mockConstruction(InjectedVideoCapturer::class.java).use {
            f.backend.injectVideoFrame(ByteBuffer.allocate(6), 2, 2, 0)
            // Exact SDK full-reconnect boundary: registry empty while the
            // same track is retained privately for republishing.
            f.active = null
            f.setField("isConnected", false)
            try { f.backend.stopCamera(); fail("SDK retains the track for republish") }
            catch (_: StreamError.NotConnected) {}
            Mockito.verify(f.track, Mockito.never()).dispose()
            f.active = f.publication
            f.setField("isConnected", true)
            f.backend.stopCamera()
            Mockito.verify(f.participant).unpublishTrack(f.track)
            Mockito.verify(f.track).dispose()
        }
    }

    @Test fun disconnectedStateDoesNotAuthorizeCleanupBeforeTerminalReceipt() = runBlocking {
        val f = Fixture()
        f.prepareInjection()
        Mockito.mockConstruction(InjectedVideoCapturer::class.java).use {
            f.backend.injectVideoFrame(ByteBuffer.allocate(6), 2, 2, 0)
            f.active = null // failed SDK republish dropped the private pool
            Mockito.`when`(f.room.state).thenReturn(Room.State.DISCONNECTED)
            f.onDisconnect = {} // SDK disconnect returns early, cleanup still in flight
            val closed = launch(start = CoroutineStart.UNDISPATCHED) { f.backend.disconnect() }
            assertFalse(closed.isCompleted)
            Mockito.verify(f.track, Mockito.never()).dispose()
            f.terminalEvent()
            closed.join()
            Mockito.verify(f.track, Mockito.times(1)).dispose()
            f.backend.disconnect()
            Mockito.verify(f.track, Mockito.times(1)).dispose()
        }
    }

    @Test fun transportCloseFailureDetachesAdmissionButRetainsItsOwnFailure() = runBlocking {
        val f = Fixture()
        val states = mutableListOf<ConnectionState>()
        f.backend.onConnectionStateChanged = { states.add(it) }
        Mockito.doThrow(IllegalStateException("transport close failed")).`when`(f.room).disconnect()
        try { f.backend.disconnect(); fail("Transport close failure must remain visible") }
        catch (_: IllegalStateException) {}
        assertEquals(listOf(ConnectionState.DISCONNECTED), states)
        try { f.backend.injectVideoFrame(ByteBuffer.allocate(6), 2, 2, 0); fail("Admission must be closed") }
        catch (_: StreamError.NotConnected) {}
        try { f.backend.connect(SessionConfig.DEFAULT); fail("Transport fault cannot be hidden as camera quarantine") }
        catch (_: IllegalStateException) {}
        Mockito.verify(f.room, Mockito.times(2)).disconnect()
    }
}
