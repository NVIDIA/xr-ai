// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.xrai.streamkitsample.streamkit.backends.livekit

import android.content.Context
import com.nvidia.xrai.streamkitsample.streamkit.config.CameraConfig
import com.nvidia.xrai.streamkitsample.streamkit.config.CameraEncodingConfig
import com.nvidia.xrai.streamkitsample.streamkit.config.LiveKitConfig
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
import kotlinx.coroutines.CancellationException

/** Runs the real backend transaction; mocks only the SDK/native boundary.
 * Continuations force the original interleaving without sleeps or a device.
 */
class CameraPublicationTests {
    private class Fixture {
        val participant = Mockito.mock(LocalParticipant::class.java)
        val room = Mockito.mock(Room::class.java)
        val track = Mockito.mock(LocalVideoTrack::class.java)
        val publication = Mockito.mock(LocalTrackPublication::class.java)
        val backend = LiveKitBackend(LiveKitConfig(host = "test"), Mockito.mock(Context::class.java))
        var defaults = VideoTrackPublishDefaults(simulcast = false, videoCodec = "h264")
        var captureDefaults = LocalVideoTrackOptions()
        var active: LocalTrackPublication? = null

        init {
            Mockito.`when`(room.localParticipant).thenReturn(participant)
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
            setField("room", room)
            setField("isConnected", true)
            setField("cameraPublishBaseline", defaults)
        }

        fun setField(name: String, value: Any) {
            backend.javaClass.getDeclaredField(name).apply { isAccessible = true }.set(backend, value)
        }

        suspend fun prepareDeviceCamera() {
            Mockito.doAnswer { true }.`when`(participant).setCameraEnabled(false)
            Mockito.doAnswer {
                // A retained publication is unmuted, not replaced. A removed
                // publication gets a new track identity on the next enable.
                if (active == null) {
                    val deviceTrack = Mockito.mock(LocalVideoTrack::class.java)
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
            // Match native arguments without constructing a WebRTC capturer.
            // Java Mockito matchers register correctly but return null. Give
            // Kotlin's non-null call boundary a value after registration.
            val capturerArgument = Mockito.mock(livekit.org.webrtc.VideoCapturer::class.java)
            val captureArgument = LocalVideoTrackOptions()
            val publishArgument = VideoTrackPublishOptions(base = defaults)
            Mockito.doReturn(track).`when`(participant).createVideoTrack(
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
        f.prepareInjection()
        f.defaults = f.defaults.copy(simulcast = true)
        f.setField("cameraPublishBaseline", f.defaults)
        Mockito.`when`(f.track.dimensions).thenReturn(Track.Dimensions(1280, 960))
        val captureArgument = LocalVideoTrackOptions()
        Mockito.doReturn(f.track).`when`(f.participant).createVideoTrack(
            Mockito.anyString(), Mockito.any(LocalVideoTrackOptions::class.java) ?: captureArgument, Mockito.isNull(),
        )
        f.backend.startCamera(CameraConfig(deviceId = "pinned", facing = CameraConfig.CameraFacing.BACK,
                                          encoding = CameraEncodingConfig(maxBitrateBps = 50_000)))
        assertSame(f.publication, f.active)
        assertEquals("pinned", f.captureDefaults.deviceId)
        assertEquals(CameraPosition.BACK, f.captureDefaults.position)
        val options = org.mockito.ArgumentCaptor.forClass(VideoTrackPublishOptions::class.java)
        Mockito.verify(f.participant).publishVideoTrack(Mockito.eq(f.track) ?: f.track,
                                                       options.capture() ?: VideoTrackPublishOptions(base = f.defaults), Mockito.isNull())
        assertTrue(options.value.simulcastLayers!!.all { it.encoding.maxBitrate <= 50_000 })
        assertEquals(listOf(240, 480), options.value.simulcastLayers!!.map { it.capture.width })
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

    @Test fun explicitlyCreatedDeviceIsDisposedWhenReturningToSDKCamera() = runBlocking {
        val f = Fixture()
        f.prepareDeviceCamera()
        f.prepareInjection()
        f.defaults = f.defaults.copy(simulcast = true)
        f.setField("cameraPublishBaseline", f.defaults)
        Mockito.`when`(f.track.dimensions).thenReturn(Track.Dimensions(1280, 720))
        val captureArgument = LocalVideoTrackOptions()
        Mockito.doReturn(f.track).`when`(f.participant).createVideoTrack(
            Mockito.anyString(), Mockito.any(LocalVideoTrackOptions::class.java) ?: captureArgument, Mockito.isNull(),
        )
        f.backend.startCamera(CameraConfig(encoding = CameraEncodingConfig(maxBitrateBps = 50_000)))
        Mockito.verify(f.track, Mockito.never()).dispose()
        f.backend.startCamera(CameraConfig())
        Mockito.verify(f.participant, Mockito.times(1)).unpublishTrack(f.track)
        Mockito.verify(f.track, Mockito.times(1)).dispose()
        assertNotSame(f.track, f.active!!.track)
        f.backend.stopCamera()
        Mockito.verify(f.active!!.track!!, Mockito.never()).dispose()
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

    @Test fun failedDisposalRemainsOwnedUntilStopRetrySucceeds() = runBlocking {
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
            f.backend.stopCamera()
            Mockito.verify(f.participant, Mockito.times(1)).unpublishTrack(f.track)
            Mockito.verify(f.track, Mockito.times(2)).dispose()
            f.backend.stopCamera()
            Mockito.verify(f.track, Mockito.times(2)).dispose()
        }
    }
}
