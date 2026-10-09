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
        var active: LocalTrackPublication? = null

        init {
            Mockito.`when`(room.localParticipant).thenReturn(participant)
            Mockito.`when`(participant.videoTrackCaptureDefaults).thenReturn(LocalVideoTrackOptions())
            Mockito.`when`(participant.videoTrackPublishDefaults).thenAnswer { defaults }
            Mockito.doAnswer { defaults = it.getArgument(0); null }
                .`when`(participant).videoTrackPublishDefaults = defaults
            Mockito.`when`(participant.getTrackPublication(Track.Source.CAMERA)).thenAnswer { active }
            Mockito.`when`(publication.track).thenReturn(track)
            setField("room", room)
            setField("isConnected", true)
            setField("cameraPublishBaseline", defaults)
        }

        fun setField(name: String, value: Any) {
            backend.javaClass.getDeclaredField(name).apply { isAccessible = true }.set(backend, value)
        }

        suspend fun prepareInjection() {
            // Match native arguments without constructing a WebRTC capturer.
            Mockito.doReturn(track).`when`(participant).createVideoTrack(
                Mockito.anyString(), Mockito.any(livekit.org.webrtc.VideoCapturer::class.java),
                Mockito.any(LocalVideoTrackOptions::class.java), Mockito.isNull(),
            )
            Mockito.doAnswer { active = publication; true }.`when`(participant).publishVideoTrack(
                Mockito.eq(track), Mockito.any(VideoTrackPublishOptions::class.java), Mockito.isNull(),
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
            assertNotNull(stopped)
            val frame = launch(start = CoroutineStart.UNDISPATCHED) {
                f.backend.injectVideoFrame(ByteBuffer.allocate(6), 2, 2, 0)
            }
            // On the former code this frame published during stop, then the
            // resumed start re-looked up CAMERA and disposed the injected track.
            assertTrue(capturers.constructed().isEmpty())
            assertFalse(frame.isCompleted)
            stopped!!.resume(true)
            start.join()
            frame.join()
            assertSame(f.publication, f.active)
            assertEquals(1, capturers.constructed().size)
            Mockito.verify(f.participant, Mockito.never()).unpublishTrack(f.track)
        }
    }

    @Test fun initialPushOwnsCapturerUntilDeliveryCompletes() = runBlocking {
        val f = Fixture()
        f.prepareInjection()
        var stop: Job? = null
        Mockito.mockConstruction(InjectedVideoCapturer::class.java) { capturer, _ ->
            Mockito.doAnswer {
                stop = launch(start = CoroutineStart.UNDISPATCHED) { f.backend.stopCamera() }
                assertFalse(stop!!.isCompleted)
                assertSame(f.publication, f.active)
                null
            }.`when`(capturer).pushI420Frame(Mockito.any(ByteBuffer::class.java), Mockito.anyInt(),
                                           Mockito.anyInt(), Mockito.anyLong())
        }.use {
            f.backend.injectVideoFrame(ByteBuffer.allocate(6), 2, 2, 0)
            stop!!.join()
            assertNull(f.active)
            Mockito.verify(f.participant).unpublishTrack(f.track)
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
}
