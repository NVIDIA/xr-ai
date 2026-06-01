// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.xrai.streamkitsample.streamkit.backends.livekit

import android.content.Context
import livekit.org.webrtc.CapturerObserver
import livekit.org.webrtc.JavaI420Buffer
import livekit.org.webrtc.SurfaceTextureHelper
import livekit.org.webrtc.VideoCapturer
import livekit.org.webrtc.VideoFrame
import livekit.org.webrtc.YuvHelper
import java.nio.ByteBuffer

/**
 * A [VideoCapturer] that emits frames fed in externally via [pushI420Frame].
 *
 * LiveKit Android (unlike the iOS SDK) has no built-in `BufferCapturer`, so an
 * external video source (a file, a synthetic generator, an external camera
 * adapter) has to be plumbed through a custom [VideoCapturer].
 */
internal class InjectedVideoCapturer : VideoCapturer {

    private var observer: CapturerObserver? = null
    @Volatile private var started: Boolean = false

    override fun initialize(
        surfaceTextureHelper: SurfaceTextureHelper?,
        applicationContext: Context?,
        capturerObserver: CapturerObserver?,
    ) {
        observer = capturerObserver
    }

    override fun startCapture(width: Int, height: Int, framerate: Int) {
        started = true
        observer?.onCapturerStarted(true)
    }

    override fun stopCapture() {
        started = false
        observer?.onCapturerStopped()
    }

    override fun changeCaptureFormat(width: Int, height: Int, framerate: Int) {
        // Format is driven by injected frame dimensions, not by LiveKit.
    }

    override fun dispose() {
        started = false
        observer = null
    }

    override fun isScreencast(): Boolean = false

    /**
     * Pushes a raw I420 frame downstream. Safe to call from any thread.
     *
     * The caller's frame buffer must remain valid until this call returns —
     * planes are copied into a WebRTC-owned I420 buffer because the encoder
     * may retain frames past the [CapturerObserver.onFrameCaptured] return.
     *
     * @param i420 Read-only buffer containing Y[w*h], U[w/2*h/2], V[w/2*h/2].
     * @param width Even pixel width.
     * @param height Even pixel height.
     * @param timestampUs Source-side presentation timestamp, microseconds.
     */
    fun pushI420Frame(i420: ByteBuffer, width: Int, height: Int, timestampUs: Long) {
        val obs = observer ?: return
        if (!started) return
        if (width <= 0 || height <= 0 || (width and 1) == 1 || (height and 1) == 1) return

        val halfWidth = width / 2
        val halfHeight = height / 2
        val ySize = width * height
        val uvSize = halfWidth * halfHeight
        val totalSize = ySize + 2 * uvSize
        if (i420.remaining() < totalSize) return

        val buffer = JavaI420Buffer.allocate(width, height)
        val origin = i420.position()
        try {
            // Native plane copy — handles stride mismatches without per-row JNI hops.
            YuvHelper.copyPlane(slice(i420, origin, ySize), width, buffer.dataY, buffer.strideY, width, height)
            YuvHelper.copyPlane(slice(i420, origin + ySize, uvSize), halfWidth, buffer.dataU, buffer.strideU, halfWidth, halfHeight)
            YuvHelper.copyPlane(slice(i420, origin + ySize + uvSize, uvSize), halfWidth, buffer.dataV, buffer.strideV, halfWidth, halfHeight)
        } catch (t: Throwable) {
            buffer.release()
            return
        }

        // Use a locally-monotonic clock instead of the upstream
        // presentation timestamp. Some external sources tie
        // `presentationTimeUs` to a device-side camera clock that restarts at
        // 0 each time a stream is opened — feeding that to WebRTC across
        // stream restarts produces backwards-jumping timestamps, which the
        // encoder responds to by emitting frames the decoder renders as a
        // flat green field on the receiver. System.nanoTime() is guaranteed
        // monotonic within the process and is the recommended clock for
        // WebRTC capturers.
        val timestampNs = System.nanoTime()
        val frame = VideoFrame(buffer, /* rotation = */ 0, timestampNs)
        try {
            obs.onFrameCaptured(frame)
        } finally {
            frame.release()
        }
    }

    /**
     * Returns a [ByteBuffer] view of [src] starting at [offset] and spanning
     * [length] bytes. Must use [ByteBuffer.slice] (not [ByteBuffer.duplicate])
     * because [YuvHelper.copyPlane]'s native side resolves the source address
     * with `GetDirectBufferAddress`, which returns the **base** address of
     * the underlying memory and ignores `.position()`. With a `duplicate`,
     * the base is unchanged, so the U and V plane copies read the start of
     * the Y plane instead of the chroma planes — luma is correct, chroma
     * is garbled, and the encoded output renders on the receiver as a
     * green/purple tint. `slice()` produces a buffer whose backing memory
     * pointer is offset, so `GetDirectBufferAddress` returns the right
     * plane base.
     */
    private fun slice(src: ByteBuffer, offset: Int, length: Int): ByteBuffer =
        src.duplicate().apply {
            position(offset)
            limit(offset + length)
        }.slice()
}
