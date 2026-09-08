// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.xrai.streamkitsample.streamkit

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import io.livekit.android.room.track.LocalVideoTrack
import kotlinx.coroutines.suspendCancellableCoroutine
import livekit.org.webrtc.VideoFrame
import livekit.org.webrtc.VideoSink
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

internal fun encodeI420Jpeg(i420: ByteBuffer, width: Int, height: Int): ByteArray {
    val ySize = width * height
    val chromaWidth = width / 2
    val chromaHeight = height / 2
    val chromaSize = chromaWidth * chromaHeight
    val source = i420.duplicate()
    val nv21 = ByteArray(ySize + 2 * chromaSize)
    source.get(nv21, 0, ySize)
    val uOffset = source.position()
    val vOffset = uOffset + chromaSize
    var output = ySize
    repeat(chromaSize) { index ->
        nv21[output++] = source.get(vOffset + index)
        nv21[output++] = source.get(uOffset + index)
    }
    return ByteArrayOutputStream().use { encoded ->
        check(YuvImage(nv21, ImageFormat.NV21, width, height, null).compressToJpeg(
            Rect(0, 0, width, height),
            90,
            encoded,
        )) { "JPEG encoding failed" }
        encoded.toByteArray()
    }
}

private fun encodeFrameJpeg(frame: VideoFrame): ByteArray {
    val buffer = frame.buffer.toI420()
        ?: throw IllegalStateException("Captured video frame could not be converted to I420")
    try {
        val width = buffer.width
        val height = buffer.height
        val contiguous = ByteBuffer.allocate(width * height * 3 / 2)
        for (row in 0 until height) {
            for (column in 0 until width) {
                contiguous.put(buffer.dataY.get(buffer.dataY.position() + row * buffer.strideY + column))
            }
        }
        val chromaWidth = width / 2
        val chromaHeight = height / 2
        for (row in 0 until chromaHeight) {
            for (column in 0 until chromaWidth) {
                contiguous.put(buffer.dataU.get(buffer.dataU.position() + row * buffer.strideU + column))
            }
        }
        for (row in 0 until chromaHeight) {
            for (column in 0 until chromaWidth) {
                contiguous.put(buffer.dataV.get(buffer.dataV.position() + row * buffer.strideV + column))
            }
        }
        contiguous.flip()
        return rotateJpeg(encodeI420Jpeg(contiguous, width, height), frame.rotation)
    } finally {
        buffer.release()
    }
}

private fun rotateJpeg(data: ByteArray, rotation: Int): ByteArray {
    val normalized = ((rotation % 360) + 360) % 360
    if (normalized == 0) return data
    val source = BitmapFactory.decodeByteArray(data, 0, data.size)
        ?: throw IllegalStateException("Captured JPEG could not be decoded")
    val rotated = Bitmap.createBitmap(
        source,
        0,
        0,
        source.width,
        source.height,
        Matrix().apply { postRotate(normalized.toFloat()) },
        true,
    )
    return try {
        ByteArrayOutputStream().use { encoded ->
            check(rotated.compress(Bitmap.CompressFormat.JPEG, 90, encoded)) {
                "Rotated JPEG encoding failed"
            }
            encoded.toByteArray()
        }
    } finally {
        if (rotated !== source) rotated.recycle()
        source.recycle()
    }
}

internal suspend fun LocalVideoTrack.captureJpeg(): ByteArray =
    suspendCancellableCoroutine { continuation ->
        val completed = AtomicBoolean(false)
        lateinit var sink: VideoSink
        sink = VideoSink { frame ->
            if (!completed.compareAndSet(false, true)) return@VideoSink
            removeRenderer(sink)
            try {
                continuation.resume(encodeFrameJpeg(frame))
            } catch (error: Throwable) {
                continuation.resumeWithException(error)
            }
        }
        continuation.invokeOnCancellation {
            if (completed.compareAndSet(false, true)) removeRenderer(sink)
        }
        addRenderer(sink)
    }
