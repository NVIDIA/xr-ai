// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.xrai.streamkitsample

import java.nio.ByteBuffer

/**
 * Generates synthetic I420 video frames for the "Virtual Camera" provider.
 *
 * Demonstrates the public [StreamSession.injectVideoFrame][com.nvidia.xrai.streamkitsample.streamkit.StreamSession.injectVideoFrame]
 * API: an external frame source (no physical camera) feeding raw I420 buffers
 * into the stream. Produces scrolling SMPTE-style colour bars plus a bouncing
 * white box so motion is obvious on the receiver and in the local preview.
 *
 * Not thread-safe: drive it from a single coroutine. The returned buffer is
 * reused across calls — `injectVideoFrame` copies the planes before returning,
 * so the caller may overwrite it on the next frame.
 *
 * @param width  Even pixel width (default 640).
 * @param height Even pixel height (default 480).
 */
class SyntheticCameraSource(
    val width: Int = 640,
    val height: Int = 480,
) {
    init {
        require(width > 0 && height > 0 && (width and 1) == 0 && (height and 1) == 0) {
            "width/height must be positive and even"
        }
    }

    private val ySize = width * height
    private val cWidth = width / 2
    private val cHeight = height / 2
    private val cSize = cWidth * cHeight
    private val total = ySize + 2 * cSize

    // Generate into a heap array, then bulk-copy into the direct buffer once —
    // far cheaper than per-byte writes to a direct ByteBuffer.
    private val scratch = ByteArray(total)
    private val buffer: ByteBuffer = ByteBuffer.allocateDirect(total)

    // Classic 8-bar palette in BT.601 limited-range YUV: white, yellow, cyan,
    // green, magenta, red, blue, black.
    private val barY = intArrayOf(235, 210, 170, 145, 106, 81, 41, 16)
    private val barU = intArrayOf(128, 16, 166, 54, 202, 90, 240, 128)
    private val barV = intArrayOf(128, 146, 16, 34, 222, 240, 110, 128)
    private val barCount = barY.size

    private val barWidth = width / barCount
    private val boxSize = (height / 6).coerceAtLeast(8)

    /**
     * Fill the I420 buffer for frame [frameIndex] and return it (position 0,
     * limit = frame size). The same buffer instance is returned every call.
     */
    fun renderFrame(frameIndex: Int): ByteBuffer {
        // Bars scroll left→right; box bounces horizontally.
        val scroll = (frameIndex * 2)
        val travel = width - boxSize
        val phase = if (travel > 0) (frameIndex * 3) % (2 * travel) else 0
        val boxX = if (phase < travel) phase else (2 * travel - phase)
        val boxY = (height - boxSize) / 2

        // Luma plane (full resolution).
        for (y in 0 until height) {
            val rowBase = y * width
            val inBoxRow = y >= boxY && y < boxY + boxSize
            for (x in 0 until width) {
                val value = if (inBoxRow && x >= boxX && x < boxX + boxSize) {
                    235 // white box
                } else {
                    val bar = (((x + scroll) / barWidth) % barCount)
                    barY[bar]
                }
                scratch[rowBase + x] = value.toByte()
            }
        }

        // Chroma planes (half resolution). The box is left luma-only (greyish),
        // which still reads clearly against the bars.
        val uBase = ySize
        val vBase = ySize + cSize
        for (cy in 0 until cHeight) {
            val rowBase = cy * cWidth
            for (cx in 0 until cWidth) {
                val bar = ((((cx * 2) + scroll) / barWidth) % barCount)
                scratch[uBase + rowBase + cx] = barU[bar].toByte()
                scratch[vBase + rowBase + cx] = barV[bar].toByte()
            }
        }

        buffer.clear()
        buffer.put(scratch)
        buffer.flip()
        return buffer
    }
}
