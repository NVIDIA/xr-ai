// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.xrai.streamkitsample.streamkit.config

/** Encoder tradeoff under resource or bandwidth pressure; not a quality guarantee. */
enum class VideoQualityPreference {
    BALANCED,
    /** Favor resolution over frame rate. */
    DETAIL,
    /** Favor frame rate over resolution. */
    MOTION,
}

/**
 * Optional publishing policy. Capture format is unchanged. Limits apply to each
 * encoded stream, not aggregate network traffic. Backends apply supported settings.
 * Restart the camera to apply changes. Defaults when opting in are 3 Mbps and 30 FPS.
 */
data class CameraEncodingConfig(
    /** Positive ceiling in bits per second. */
    val maxBitrateBps: Int = 3_000_000,
    /** Positive integer ceiling in frames per second. Actual FPS may be lower. */
    val maxFramerate: Int = 30,
    /** Null preserves the backend's adaptation preference. */
    val qualityPreference: VideoQualityPreference? = null,
    /** Publish multiple resolutions when supported. Null preserves backend defaults. */
    val simulcast: Boolean? = null,
) {
    init {
        require(maxBitrateBps > 0) { "maxBitrateBps must be positive" }
        require(maxFramerate > 0) { "maxFramerate must be positive" }
    }

    companion object {
        /** Favors resolution and disables simulcast; cannot guarantee minimum resolution. */
        @JvmField val DETAIL = CameraEncodingConfig(qualityPreference = VideoQualityPreference.DETAIL, simulcast = false)
        @JvmField val MOTION = CameraEncodingConfig(qualityPreference = VideoQualityPreference.MOTION)
        @JvmField val BALANCED = CameraEncodingConfig(qualityPreference = VideoQualityPreference.BALANCED)
    }
}

/**
 * Configures camera capture for a [StreamSession].
 *
 * Mirror of Swift `CameraConfig` / web `CameraConfig`.
 * Capture resolution and frame-rate are intentionally omitted: LiveKit and the
 * hardware negotiate the best supported format automatically.
 *
 * ## Presets
 * ```kotlin
 * CameraConfig.DEFAULT   // enabled, back-facing (primary camera on Android)
 * CameraConfig.FRONT     // enabled, front-facing (selfie camera)
 * CameraConfig.DISABLED  // camera off
 * ```
 */
data class CameraConfig(
    val enabled: Boolean = true,
    val facing: CameraFacing = CameraFacing.BACK,
    /**
     * Optional Camera2 camera id (e.g. `"0"`, `"1"`). When non-null, the
     * backend pins capture to that exact camera and ignores [facing]. When
     * null, the backend picks any camera matching [facing].
     *
     * Use this to choose between multiple cameras on the same side
     * (e.g. wide vs. ultra-wide vs. telephoto on the back).
     */
    val deviceId: String? = null,
    /** Optional publish-side policy. Null preserves backend defaults. */
    val encoding: CameraEncodingConfig? = null,
) {

    /**
     * Camera facing direction.
     *
     * Mirror of Swift `CameraConfig.Position` and web `CameraFacing`.
     */
    enum class CameraFacing {
        /** Front-facing (selfie) camera. */
        FRONT,

        /** Rear-facing (primary) camera. */
        BACK,
    }

    companion object {
        /** Camera enabled, rear-facing — natural default on Android. */
        @JvmField val DEFAULT = CameraConfig(enabled = true, facing = CameraFacing.BACK)

        /** Camera enabled, front-facing (selfie). */
        @JvmField val FRONT = CameraConfig(enabled = true, facing = CameraFacing.FRONT)

        /** Camera disabled — nothing is captured or published. */
        @JvmField val DISABLED = CameraConfig(enabled = false, facing = CameraFacing.BACK)
    }
}
