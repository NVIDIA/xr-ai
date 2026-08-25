// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.xrai.streamkitsample.streamkit.config

/**
 * Configures microphone capture for a [StreamSession].
 *
 * Mirror of Swift `AudioConfig` / web `AudioConfig`.
 *
 * ## Presets
 * ```kotlin
 * AudioConfig.DEFAULT          // LiveKit Android capture defaults
 * AudioConfig.SOFTWARE         // Same behavior in the current Android backend
 * AudioConfig.RAW              // Same behavior in the current Android backend
 * AudioConfig.DISABLED         // Microphone off
 * ```
 *
 * The mode names preserve the cross-platform StreamKit configuration shape.
 * The current Android backend maps every enabled mode to LiveKit's default
 * microphone capture and does not select different DSP settings.
 */
data class AudioConfig(
    val mode: MicrophoneMode = MicrophoneMode.VOICE_PROCESSING,
) {

    /**
     * Microphone processing mode.
     *
     * Mirror of Swift `AudioConfig.MicrophoneMode` and web `MicrophoneMode`.
     */
    enum class MicrophoneMode {
        /**
         * Uses LiveKit Android's default microphone capture settings.
         */
        VOICE_PROCESSING,

        /**
         * Reserved for cross-platform parity; currently uses the same LiveKit
         * Android defaults as [VOICE_PROCESSING].
         */
        SOFTWARE_PROCESSING,

        /**
         * Reserved for cross-platform parity; currently uses the same LiveKit
         * Android defaults as [VOICE_PROCESSING].
         */
        RAW,

        /**
         * Microphone is not captured or published.
         *
         * Mirrors Swift `.disabled` and web `MicrophoneMode.DISABLED`.
         */
        DISABLED,
    }

    companion object {
        /** LiveKit Android's default microphone capture. */
        @JvmField val DEFAULT = AudioConfig(mode = MicrophoneMode.VOICE_PROCESSING)

        /** Cross-platform alias; equivalent to [DEFAULT] in this backend. */
        @JvmField val SOFTWARE = AudioConfig(mode = MicrophoneMode.SOFTWARE_PROCESSING)

        /** Cross-platform alias; equivalent to [DEFAULT] in this backend. */
        @JvmField val RAW = AudioConfig(mode = MicrophoneMode.RAW)

        /** Microphone disabled. */
        @JvmField val DISABLED = AudioConfig(mode = MicrophoneMode.DISABLED)
    }
}
