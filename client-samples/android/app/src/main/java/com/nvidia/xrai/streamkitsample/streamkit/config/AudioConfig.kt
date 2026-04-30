package com.nvidia.xrai.streamkitsample.streamkit.config

/**
 * Configures microphone capture for a [StreamSession].
 *
 * Mirror of Swift `AudioConfig` / web `AudioConfig`.
 *
 * ## Presets
 * ```kotlin
 * AudioConfig.DEFAULT          // Voice processing — best for voice calls
 * AudioConfig.SOFTWARE         // WebRTC DSP stack
 * AudioConfig.RAW              // No processing — let the server handle DSP
 * AudioConfig.DISABLED         // Microphone off
 * ```
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
         * Android hardware acoustic echo cancellation / AGC / noise suppression via
         * [android.media.audiofx.AcousticEchoCanceler] etc. Mapped to
         * `echoCancellation=false` on the LiveKit WebRTC track because the OS-level
         * processing happens before the audio ever reaches WebRTC.
         *
         * Mirrors Swift `.voiceProcessing` (AUVoiceIO).
         */
        VOICE_PROCESSING,

        /**
         * WebRTC software DSP: echo cancellation, AGC, and noise suppression applied
         * inside libwebrtc. Use when hardware effects are unavailable.
         *
         * Mirrors Swift `.softwareProcessing` and web `MicrophoneMode.SOFTWARE_PROCESSING`.
         */
        SOFTWARE_PROCESSING,

        /**
         * Raw PCM capture — all DSP disabled. Choose this when the server handles
         * audio processing (e.g. the echo-agent / vlm-agent workers).
         *
         * Mirrors Swift `.raw` and web `MicrophoneMode.RAW`.
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
        /** Voice-processing mode — best default for voice calls. */
        @JvmField val DEFAULT = AudioConfig(mode = MicrophoneMode.VOICE_PROCESSING)

        /** WebRTC software DSP. */
        @JvmField val SOFTWARE = AudioConfig(mode = MicrophoneMode.SOFTWARE_PROCESSING)

        /** Raw PCM — no DSP. */
        @JvmField val RAW = AudioConfig(mode = MicrophoneMode.RAW)

        /** Microphone disabled. */
        @JvmField val DISABLED = AudioConfig(mode = MicrophoneMode.DISABLED)
    }
}
