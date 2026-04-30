package com.nvidia.xrai.streamkitsample.streamkit.config

/**
 * Configuration passed to [StreamSession.connect].
 *
 * Only carries participant identity — network details live in [BackendConfiguration],
 * and media settings are passed directly to [StreamSession.startAudio] /
 * [StreamSession.startCamera].
 *
 * Mirror of Swift `SessionConfig` and web `SessionConfig`.
 */
data class SessionConfig(
    /**
     * A unique label for this participant in the session.
     * Must be non-empty and unique within the LiveKit room.
     */
    val identity: String = "participant-${(100_000..999_999).random()}",
) {
    companion object {
        /** Default session: random identity. */
        @JvmField val DEFAULT = SessionConfig()
    }
}
