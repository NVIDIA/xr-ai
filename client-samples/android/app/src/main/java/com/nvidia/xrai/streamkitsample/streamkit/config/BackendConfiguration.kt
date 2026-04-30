package com.nvidia.xrai.streamkitsample.streamkit.config

import android.content.Context
import com.nvidia.xrai.streamkitsample.streamkit.backends.StreamingBackend
import com.nvidia.xrai.streamkitsample.streamkit.backends.livekit.LiveKitBackend

/**
 * Selects the networking backend used by [StreamSession][com.nvidia.xrai.streamkitsample.streamkit.StreamSession].
 *
 * Pass this to `StreamSession(backendConfig, context)` to use a built-in backend.
 * To supply a completely custom implementation, pass your own [StreamingBackend]
 * directly to `StreamSession(backend)`.
 *
 * Mirror of Swift `BackendConfiguration` and web `BackendConfiguration`.
 *
 * ```kotlin
 * // Built-in LiveKit backend
 * val session = StreamSession(
 *     BackendConfiguration.LiveKit(LiveKitConfig(host = "192.168.1.100", token = jwt)),
 *     applicationContext,
 * )
 * ```
 */
sealed class BackendConfiguration {

    /** The LiveKit WebRTC backend. */
    data class LiveKit(val config: LiveKitConfig) : BackendConfiguration() {
        fun makeBackend(context: Context): StreamingBackend =
            LiveKitBackend(config, context.applicationContext)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LiveKitConfig
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Connection parameters for the LiveKit backend.
 *
 * Exactly one of [token] or [tokenURL] must be provided before calling
 * [StreamSession.connect][com.nvidia.xrai.streamkitsample.streamkit.StreamSession.connect].
 *
 * Mirror of Swift `LiveKitConfig` and web `LiveKitConfig`.
 */
data class LiveKitConfig(
    /**
     * IP address or hostname of the LiveKit server (e.g. `"192.168.1.100"`).
     * Do not include a scheme or port.
     */
    val host: String,

    /** WebSocket port. Defaults to `7880` (LiveKit's default). */
    val port: Int = 7880,

    /**
     * Use `https://` for the default token endpoint. Set `false` when the
     * token server speaks plain `http://`.
     *
     * The LiveKit signaling socket is **always** plain `ws://` in the xr-ai
     * reference deployment (port 7880) — TLS terminates at the web/token
     * server, not at LiveKit. This flag therefore does not affect the
     * signaling URL.
     */
    val secure: Boolean = false,

    /**
     * A pre-signed LiveKit JWT token.
     * The token must encode the room name and participant identity.
     */
    val token: String? = null,

    /**
     * URL string of a token-generation endpoint.
     * The SDK appends `?identity=<identity>` as a query parameter.
     * The endpoint must return either a plain JWT string or `{ "token": "eyJ…" }`.
     */
    val tokenURL: String? = null,
)
