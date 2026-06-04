// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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
     * IP address or hostname of the xr-ai hub (e.g. `"192.168.1.100"`).
     * Do not include a scheme or port.
     */
    val host: String,

    /**
     * Hub web-server port. Defaults to `8080`.
     *
     * The client connects to `wss://<host>:<port>`; the hub serves a
     * same-origin /rtc proxy that forwards LiveKit signaling internally.
     * This is *not* LiveKit's native signaling port (7880).
     */
    val port: Int = 8080,

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

    /**
     * Identity of the server-side hub participant the agent publishes through
     * (the LiveKit connector — `xr-hub-connector` by default, see
     * `server-runtime/.../transport/livekit/config.py`). Outbound data is
     * addressed only to this identity so it is never delivered to peer
     * participants in the same room. Set to `null` to broadcast to the whole
     * room (the pre-isolation behaviour).
     */
    val hubIdentity: String? = "xr-hub-connector",
)
