// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * StreamKit — BackendConfiguration
 *
 * Enum-based backend selection. Add a new case here when a new transport is integrated.
 * Alternatively, bypass this entirely with StreamSession(backend: myCustomBackend).
 */

import Foundation

// MARK: - BackendConfiguration

/// Selects the networking backend used by ``StreamSession``.
///
/// Pass this to ``StreamSession/init(_:)`` to use a built-in backend.
/// To supply a completely custom implementation, use ``StreamSession/init(backend:)``
/// with your own ``StreamingBackend`` conformance instead.
///
/// ```swift
/// // Built-in LiveKit backend
/// let session = StreamSession(.liveKit(LiveKitConfig(host: "192.168.1.100", token: jwt)))
///
/// // Custom backend (e.g. your own streaming SDK)
/// let session = StreamSession(backend: MyCustomBackend())
/// ```
public enum BackendConfiguration: Sendable {

    /// The LiveKit WebRTC backend.
    case liveKit(LiveKitConfig)
}

// MARK: - Factory

extension BackendConfiguration {
    /// Instantiates the concrete ``StreamingBackend`` for this configuration.
    public func makeBackend() -> any StreamingBackend {
        switch self {
        case .liveKit(let config):
            return LiveKitBackend(config: config)
        }
    }
}

// MARK: - LiveKitConfig

/// Connection parameters for the LiveKit backend.
///
/// Exactly one of ``token`` or ``tokenURL`` must be provided.
public struct LiveKitConfig: Sendable {

    /// IP address or hostname of the xr-ai hub (e.g. `"192.168.1.100"`).
    public var host: String

    /// Hub web-server port. Defaults to `8080`.
    ///
    /// The client connects to `wss://<host>:<port>`; the hub serves a
    /// same-origin /rtc proxy that forwards LiveKit signaling internally.
    /// This is *not* LiveKit's native signaling port (7880).
    public var port: Int

    /// A pre-signed LiveKit JWT token.
    /// The token must encode the room name and participant identity.
    public var token: String?

    /// URL of a token-generation endpoint.
    ///
    /// The SDK appends `?room=<roomName>&identity=<identity>` query parameters.
    /// The endpoint must return either a plain JWT string or `{ "token": "eyJ…" }`.
    public var tokenURL: URL?

    /// Identity of the server-side hub participant the agent publishes through
    /// (the LiveKit connector — `xr-hub-connector` by default, see
    /// `server-runtime/.../transport/livekit/config.py`). Outbound data is
    /// addressed only to this identity so it is never delivered to peer
    /// participants in the same room. Set to `nil` to broadcast to the whole
    /// room (the pre-isolation behaviour).
    public var hubIdentity: String?

    public init(
        host: String,
        port: Int = 8080,
        token: String? = nil,
        tokenURL: URL? = nil,
        hubIdentity: String? = "xr-hub-connector"
    ) {
        self.host = host
        self.port = port
        self.token = token
        self.tokenURL = tokenURL
        self.hubIdentity = hubIdentity
    }
}
