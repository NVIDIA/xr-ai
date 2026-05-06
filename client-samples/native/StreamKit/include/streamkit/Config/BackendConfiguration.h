// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/*
 * StreamKit — BackendConfiguration
 *
 * Variant-based backend selection. Add a new alternative here when a new
 * transport is integrated. To bypass entirely, construct StreamSession with
 * a custom StreamingBackend directly.
 *
 * Mirror of Swift `BackendConfiguration`, Kotlin `BackendConfiguration`,
 * and JS `BackendConfiguration`.
 */

#include <memory>
#include <optional>
#include <string>
#include <variant>

namespace streamkit {

class StreamingBackend;

// ─────────────────────────────────────────────────────────────────────────────
// LiveKitConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Connection parameters for the LiveKit backend.
///
/// Exactly one of `token` or `token_url` must be set before calling
/// StreamSession::Connect().
///
/// Mirror of Swift `LiveKitConfig` and Kotlin `LiveKitConfig`.
struct LiveKitConfig {
    /// IP address or hostname of the LiveKit server (e.g. "192.168.1.100").
    /// Do not include a scheme or port.
    std::string host;

    /// WebSocket port. Defaults to 7880 (LiveKit's default).
    int port = 7880;

    /// Use wss:// / https://. Set false for local / LAN connections.
    bool secure = false;

    /// A pre-signed LiveKit JWT. The token must encode the room name and
    /// participant identity.
    std::optional<std::string> token;

    /// URL of a token-generation endpoint.
    /// The SDK appends `?identity=<identity>` as a query parameter.
    /// The endpoint must return either a plain JWT string or {"token":"eyJ…"}.
    std::optional<std::string> token_url;
};

// ─────────────────────────────────────────────────────────────────────────────
// BackendConfiguration
// ─────────────────────────────────────────────────────────────────────────────

/// Selects the networking backend used by StreamSession.
///
/// ```cpp
/// // Built-in LiveKit backend
/// auto session = StreamSession(BackendConfiguration{LiveKitConfig{
///     .host      = "192.168.1.100",
///     .token     = jwt,
/// }});
///
/// // Custom backend
/// auto session = StreamSession(std::make_unique<MyCustomBackend>());
/// ```
using BackendConfiguration = std::variant<LiveKitConfig>;

/// Instantiates the concrete StreamingBackend for the given configuration.
std::unique_ptr<StreamingBackend> MakeBackend(const BackendConfiguration& config);

} // namespace streamkit
