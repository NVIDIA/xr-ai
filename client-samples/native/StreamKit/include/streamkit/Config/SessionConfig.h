// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <random>
#include <string>

namespace streamkit {

/// Configuration passed to StreamSession::Connect().
///
/// Only carries identity — network details live in BackendConfiguration /
/// LiveKitConfig, and media settings are passed directly to StartAudio()
/// and StartCamera().
///
/// Mirror of Swift `SessionConfig` and Kotlin `SessionConfig`.
struct SessionConfig {
    /// A unique label for this participant in the session.
    std::string identity;

    /// Returns a config with a random participant ID.
    static SessionConfig Default() {
        static std::mt19937 rng{std::random_device{}()};
        static std::uniform_int_distribution<uint32_t> dist{100000, 999999};
        return SessionConfig{"participant-" + std::to_string(dist(rng))};
    }
};

} // namespace streamkit
