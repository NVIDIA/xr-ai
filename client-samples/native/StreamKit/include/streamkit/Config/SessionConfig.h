// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
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

    /// Returns a config with a process-local unique participant ID.
    static SessionConfig Default() {
        static std::atomic_uint64_t sequence{0};
        const auto timestamp =
            std::chrono::steady_clock::now().time_since_epoch().count();
        return SessionConfig{
            "participant-" + std::to_string(timestamp) + "-" +
            std::to_string(sequence.fetch_add(1, std::memory_order_relaxed))};
    }
};

} // namespace streamkit
