// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

namespace streamkit {

/// Transport-neutral quality estimate from the active streaming backend.
enum class NetworkQuality { kUnknown, kExcellent, kGood, kPoor, kLost };

/// A once-per-second snapshot of LiveKit's native WebRTC telemetry.
/// RTT and jitter are milliseconds and remain empty until stats are available.
struct NetworkMetrics {
    NetworkQuality quality = NetworkQuality::kUnknown;
    std::optional<double> round_trip_time_ms;
    std::optional<double> receive_jitter_ms;
};

} // namespace streamkit
