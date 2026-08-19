// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import Foundation

/// Transport-neutral quality estimate from the active streaming backend.
public enum NetworkQuality: String, Sendable {
    case unknown, excellent, good, poor, lost
}

/// A once-per-second snapshot of LiveKit's native WebRTC telemetry.
/// RTT and jitter are milliseconds and remain nil until stats are available.
public struct NetworkMetrics: Sendable, Equatable {
    public let quality: NetworkQuality
    public let roundTripTimeMs: Double?
    public let receiveJitterMs: Double?

    public init(
        quality: NetworkQuality = .unknown,
        roundTripTimeMs: Double? = nil,
        receiveJitterMs: Double? = nil
    ) {
        self.quality = quality
        self.roundTripTimeMs = roundTripTimeMs
        self.receiveJitterMs = receiveJitterMs
    }
}
