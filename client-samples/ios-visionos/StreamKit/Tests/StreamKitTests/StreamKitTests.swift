// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import Testing
@testable import StreamKit

@Suite("ConnectionState")
struct ConnectionStateTests {
    @Test func equatable() {
        #expect(ConnectionState.connected == .connected)
        #expect(ConnectionState.disconnected != .connected)
    }
}

@Suite("NetworkMetrics")
struct NetworkMetricsTests {
    @Test func storesMillisecondsAndQuality() {
        let metrics = NetworkMetrics(
            quality: .good,
            roundTripTimeMs: 24,
            receiveJitterMs: 3
        )
        #expect(metrics.quality == .good)
        #expect(metrics.roundTripTimeMs == 24)
        #expect(metrics.receiveJitterMs == 3)
    }
}
