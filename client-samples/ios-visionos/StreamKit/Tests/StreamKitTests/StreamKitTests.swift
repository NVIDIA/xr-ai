// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import Foundation
import LiveKit
import Testing
@testable import StreamKit

private final class MockStreamingBackend: StreamingBackend, @unchecked Sendable {
    var onConnectionStateChanged: (@Sendable (StreamKit.ConnectionState) -> Void)?
    var onDataReceived: (@Sendable (String, Data) -> Void)?
    var onAgentStatus: (@Sendable (String) -> Void)?
    var onNetworkMetrics: (@Sendable (NetworkMetrics) -> Void)?

    func connect(config: SessionConfig) async throws {}
    func disconnect() async {}
    func startAudio(config: AudioConfig) async throws {}
    func stopAudio() async throws {}
    func startCamera(config: CameraConfig) async throws {}
    func stopCamera() async throws {}
    func send(_ data: Data, reliable: Bool) async throws {}
}

@Suite("ConnectionState")
struct ConnectionStateTests {
    @Test func equatable() {
        #expect(StreamKit.ConnectionState.connected == .connected)
        #expect(StreamKit.ConnectionState.disconnected != .connected)
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

    @Test func mapsEveryLiveKitQuality() {
        #expect(LiveKit.ConnectionQuality.excellent.toStreamKitQuality() == .excellent)
        #expect(LiveKit.ConnectionQuality.good.toStreamKitQuality() == .good)
        #expect(LiveKit.ConnectionQuality.poor.toStreamKitQuality() == .poor)
        #expect(LiveKit.ConnectionQuality.lost.toStreamKitQuality() == .lost)
        #expect(LiveKit.ConnectionQuality.unknown.toStreamKitQuality() == .unknown)
    }
}

@Suite("StreamSession telemetry lifecycle")
@MainActor
struct StreamSessionTelemetryTests {
    @Test func clearsMetricsOutsideConnectedStateAndDropsLateSamples() async {
        let backend = MockStreamingBackend()
        let session = StreamSession(backend: backend)
        let metrics = NetworkMetrics(quality: .good, roundTripTimeMs: 24)

        backend.onConnectionStateChanged?(.connected)
        await Task.yield()
        backend.onNetworkMetrics?(metrics)
        await Task.yield()
        #expect(session.networkMetrics == metrics)

        backend.onConnectionStateChanged?(.reconnecting)
        await Task.yield()
        #expect(session.networkMetrics == nil)

        backend.onNetworkMetrics?(metrics)
        await Task.yield()
        #expect(session.networkMetrics == nil)

        backend.onConnectionStateChanged?(.disconnected)
        await Task.yield()
        #expect(session.networkMetrics == nil)
    }
}
