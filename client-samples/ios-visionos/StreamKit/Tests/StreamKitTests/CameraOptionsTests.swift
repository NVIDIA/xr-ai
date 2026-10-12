// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

@testable import LiveKit
import Testing
@testable import StreamKit

@Suite("Camera encoding publication policy")
struct CameraOptionsTests {
    @Test func absentPolicyUsesSDKDefaults() throws {
        #expect(try cameraPublishOptions(nil, defaults: VideoPublishOptions(simulcast: false)) == nil)
    }

    @Test func unspecifiedFieldsInheritDefaults() throws {
        let defaults = VideoPublishOptions(name: "camera", simulcast: false,
                                           degradationPreference: .maintainFramerate,
                                           streamName: "existing-stream")
        let options = try #require(try cameraPublishOptions(CameraEncodingConfig(), defaults: defaults))
        #expect(!options.simulcast)
        #expect(options.degradationPreference == .maintainFramerate)
        #expect(options.name == defaults.name && options.streamName == defaults.streamName)
        #expect(options.simulcastLayers == defaults.simulcastLayers)
        #expect(options.screenShareEncoding == defaults.screenShareEncoding)
        #expect(options.preferredCodec == defaults.preferredCodec)
        #expect(options.preferredBackupCodec == defaults.preferredBackupCodec)
        #expect(options.encoding == VideoEncoding(maxBitrate: 3_000_000, maxFps: 30))
    }

    @Test func presetsMapAdaptationPreference() throws {
        let defaults = VideoPublishOptions()
        #expect(try cameraPublishOptions(.detail, defaults: defaults)?.degradationPreference == .maintainResolution)
        #expect(try cameraPublishOptions(.motion, defaults: defaults)?.degradationPreference == .maintainFramerate)
        #expect(try cameraPublishOptions(.balanced, defaults: defaults)?.degradationPreference == .balanced)
        #expect(try cameraPublishOptions(.detail, defaults: defaults)?.simulcast == false)
    }

    @Test(arguments: [true, false]) func explicitSimulcastOverridesDefault(_ enabled: Bool) throws {
        let options = try #require(try cameraPublishOptions(CameraEncodingConfig(simulcast: enabled),
                                                       defaults: VideoPublishOptions(simulcast: !enabled)))
        #expect(options.simulcast == enabled)
    }

    @Test func invalidMutablePolicyIsRejectedWhenConsumed() {
        var policy = CameraEncodingConfig()
        policy.maxBitrateBps = 0
        #expect(throws: StreamKit.StreamError.self) { try cameraPublishOptions(policy, defaults: VideoPublishOptions()) }
        policy.maxBitrateBps = 1
        policy.maxFramerate = -1
        #expect(throws: StreamKit.StreamError.self) { try cameraPublishOptions(policy, defaults: VideoPublishOptions()) }
    }

    @Test func defaultSimulcastLayersKeepSDKGeometryAndSmallerLimits() throws {
        let defaults = VideoPublishOptions()
        let low = try #require(try cameraPublishOptions(CameraEncodingConfig(maxBitrateBps: 50_000),
                                                       defaults: defaults, dimensions: Dimensions(width: 1280, height: 720)))
        #expect(low.simulcastLayers.map(\.encoding.maxBitrate) == [17_777, 50_000])
        #expect(low.simulcastLayers.map(\.dimensions) == [VideoParameters.presetH180_169.dimensions,
                                                        VideoParameters.presetH360_169.dimensions])
        let normal = try #require(try cameraPublishOptions(.motion, defaults: defaults,
                                                          dimensions: Dimensions(width: 1280, height: 960)))
        #expect(normal.simulcastLayers == [VideoParameters.presetH180_43, VideoParameters.presetH360_43])
        #expect(defaults.simulcastLayers.isEmpty)
        let before = LiveKit.Utils.computeVideoEncodings(dimensions: Dimensions(width: 1280, height: 720),
                                                         publishOptions: VideoPublishOptions(encoding: VideoEncoding(maxBitrate: 50_000, maxFps: 30)))
        let actual = LiveKit.Utils.computeVideoEncodings(dimensions: Dimensions(width: 1280, height: 720), publishOptions: low)
        #expect(before.contains { $0.maxBitrateBps!.intValue > 50_000 })
        #expect(actual.allSatisfy { $0.maxBitrateBps!.intValue <= 50_000 })
        #expect(actual.map(\.scaleResolutionDownBy) == before.map(\.scaleResolutionDownBy))
        #expect(actual.map(\.rid) == before.map(\.rid))
    }

    @Test func customLayersScaleTogetherWhileDisabledSimulcastPreservesDefaults() throws {
        let layers = [
            VideoParameters(dimensions: .h180_169, encoding: VideoEncoding(maxBitrate: 20_000, maxFps: 12)),
            VideoParameters.presetH360_169,
        ]
        let defaults = VideoPublishOptions(simulcastLayers: layers)
        let options = try #require(try cameraPublishOptions(CameraEncodingConfig(maxBitrateBps: 50_000), defaults: defaults))
        #expect(options.simulcastLayers.map(\.encoding.maxBitrate) == [2_222, 50_000])
        #expect(options.simulcastLayers.map(\.dimensions) == layers.map(\.dimensions))
        #expect(options.simulcastLayers.map(\.encoding.maxFps) == layers.map(\.encoding.maxFps))
        #expect(defaults.simulcastLayers == layers)
        let disabled = try #require(try cameraPublishOptions(CameraEncodingConfig(simulcast: false), defaults: defaults))
        #expect(disabled.simulcastLayers == layers)
        #expect(throws: StreamKit.StreamError.self) {
            try cameraPublishOptions(CameraEncodingConfig(maxBitrateBps: 1), defaults: defaults)
        }
        let large = VideoPublishOptions(simulcastLayers: [
            VideoParameters(dimensions: .h180_169, encoding: VideoEncoding(maxBitrate: Int.max - 1, maxFps: 20)),
            VideoParameters(dimensions: .h360_169, encoding: VideoEncoding(maxBitrate: Int.max, maxFps: 20)),
        ])
        let bounded = try #require(try cameraPublishOptions(CameraEncodingConfig(maxBitrateBps: 50_000), defaults: large))
        #expect(bounded.simulcastLayers.map(\.encoding.maxBitrate) == [49_999, 50_000])
    }
}
