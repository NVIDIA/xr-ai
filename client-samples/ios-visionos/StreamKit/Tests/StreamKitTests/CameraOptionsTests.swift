// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import LiveKit
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
        let options = try #require(cameraPublishOptions(CameraEncodingConfig(), defaults: defaults))
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
        let options = try #require(cameraPublishOptions(CameraEncodingConfig(simulcast: enabled),
                                                       defaults: VideoPublishOptions(simulcast: !enabled)))
        #expect(options.simulcast == enabled)
    }

    @Test func invalidMutablePolicyIsRejectedWhenConsumed() {
        var policy = CameraEncodingConfig()
        policy.maxBitrateBps = 0
        #expect(throws: StreamError.self) { try cameraPublishOptions(policy, defaults: VideoPublishOptions()) }
        policy.maxBitrateBps = 1
        policy.maxFramerate = -1
        #expect(throws: StreamError.self) { try cameraPublishOptions(policy, defaults: VideoPublishOptions()) }
    }
}
