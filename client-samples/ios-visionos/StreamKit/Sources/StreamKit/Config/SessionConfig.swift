// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import Foundation

/// Configuration passed to ``StreamSession/connect(config:)``.
///
/// Only carries identity — network details live in the backend config
/// (e.g. ``LiveKitConfig``), and media settings are passed directly to
/// ``StreamSession/startAudio(config:)`` and ``StreamSession/startCamera(config:)``.
public struct SessionConfig: Sendable {

    /// A unique label for this participant in the session.
    public var identity: String

    public static let `default` = SessionConfig()

    public init(identity: String = "participant-\(UInt32.random(in: 100_000...999_999))") {
        self.identity = identity
    }
}
