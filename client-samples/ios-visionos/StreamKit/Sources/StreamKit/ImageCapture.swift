// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import Foundation

/// A participant-scoped request from the remote agent to capture one image.
public struct ImageCaptureRequest: Sendable {
    public let requestID: String
    public let timeoutMilliseconds: Int
}

/// Encoded still image returned by an application's opt-in capture handler.
public struct CapturedImage: Sendable {
    public let data: Data
    public let mimeType: String
    public let name: String

    public init(data: Data, mimeType: String = "image/jpeg", name: String = "camera.jpg") {
        self.data = data
        self.mimeType = mimeType
        self.name = name
    }
}
