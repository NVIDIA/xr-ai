// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import Foundation

/// Application metadata for a client-to-agent file transfer.
public struct FileSendOptions: Sendable {
    public var topic: String
    public var name: String?
    public var mimeType: String?
    public var attributes: [String: String]

    public init(
        topic: String,
        name: String? = nil,
        mimeType: String? = nil,
        attributes: [String: String] = [:]
    ) {
        self.topic = topic
        self.name = name
        self.mimeType = mimeType
        self.attributes = attributes
    }
}

/// Identifies a file transfer completed by the local transport.
public struct FileTransferInfo: Sendable, Equatable {
    public let id: String
    public let topic: String
    public let name: String
    public let mimeType: String
    public let size: Int

    public init(id: String, topic: String, name: String, mimeType: String, size: Int) {
        self.id = id
        self.topic = topic
        self.name = name
        self.mimeType = mimeType
        self.size = size
    }
}
