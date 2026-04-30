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
