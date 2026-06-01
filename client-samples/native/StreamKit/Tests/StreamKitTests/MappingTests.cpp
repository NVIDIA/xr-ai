// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "test_assert.h"

#include "streamkit/ConnectionState.h"

int main() {
    using streamkit::ConnectionState;
    using streamkit::test::Expect;
    using streamkit::test::ExpectEq;

    ExpectEq(ConnectionState::kConnected, ConnectionState::kConnected);
    Expect(ConnectionState::kDisconnected != ConnectionState::kConnected);
    Expect(ConnectionState::kConnecting   != ConnectionState::kConnected);
    Expect(ConnectionState::kReconnecting != ConnectionState::kConnected);

    return 0;
}
