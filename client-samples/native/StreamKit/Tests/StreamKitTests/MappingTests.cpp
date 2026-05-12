// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "test_assert.h"

#include "streamkit/ConnectionState.h"

int main() {
    using streamkit::ConnectionState;

    SK_EXPECT_EQ(ConnectionState::kConnected, ConnectionState::kConnected);
    SK_EXPECT(ConnectionState::kDisconnected != ConnectionState::kConnected);
    SK_EXPECT(ConnectionState::kConnecting   != ConnectionState::kConnected);
    SK_EXPECT(ConnectionState::kReconnecting != ConnectionState::kConnected);

    return 0;
}
