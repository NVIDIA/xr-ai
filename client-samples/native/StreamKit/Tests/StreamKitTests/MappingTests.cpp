// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "test_assert.h"

#include "streamkit/ConnectionState.h"

int main() {
    using enum streamkit::ConnectionState;
    using streamkit::test::Expect;
    using streamkit::test::ExpectEq;

    ExpectEq(kConnected, kConnected);
    Expect(kDisconnected != kConnected);
    Expect(kConnecting   != kConnected);
    Expect(kReconnecting != kConnected);

    return 0;
}
