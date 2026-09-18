// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "test_assert.h"

#include "Backends/LiveKit/ByteStreamTransport.h"

#include <array>
#include <cstddef>
#include <string>

int main() {
    using streamkit::test::Expect;
    using streamkit::detail::ByteStreamConnection;
    using streamkit::detail::ByteStreamConnectionChanged;
    using streamkit::detail::ByteStreamWireOptions;
    using streamkit::detail::LiveKitByteStreamWriter;

    bool active = true;
    const ByteStreamConnection connection{
        .room = nullptr,
        .is_active = [&active]() { return active; },
    };
    const ByteStreamWireOptions options{
        .topic = "capture.response",
        .attributes = {{"request_id", "request-1"}},
        .destination_identities = {"xr-hub-connector"},
        .mime_type = "image/png",
        .name = "capture.png",
        .total_size = 3,
    };
    const std::array<std::byte, 3> bytes{
        std::byte{1},
        std::byte{2},
        std::byte{3},
    };

    const auto id = LiveKitByteStreamWriter::SendBytes(bytes, options, connection);
    Expect(id.starts_with("stub-"));

    active = false;
    bool rejected = false;
    try {
        (void)LiveKitByteStreamWriter::SendBytes(bytes, options, connection);
    } catch (const ByteStreamConnectionChanged&) {
        rejected = true;
    }
    Expect(rejected);
}
