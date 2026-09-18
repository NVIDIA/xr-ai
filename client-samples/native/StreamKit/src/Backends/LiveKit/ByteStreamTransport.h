// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace livekit {
class Room;
}

namespace streamkit::detail {

struct ByteStreamWireOptions {
    std::string topic;
    std::map<std::string, std::string> attributes;
    std::vector<std::string> destination_identities;
    std::string mime_type;
    std::string name;
    std::size_t total_size;
};

struct ByteStreamConnection {
    std::shared_ptr<livekit::Room> room;
    std::function<bool()> is_active;
};

class ByteStreamConnectionChanged : public std::runtime_error {
public:
    ByteStreamConnectionChanged();
};

class LiveKitByteStreamWriter {
public:
    static std::string SendBytes(
        std::span<const std::byte> data,
        const ByteStreamWireOptions& options,
        const ByteStreamConnection& connection);

    static std::string SendFile(
        const std::filesystem::path& path,
        const ByteStreamWireOptions& options,
        const ByteStreamConnection& connection);
};

} // namespace streamkit::detail
