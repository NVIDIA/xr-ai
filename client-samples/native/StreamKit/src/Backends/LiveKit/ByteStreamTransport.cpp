// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "ByteStreamTransport.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <string>
#include <utility>

#if STREAMKIT_HAVE_LIVEKIT
#include "livekit/data_stream.h"
#include "livekit/local_participant.h"
#include "livekit/room.h"
#endif

namespace streamkit::detail {

namespace {

constexpr std::size_t kWriteChunkBytes = 64 * 1024;

void RequireActive(const ByteStreamConnection& connection) {
    if (!connection.is_active()) {
        throw ByteStreamConnectionChanged{};
    }
}

#if STREAMKIT_HAVE_LIVEKIT
template <typename Write>
std::string Send(
    const ByteStreamWireOptions& options,
    const ByteStreamConnection& connection,
    Write&& write) {
    RequireActive(connection);
    if (!connection.room) {
        throw ByteStreamConnectionChanged{};
    }
    const auto participant = connection.room->localParticipant().lock();
    if (!participant) {
        throw ByteStreamConnectionChanged{};
    }
    livekit::ByteStreamWriter writer(
        *participant,
        options.name,
        options.topic,
        options.attributes,
        "",
        options.total_size,
        options.mime_type,
        options.destination_identities);
    try {
        std::forward<Write>(write)(writer);
        RequireActive(connection);
        writer.close();
        if (!writer.isClosed()) {
            throw ByteStreamConnectionChanged{};
        }
        RequireActive(connection);
    } catch (...) {
        if (!writer.isClosed()) {
            try {
                writer.close("StreamKit send failed");
            } catch (...) {
            }
        }
        throw;
    }
    return writer.info().stream_id;
}
#endif

} // namespace

ByteStreamConnectionChanged::ByteStreamConnectionChanged()
    : std::runtime_error("The byte stream is no longer on the active connection") {}

std::string LiveKitByteStreamWriter::SendBytes(
    std::span<const std::byte> data,
    const ByteStreamWireOptions& options,
    const ByteStreamConnection& connection) {
#if STREAMKIT_HAVE_LIVEKIT
    return Send(options, connection, [&](livekit::ByteStreamWriter& writer) {
        for (std::size_t offset = 0; offset < data.size(); offset += kWriteChunkBytes) {
            RequireActive(connection);
            const auto size = std::min(kWriteChunkBytes, data.size() - offset);
            std::vector<std::uint8_t> chunk(size);
            for (std::size_t index = 0; index < size; ++index) {
                chunk[index] = std::to_integer<std::uint8_t>(data[offset + index]);
            }
            writer.write(chunk);
        }
    });
#else
    (void)data;
    (void)options;
    RequireActive(connection);
    return "stub-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
#endif
}

std::string LiveKitByteStreamWriter::SendFile(
    const std::filesystem::path& path,
    const ByteStreamWireOptions& options,
    const ByteStreamConnection& connection) {
#if STREAMKIT_HAVE_LIVEKIT
    return Send(options, connection, [&](livekit::ByteStreamWriter& writer) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Could not open file: " + path.string());
        }
        std::vector<std::uint8_t> chunk(kWriteChunkBytes);
        while (file) {
            file.read(
                reinterpret_cast<char*>(chunk.data()),
                static_cast<std::streamsize>(chunk.size()));
            const auto count = file.gcount();
            if (count > 0) {
                RequireActive(connection);
                writer.write(std::vector<std::uint8_t>(
                    chunk.begin(), chunk.begin() + count));
            }
        }
        if (!file.eof()) {
            throw std::runtime_error("Could not read file: " + path.string());
        }
    });
#else
    (void)path;
    (void)options;
    RequireActive(connection);
    return "stub-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
#endif
}

} // namespace streamkit::detail
