// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <map>
#include <string>

namespace streamkit {

/// Application metadata for a client-to-agent file transfer.
struct FileSendOptions {
    std::string topic;
    std::string name;
    std::string mime_type = "application/octet-stream";
    std::map<std::string, std::string> attributes;
};

/// Identifies a file transfer completed by the local transport.
struct FileTransferInfo {
    std::string id;
    std::string topic;
    std::string name;
    std::string mime_type;
    std::size_t size = 0;
};

} // namespace streamkit
