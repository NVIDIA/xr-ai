// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace streamkit {

struct ImageCaptureRequest {
    std::string request_id;
    std::int64_t timeout_ms = 0;
};

struct CapturedImage {
    std::vector<std::uint8_t> data;
    std::string mime_type = "image/jpeg";
    std::string name = "camera.jpg";
};

} // namespace streamkit
