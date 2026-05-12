// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/*
 * Internal helper — parses the `_agent.status` topic payload. The contract is
 * a JSON object `{"status": "..."}` per xr-ai-pipecat's
 * `agent-sdk/xr_ai_agent/_processor.py::set_status`. Inlined into a header
 * so the unit tests can exercise it without exporting a public symbol.
 */

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace streamkit::internal {

inline std::optional<std::string> ExtractAgentStatus(std::span<const std::byte> payload) {
    std::string_view sv(reinterpret_cast<const char*>(payload.data()), payload.size());
    auto key = sv.find("\"status\"");
    if (key == std::string_view::npos) return std::nullopt;
    auto colon = sv.find(':', key + 8);
    if (colon == std::string_view::npos) return std::nullopt;
    auto quote_open = sv.find('"', colon + 1);
    if (quote_open == std::string_view::npos) return std::nullopt;
    auto quote_close = sv.find('"', quote_open + 1);
    if (quote_close == std::string_view::npos) return std::nullopt;
    return std::string(sv.substr(quote_open + 1, quote_close - quote_open - 1));
}

} // namespace streamkit::internal
