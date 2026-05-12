// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "test_assert.h"

#include "Backends/LiveKit/AgentStatusParser.h"

#include <cstddef>
#include <cstring>
#include <span>
#include <string>

namespace {

std::span<const std::byte> bytes_of(const char* s) {
    return std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(s), std::strlen(s));
}

}  // namespace

int main() {
    using streamkit::internal::ExtractAgentStatus;

    // Canonical payload shipped by xr-ai-pipecat's set_status:
    {
        auto r = ExtractAgentStatus(bytes_of(R"({"status": "idle"})"));
        SK_EXPECT(r.has_value());
        SK_EXPECT_EQ(*r, std::string("idle"));
    }
    {
        auto r = ExtractAgentStatus(bytes_of(R"({"status":"processing"})"));
        SK_EXPECT(r.has_value());
        SK_EXPECT_EQ(*r, std::string("processing"));
    }

    // Missing key → nullopt
    {
        auto r = ExtractAgentStatus(bytes_of(R"({"other": "x"})"));
        SK_EXPECT(!r.has_value());
    }

    // Truncated payload — no closing quote.
    {
        auto r = ExtractAgentStatus(bytes_of(R"({"status": "idle)"));
        SK_EXPECT(!r.has_value());
    }

    // Empty value is a valid match; HandleDataReceived skips empty status
    // separately so the parser itself just reports what it saw.
    {
        auto r = ExtractAgentStatus(bytes_of(R"({"status": ""})"));
        SK_EXPECT(r.has_value());
        SK_EXPECT_EQ(*r, std::string(""));
    }

    // Empty payload
    {
        auto r = ExtractAgentStatus(bytes_of(""));
        SK_EXPECT(!r.has_value());
    }

    return 0;
}
