// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "test_assert.h"

#include "Backends/LiveKit/AgentStatusParser.h"

#include <cstddef>
#include <span>
#include <string>
#include <string_view>

namespace {

std::span<const std::byte> bytes_of(std::string_view s) {
    return std::as_bytes(std::span<const char>(s.data(), s.size()));
}

}  // namespace

int main() {
    using streamkit::internal::ExtractAgentStatus;
    using streamkit::test::Expect;
    using streamkit::test::ExpectEq;

    // Canonical payload shipped by xr-ai-pipecat's set_status:
    {
        auto r = ExtractAgentStatus(bytes_of(R"({"status": "idle"})"));
        Expect(r.has_value());
        ExpectEq(*r, std::string("idle"));
    }
    {
        auto r = ExtractAgentStatus(bytes_of(R"({"status":"processing"})"));
        Expect(r.has_value());
        ExpectEq(*r, std::string("processing"));
    }

    // Missing key → nullopt
    {
        auto r = ExtractAgentStatus(bytes_of(R"({"other": "x"})"));
        Expect(!r.has_value());
    }

    // Truncated payload — no closing quote.
    {
        auto r = ExtractAgentStatus(bytes_of(R"({"status": "idle)"));
        Expect(!r.has_value());
    }

    // Empty value is a valid match; HandleDataReceived skips empty status
    // separately so the parser itself just reports what it saw.
    {
        auto r = ExtractAgentStatus(bytes_of(R"({"status": ""})"));
        Expect(r.has_value());
        ExpectEq(*r, std::string(""));
    }

    // Empty payload
    {
        auto r = ExtractAgentStatus(bytes_of(""));
        Expect(!r.has_value());
    }

    return 0;
}
