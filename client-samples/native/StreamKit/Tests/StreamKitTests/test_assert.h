// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/*
 * Tiny test assertion helpers. Each test executable runs its checks in main()
 * and exits non-zero on the first failure. Keeps the test-runtime dependency
 * surface at zero (no GoogleTest, no Catch2) and the per-test wiring trivial.
 *
 * Matches the minimalism of the Swift StreamKitTests scaffold.
 */

#include <cstdio>
#include <cstdlib>
#include <source_location>

namespace streamkit::test {

inline void Expect(bool condition,
                   std::source_location where = std::source_location::current()) {
    if (!condition) {
        std::fprintf(stderr, "%s:%u: EXPECT failed\n",
                     where.file_name(), where.line());
        std::exit(1);
    }
}

template <typename A, typename B>
void ExpectEq(const A& a,
              const B& b,
              std::source_location where = std::source_location::current()) {
    if (!(a == b)) {
        std::fprintf(stderr, "%s:%u: EXPECT_EQ failed\n",
                     where.file_name(), where.line());
        std::exit(1);
    }
}

}  // namespace streamkit::test
