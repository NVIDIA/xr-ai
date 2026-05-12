// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/*
 * Tiny test assertion macros. Each test executable runs its checks in main()
 * and exits non-zero on the first failure. Keeps the test-runtime dependency
 * surface at zero (no GoogleTest, no Catch2) and the per-test wiring trivial.
 *
 * Matches the minimalism of the Swift StreamKitTests scaffold.
 */

#include <cstdio>
#include <cstdlib>

#define SK_EXPECT(cond)                                                      \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::fprintf(stderr, "%s:%d: EXPECT failed: %s\n",               \
                         __FILE__, __LINE__, #cond);                         \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

#define SK_EXPECT_EQ(a, b)                                                   \
    do {                                                                     \
        auto _sk_a = (a);                                                    \
        auto _sk_b = (b);                                                    \
        if (!(_sk_a == _sk_b)) {                                             \
            std::fprintf(stderr, "%s:%d: EXPECT_EQ failed: %s != %s\n",      \
                         __FILE__, __LINE__, #a, #b);                        \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)
