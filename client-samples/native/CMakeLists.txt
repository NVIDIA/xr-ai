# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.21)

project(StreamKitNative
    VERSION     0.1.0
    DESCRIPTION "StreamKit — native C++ client for the xr-ai streaming platform"
    LANGUAGES   CXX
)

# C++20 is required for std::span.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─────────────────────────────────────────────────────────────────────────────
# Build options
# ─────────────────────────────────────────────────────────────────────────────

option(STREAMKIT_BUILD_SAMPLE
    "Build the sample application in App/" ON)

# ─────────────────────────────────────────────────────────────────────────────
# Compiler defaults
# ─────────────────────────────────────────────────────────────────────────────

# Warn loudly without being pedantic about platform headers.
if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()

# ─────────────────────────────────────────────────────────────────────────────
# StreamKit library
# ─────────────────────────────────────────────────────────────────────────────

add_subdirectory(StreamKit)

# ─────────────────────────────────────────────────────────────────────────────
# Sample application
# ─────────────────────────────────────────────────────────────────────────────

if(STREAMKIT_BUILD_SAMPLE)
    include(App/CMakeLists.txt)
endif()

# ─────────────────────────────────────────────────────────────────────────────
# Quick-start
# ─────────────────────────────────────────────────────────────────────────────
#
#   mkdir build && cd build
#   cmake ..
#   cmake --build . --parallel
#   ./bin/streamkit_sample --host 192.168.1.100 --token <jwt>
