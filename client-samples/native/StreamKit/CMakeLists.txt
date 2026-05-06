# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ─────────────────────────────────────────────────────────────────────────────
# StreamKit static library
# ─────────────────────────────────────────────────────────────────────────────

add_library(streamkit STATIC
    src/StreamSession.cpp
    src/Backends/LiveKit/LiveKitBackend.cpp
)

target_include_directories(streamkit
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(streamkit PUBLIC cxx_std_20)

# ── LiveKit dependency ────────────────────────────────────────────────────────
#
# Replace the placeholder block below with a real livekit-ffi dependency once
# you have chosen an integration path (see README).
#
# Path A — livekit-ffi pre-built:
#   Set LIVEKIT_FFI_DIR to the directory containing livekit_ffi.h and the
#   built library. Then uncomment:
#
#     find_library(LIVEKIT_FFI_LIB livekit_ffi
#         PATHS "${LIVEKIT_FFI_DIR}/lib" REQUIRED)
#     target_include_directories(streamkit PRIVATE "${LIVEKIT_FFI_DIR}/include")
#     target_link_libraries(streamkit PRIVATE "${LIVEKIT_FFI_LIB}")
#
# Path B — livekit-ffi via FetchContent (builds Rust crate, requires Cargo):
#
#   include(FetchContent)
#   FetchContent_Declare(
#       livekit_ffi
#       GIT_REPOSITORY https://github.com/livekit/rust-sdks.git
#       GIT_TAG        main   # pin to a release tag in production
#       SOURCE_SUBDIR  livekit-ffi
#   )
#   FetchContent_MakeAvailable(livekit_ffi)
#   target_link_libraries(streamkit PRIVATE livekit_ffi)
#
# Until then, the LiveKitBackend compiles as a stub without any extra deps.

message(STATUS
    "[streamkit] LiveKit dependency not configured. "
    "LiveKitBackend will compile as a stub. "
    "See client-samples/native/StreamKit/CMakeLists.txt for integration instructions.")
