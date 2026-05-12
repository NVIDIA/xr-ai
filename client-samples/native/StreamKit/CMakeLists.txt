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

# ─────────────────────────────────────────────────────────────────────────────
# LiveKit C++ SDK dependency
# ─────────────────────────────────────────────────────────────────────────────
#
# LiveKitBackend talks to `livekit::Room` and friends from the upstream LiveKit
# C++ SDK (https://github.com/livekit/rust-sdks → `cpp/`). The SDK ships as a
# tree containing `inc/livekit/*.h` headers and `lib/liblivekit{,_ffi}.so` plus
# a vendored protobuf runtime.
#
# Two ways to supply it:
#
#   1. Set `-DLIVEKIT_SDK_ROOT=/path/to/sdk` on the cmake command line.
#      The path must contain `inc/livekit/room.h` and the corresponding lib
#      under either `lib/` or `lib/<ABI>/`.
#
#   2. Drop the SDK at `client-samples/native/third_party/livekit-cpp-sdk/`
#      with the same `inc/` + `lib/` layout, and the build picks it up
#      without any extra flags.
#
# When neither path resolves, `STREAMKIT_HAVE_LIVEKIT` stays OFF and the
# backend compiles in stub mode (Connect succeeds locally but never opens a
# real WebRTC session). This keeps CI green on machines without the SDK and
# matches the prior placeholder behaviour.

set(LIVEKIT_SDK_ROOT "" CACHE PATH "Path to a LiveKit C++ SDK install (containing inc/ and lib/)")
set(LIVEKIT_LIB_DIR  "" CACHE PATH "Override path to the LiveKit C++ SDK lib dir (defaults to LIVEKIT_SDK_ROOT/lib)")

set(_streamkit_default_sdk
    "${CMAKE_CURRENT_LIST_DIR}/../third_party/livekit-cpp-sdk")
if(NOT LIVEKIT_SDK_ROOT AND EXISTS "${_streamkit_default_sdk}/inc/livekit/room.h")
    set(LIVEKIT_SDK_ROOT "${_streamkit_default_sdk}")
endif()

set(STREAMKIT_HAVE_LIVEKIT OFF)

if(LIVEKIT_SDK_ROOT)
    if(NOT EXISTS "${LIVEKIT_SDK_ROOT}/inc/livekit/room.h")
        message(FATAL_ERROR
            "[streamkit] LIVEKIT_SDK_ROOT='${LIVEKIT_SDK_ROOT}' does not "
            "contain inc/livekit/room.h. Point it at a LiveKit C++ SDK "
            "tree with inc/ and lib/ subdirectories.")
    endif()

    if(NOT LIVEKIT_LIB_DIR)
        set(LIVEKIT_LIB_DIR "${LIVEKIT_SDK_ROOT}/lib")
    endif()

    # The main library has a hard DT_NEEDED on liblivekit_ffi.so and the
    # vendored protobuf — find them so the linker is happy on platforms
    # that don't carry them in /usr/lib.
    find_library(LIVEKIT_LIB_MAIN
        NAMES livekit
        PATHS "${LIVEKIT_LIB_DIR}" "${LIVEKIT_LIB_DIR}/${ANDROID_ABI}"
        NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)

    find_library(LIVEKIT_LIB_FFI
        NAMES livekit_ffi
        PATHS "${LIVEKIT_LIB_DIR}" "${LIVEKIT_LIB_DIR}/${ANDROID_ABI}"
        NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)

    find_library(LIVEKIT_LIB_PROTOBUF
        NAMES protobuf-cpp-full-vendor-3.9.1 protobuf
        PATHS "${LIVEKIT_LIB_DIR}" "${LIVEKIT_LIB_DIR}/${ANDROID_ABI}"
        NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)

    target_include_directories(streamkit PRIVATE "${LIVEKIT_SDK_ROOT}/inc")
    if(LIVEKIT_LIB_MAIN)
        target_link_libraries(streamkit PUBLIC "${LIVEKIT_LIB_MAIN}")
    else()
        message(WARNING
            "[streamkit] liblivekit not found under '${LIVEKIT_LIB_DIR}'. "
            "Header-only build will compile but cannot link the sample.")
    endif()
    if(LIVEKIT_LIB_FFI)
        target_link_libraries(streamkit PUBLIC "${LIVEKIT_LIB_FFI}")
    endif()
    if(LIVEKIT_LIB_PROTOBUF)
        target_link_libraries(streamkit PUBLIC "${LIVEKIT_LIB_PROTOBUF}")
    endif()
    target_compile_definitions(streamkit PRIVATE STREAMKIT_HAVE_LIVEKIT=1)
    set(STREAMKIT_HAVE_LIVEKIT ON)

    message(STATUS "[streamkit] LiveKit C++ SDK: ${LIVEKIT_SDK_ROOT}")
    message(STATUS "[streamkit]   main:     ${LIVEKIT_LIB_MAIN}")
    message(STATUS "[streamkit]   ffi:      ${LIVEKIT_LIB_FFI}")
    message(STATUS "[streamkit]   protobuf: ${LIVEKIT_LIB_PROTOBUF}")
else()
    message(STATUS
        "[streamkit] LIVEKIT_SDK_ROOT not set. LiveKitBackend compiles in "
        "stub mode — Connect() reports kConnected immediately without "
        "opening a WebRTC session. Pass -DLIVEKIT_SDK_ROOT=/path/to/sdk to "
        "enable the real backend.")
endif()

# ─────────────────────────────────────────────────────────────────────────────
# Unit tests
# ─────────────────────────────────────────────────────────────────────────────
option(STREAMKIT_BUILD_TESTS "Build StreamKit unit tests" OFF)
if(STREAMKIT_BUILD_TESTS)
    add_subdirectory(Tests)
endif()
