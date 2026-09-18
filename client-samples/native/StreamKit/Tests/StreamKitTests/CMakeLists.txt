# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Each test is a standalone executable that asserts in main() and exits
# non-zero on failure. No third-party test framework — keeps the
# contribution drag-free.

set(_streamkit_test_root "${CMAKE_CURRENT_LIST_DIR}")

function(_streamkit_add_test name source)
    add_executable(${name} "${source}")
    target_link_libraries(${name} PRIVATE streamkit)
    target_include_directories(${name} PRIVATE
        "${_streamkit_test_root}"
        # Internal transport and parser headers live under StreamKit/src.
        "${CMAKE_CURRENT_SOURCE_DIR}/../../src"
    )
    add_test(NAME ${name} COMMAND ${name})
endfunction()

_streamkit_add_test(streamkit_mapping_tests          MappingTests.cpp)
_streamkit_add_test(streamkit_agent_status_tests     AgentStatusTests.cpp)
_streamkit_add_test(streamkit_frame_sink_tests       FrameSinkTests.cpp)
_streamkit_add_test(streamkit_audio_sink_tests       AudioSinkTests.cpp)
_streamkit_add_test(streamkit_session_tests          StreamSessionTests.cpp)
_streamkit_add_test(streamkit_livekit_backend_tests  LiveKitBackendTests.cpp)
_streamkit_add_test(streamkit_byte_stream_tests      ByteStreamTransportTests.cpp)
