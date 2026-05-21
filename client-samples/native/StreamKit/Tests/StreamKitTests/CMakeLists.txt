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
        # AgentStatusParser.h lives next to LiveKitBackend.cpp in StreamKit/src.
        # Tests need that path so the parser is reachable as
        # "Backends/LiveKit/AgentStatusParser.h".
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
