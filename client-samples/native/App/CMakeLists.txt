# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(streamkit_sample App/main.cpp)

target_link_libraries(streamkit_sample PRIVATE streamkit)

target_compile_features(streamkit_sample PRIVATE cxx_std_20)

# Copy the binary next to the source tree so it's easy to run from the repo.
set_target_properties(streamkit_sample PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
