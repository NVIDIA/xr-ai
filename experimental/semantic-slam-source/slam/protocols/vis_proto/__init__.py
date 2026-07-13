# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local protocol buffer definitions for visualization."""

# Generated protobuf files use absolute imports, so we need sys.path
import os
import sys

# Add this directory to sys.path temporarily for protobuf imports
_this_dir = os.path.dirname(__file__)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

try:
    # Import the protobuf modules
    import vis_pb2
    import vis_pb2_grpc
    
    # Make them available as package attributes
    globals()['vis_pb2'] = vis_pb2
    globals()['vis_pb2_grpc'] = vis_pb2_grpc
    
finally:
    # Clean up sys.path
    if _this_dir in sys.path:
        sys.path.remove(_this_dir)

__all__ = ["vis_pb2", "vis_pb2_grpc"]