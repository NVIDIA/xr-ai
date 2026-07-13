# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Main server package for XR Scene Builder.

This package contains the gRPC server implementation and related components.
"""

# Make protobuf modules available at package level for internal imports
import os
import sys

# Temporarily add this directory to path for protobuf imports
_this_dir = os.path.dirname(__file__)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

try:
    # Import protobuf modules
    import xr_service_pb2
    import xr_service_pb2_grpc
    
    # Make them available at package level
    globals()['xr_service_pb2'] = xr_service_pb2
    globals()['xr_service_pb2_grpc'] = xr_service_pb2_grpc
    
    __all__ = ['xr_service_pb2', 'xr_service_pb2_grpc']
    
finally:
    # Clean up sys.path
    if _this_dir in sys.path:
        sys.path.remove(_this_dir)