# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Clean, importable in-process Python API for the semantic-SLAM pipeline.

Example
-------
    from semantic_slam import SemanticSLAM

    slam = SemanticSLAM(dataset_type="replica", scene_name="room0")
    slam.push(rgb, depth, pose)              # numpy arrays
    hits = slam.query("a brown chair", top_k=5)
"""

from semantic_slam.engine import SemanticSLAM

__version__ = "0.1.0"

__all__ = ["SemanticSLAM", "__version__"]
