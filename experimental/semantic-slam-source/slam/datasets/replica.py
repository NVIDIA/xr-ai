# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
from typing import Optional

from .base import BaseDataset, from_intrinsics_matrix, as_intrinsics_matrix

DEBUG = False


class ReplicaDataset(BaseDataset):
    def __init__(
        self,
        config_dict,
        stride: Optional[int] = 1,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: int = 480,
        desired_width: int = 640,
        channels_first: bool = False,
        normalize_color: bool = False,
        device="cpu",
        dtype=torch.float,
        load_embeddings: bool = False,
        embedding_dir: str = "feat_lseg_240_320",
        embedding_dim: int = 512,
        relative_pose: bool = False, # If True, the pose is relative to the first frame
        test_depth_downsampling: int = 1,           # 1 - no downsampling, n - downsampling by n
        **kwargs,
    ):
        super().__init__(
            config_dict=config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            channels_first=channels_first,
            normalize_color=normalize_color,
            device=device,
            dtype=dtype,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            relative_pose=relative_pose,
            test_depth_downsampling=test_depth_downsampling,
            **kwargs
        )

    def __len__(self):
        raise NotImplementedError
        # return self.num_imgs

    def load_poses(self, pose):
        """Load camera poses for Replica dataset."""
        if DEBUG:
            line = pose
        else:
            # For Replica, pose comes as [frame_number, pose_matrix_elements...]
            # We skip the frame number and take the pose matrix elements
            line = pose[1:]
        c2w = np.array(line).reshape(4, 4)
        c2w = torch.from_numpy(c2w).float()
        return c2w
