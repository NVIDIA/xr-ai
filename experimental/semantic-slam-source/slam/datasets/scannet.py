# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ScanNet dataset implementation for semantic SLAM operations."""

from logging import raiseExceptions
import os
import glob
import numpy as np
import torch
from natsort import natsorted
from typing import Optional

from .base import BaseDataset, from_intrinsics_matrix, as_intrinsics_matrix


class ScanNetDataset(BaseDataset):
    """ScanNet dataset implementation.
    
    ScanNet datasets have the following structure:
    - color/: RGB images (.jpg)
    - depth/: Depth images (.png)
    - pose/: Camera poses (.txt files)
    - intrinsic/: Camera intrinsics (intrinsic_color.txt)
    """
    
    def __init__(
        self,
        config_dict,
        basedir=None,
        sequence=None,
        stride: Optional[int] = 1,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 968,
        desired_width: Optional[int] = 1296,
        channels_first: bool = False,
        normalize_color: bool = False,
        device="cpu",
        dtype=torch.float,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        relative_pose: bool = False,
        **kwargs,
            ):
        
        basedir = os.environ.get('SCANNET_ROOT')


        # Set up paths if provided
        if basedir and sequence:
            try:
                self.input_folder = os.path.join(basedir, sequence)          
                # Load the intrinsic matrix from the file in each scene
                scene_intrinsic_path = os.path.join(self.input_folder, "intrinsic", "intrinsic_color.txt")
                scene_intrinsic = np.loadtxt(scene_intrinsic_path)
                # Convert numpy values to native Python floats for OmegaConf compatibility
                config_dict['camera_params']['fx'] = float(scene_intrinsic[0, 0])
                config_dict['camera_params']['fy'] = float(scene_intrinsic[1, 1])
                config_dict['camera_params']['cx'] = float(scene_intrinsic[0, 2])
                config_dict['camera_params']['cy'] = float(scene_intrinsic[1, 2])
            except Exception as e:
                raise ValueError(f"Intrinsic matrix not found at {scene_intrinsic_path} or {e}")
        else:
            self.input_folder = None
        
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
            **kwargs
        )

    def __len__(self):
        # For ScanNet, we don't preload file paths, so return NotImplemented
        # This will be used for streaming/online processing
        raise NotImplementedError("")

    def load_poses(self, pose):
        """Load camera poses for ScanNet dataset.
        
        ScanNet poses are typically 4x4 transformation matrices.
        """
        if isinstance(pose, str):
            # If pose is a file path, load from file
            pose_matrix = np.loadtxt(pose)
        elif isinstance(pose, (list, np.ndarray)):
            # If pose is already a matrix or list, convert to numpy array
            pose_matrix = np.array(pose)
            if pose_matrix.size == 16:
                pose_matrix = pose_matrix.reshape(4, 4)
            elif pose_matrix.size == 17:
                pose_matrix = pose_matrix[1:]
                pose_matrix = pose_matrix.reshape(4, 4)
            else:
                raise ValueError(f"Length of pose is not 16: {pose.shape}")
        else:
            raise ValueError(f"Unsupported pose type: {type(pose)}")
        
        # Convert to torch tensor
        c2w = torch.from_numpy(pose_matrix).float()
        return c2w

    def get_filepaths(self):
        """Return paths to color images, depth images, and embeddings.
        
        This method is used when the dataset is used in local mode.
        """
        if self.input_folder is None:
            raise ValueError("input_folder not set. Provide basedir and sequence in constructor.")
            
        color_paths = natsorted(glob.glob(f"{self.input_folder}/color/*.jpg"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        embedding_paths = None
        
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
            
        return color_paths, depth_paths, embedding_paths

    def load_poses_from_files(self):
        """Load all poses from pose files.
        
        This method is used when the dataset is used in local mode.
        """
        if self.input_folder is None:
            raise ValueError("input_folder not set. Provide basedir and sequence in constructor.")
            
        poses = []
        posefiles = natsorted(glob.glob(f"{self.input_folder}/pose/*.txt"))
        for posefile in posefiles:
            _pose = torch.from_numpy(np.loadtxt(posefile))
            poses.append(_pose)
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        """Read embedding from file for ScanNet dataset."""
        embedding = torch.load(embedding_file_path, map_location="cpu")
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)