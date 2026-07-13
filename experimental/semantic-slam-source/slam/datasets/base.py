# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base dataset class for semantic SLAM operations."""

import abc
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional

from PIL import Image

from gradslam.datasets import datautils
from gradslam.geometry.geometryutils import relative_transformation

from slam.utils.general_utils import to_scalar, measure_time


def from_intrinsics_matrix(K: torch.Tensor) -> tuple[float, float, float, float]:
    """
    Get fx, fy, cx, cy from the intrinsics matrix
    
    return 4 scalars
    """
    fx = to_scalar(K[0, 0])
    fy = to_scalar(K[1, 1])
    cx = to_scalar(K[0, 2])
    cy = to_scalar(K[1, 2])
    return fx, fy, cx, cy


def as_intrinsics_matrix(intrinsics):
    """
    Get matrix representation of intrinsics.
    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K


class BaseDataset(torch.utils.data.Dataset):
    """Base dataset class for semantic SLAM operations.
    
    This class provides common functionality for all dataset types including:
    - Image and depth preprocessing
    - Camera intrinsics handling
    - Pose processing
    - Common configuration management
    """
    
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
        relative_pose: bool = False,  # If True, the pose is relative to the first frame
        test_depth_downsampling: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.name = config_dict["dataset_name"]
        self.device = device
        self.png_depth_scale = config_dict["camera_params"]["png_depth_scale"]

        self.orig_height = config_dict["camera_params"]["image_height"]
        self.orig_width = config_dict["camera_params"]["image_width"]
        self.fx = config_dict["camera_params"]["fx"]
        self.fy = config_dict["camera_params"]["fy"]
        self.cx = config_dict["camera_params"]["cx"]
        self.cy = config_dict["camera_params"]["cy"]

        self.dtype = dtype

        self.desired_height = desired_height
        self.desired_width = desired_width
        self.height_downsample_ratio = float(self.desired_height) / self.orig_height
        self.width_downsample_ratio = float(self.desired_width) / self.orig_width
        self.channels_first = channels_first
        self.normalize_color = normalize_color

        self.relative_pose = relative_pose
        self.distortion = None
        self.test_depth_downsampling = test_depth_downsampling
        # Setup camera intrinsics
        self.K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        self.K = torch.from_numpy(self.K)
        self.K = datautils.scale_intrinsics(
            self.K, self.height_downsample_ratio, self.width_downsample_ratio
        )
        self.intrinsics = torch.eye(4).to(self.K)
        self.intrinsics[:3, :3] = self.K

    def __len__(self):
        raise NotImplementedError("Subclasses must implement __len__")

    @abc.abstractmethod
    def load_poses(self, pose):
        """Load camera poses. Must be implemented in subclass."""
        raise NotImplementedError("Subclasses must implement load_poses")

    def _preprocess_color(self, color: np.ndarray):
        """Preprocesses the color image by resizing to (H, W, C), (optionally) normalizing values to
        [0, 1], and (optionally) using channels first (C, H, W) representation.

        Args:
            color (np.ndarray): Raw input rgb image

        Returns:
            np.ndarray: Preprocessed rgb image

        Shape:
            - Input: (H_old, W_old, C)
            - Output: (H, W, C) if `self.channels_first == False`, else (C, H, W).
        """
        color = cv2.resize(
            color,
            (self.desired_width, self.desired_height),
            interpolation=cv2.INTER_LINEAR,
        )
        if self.normalize_color:
            color = datautils.normalize_image(color)
        if self.channels_first:
            color = datautils.channels_first(color)
        return color

    def _downsample_depth(self, depth: np.ndarray):
        """Downsamples the depth image by the given downsampling factor."""
        return cv2.resize(
            depth.astype(float),
            (self.desired_width // self.test_depth_downsampling, self.desired_height // self.test_depth_downsampling),
            interpolation=cv2.INTER_NEAREST,
        )
    def _preprocess_depth(self, depth: np.ndarray):
        """Preprocesses the depth image by resizing, adding channel dimension, and scaling values to meters. Optionally
        converts depth from channels last (H, W, 1) to channels first (1, H, W) representation.

        Args:
            depth (np.ndarray): Raw depth image

        Returns:
            np.ndarray: Preprocessed depth

        Shape:
            - depth: (H_old, W_old)
            - Output: (H, W, 1) if `self.channels_first == False`, else (1, H, W).
        """
        if self.test_depth_downsampling > 1:
            depth = self._downsample_depth(depth)

        depth = cv2.resize(
            depth.astype(float),
            (self.desired_width, self.desired_height),
            interpolation=cv2.INTER_NEAREST,
        )
        depth = np.expand_dims(depth, -1)

        return depth / self.png_depth_scale
    
    def _preprocess_poses(self, poses: torch.Tensor):
        """Preprocesses the poses by setting first pose in a sequence to identity and computing the relative
        homogenous transformation for all other poses.

        Args:
            poses (torch.Tensor): Pose matrices to be preprocessed

        Returns:
            Output (torch.Tensor): Preprocessed poses

        Shape:
            - poses: (L, 4, 4) where L denotes sequence length.
            - Output: (L, 4, 4) where L denotes sequence length.
        """
        return relative_transformation(
            poses[0].unsqueeze(0).repeat(poses.shape[0], 1, 1),
            poses,
            orthogonal_rotations=False,
        )
        
    def get_cam_K(self):
        """
        Return camera intrinsics matrix K
        
        Returns:
            K (torch.Tensor): Camera intrinsics matrix, of shape (3, 3)
        """
        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        K = torch.from_numpy(K)
        return K

    def __getitem__(self, color, depth, pose):
        """Process color, depth, and pose data for a single frame.
        
        Args:
            color: Color image (PIL Image or numpy array)
            depth: Depth data (numpy array or path)
            pose: Camera pose data
            
        Returns:
            Tuple of (color_tensor, depth_tensor, intrinsics, pose_tensor)
        """
        # Preprocess color
        color = self._preprocess_color(color)
        color = torch.from_numpy(color)
        
        # Preprocess depth
        if type(depth) != np.ndarray:
            depth = np.asarray(Image.open(depth), dtype=np.int64)
        elif type(depth) == np.ndarray:
            depth = depth
        else:
            raise NotImplementedError(f"Unsupported depth type: {type(depth)}")

        if self.distortion is not None:
            # undistortion is only applied on color image, not depth!
            color = cv2.undistort(color, self.K, self.distortion)

        depth = self._preprocess_depth(depth)
        depth = torch.from_numpy(depth)

        # Process pose
        pose = self.load_poses(pose)

        return (
            color.to(self.device).type(self.dtype),
            depth.to(self.device).type(self.dtype),
            self.intrinsics.to(self.device).type(self.dtype),
            pose.to(self.device).type(self.dtype),
        )
        
    def getItems(self, color, depth, pose):
        """Alias for __getitem__ for backward compatibility."""
        return self.__getitem__(color, depth, pose)
