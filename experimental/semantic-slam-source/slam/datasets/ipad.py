# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import abc
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from natsort import natsorted
from scipy.spatial.transform import Rotation as R

from gradslam.datasets import datautils
from gradslam.geometry.geometryutils import relative_transformation
from gradslam.slam.pointfusion import PointFusion
from gradslam.structures.rgbdimages import RGBDImages
from PIL import Image

from slam.utils.general_utils import to_scalar, measure_time

DEBUG = False

def from_intrinsics_matrix(K: torch.Tensor) -> tuple[float, float, float, float]:
    '''
    Get fx, fy, cx, cy from the intrinsics matrix
    
    return 4 scalars
    '''
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

class IPADDataset(torch.utils.data.Dataset):
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
        # self.poses = self.load_poses()

        # self.transformed_poses = datautils.poses_to_transforms(self.poses)
        # self.poses = torch.stack(self.poses)
        # if self.relative_pose:
        #     self.transformed_poses = self._preprocess_poses(self.poses)
        # else:
        #     self.transformed_poses = self.poses
            
        self.K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        self.K = torch.from_numpy(self.K)
        self.K = datautils.scale_intrinsics(
            self.K, self.height_downsample_ratio, self.width_downsample_ratio
        )
        self.intrinsics = torch.eye(4).to(self.K)
        self.intrinsics[:3, :3] = self.K

    def __len__(self):
        raise NotImplementedError
        # return self.num_imgs

    # def get_filepaths(self):
    #     """Return paths to color images, depth images. Implement in subclass."""
    #     raise NotImplementedError

    def load_poses(self, pose):
        """Load camera poses. Implement in subclass."""
        # poses = []
        # with open(self.pose_path, "r") as f:
            # lines = f.readlines()
        # for i in range(self.num_imgs):
        #     line = lines[i]
        
            # This is weird but it works - ARKit gives us Camera to World already in a Right hand coordinate system. But we need it in a left hand coordinate system and that's why we negate the y and z rotation axes.
            #  What is weird is that we have to invert the ARKit pose. ARKit already gives us camera2world, by inverting it, we create a world to camera pose, but this works somehow.
            # c2w = np.array(list(map(float, line.split()))).reshape(4, 4)        # OLD CODE
            
            
        # RAHUL FOLLIWNG CODE is commented for debugginhg ----------------
        # line = pose.split(',')[1:]
        # line = " ".join(line)
        # ----------------
        if DEBUG:
            line = pose
        else:
            # line = pose.split(' ')[1:]
            # line = " ".join(line)
            line = pose[1:]
        c2w = np.linalg.inv(np.array(line).reshape(4, 4))
        c2w[:3, 1] *= -1
        c2w[:3, 2] *= -1
        c2w = torch.from_numpy(c2w).float()
        return c2w

    def _preprocess_color(self, color: np.ndarray):
        r"""Preprocesses the color image by resizing to :math:`(H, W, C)`, (optionally) normalizing values to
        :math:`[0, 1]`, and (optionally) using channels first :math:`(C, H, W)` representation.

        Args:
            color (np.ndarray): Raw input rgb image

        Retruns:
            np.ndarray: Preprocessed rgb image

        Shape:
            - Input: :math:`(H_\text{old}, W_\text{old}, C)`
            - Output: :math:`(H, W, C)` if `self.channels_first == False`, else :math:`(C, H, W)`.
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

    def _preprocess_depth(self, depth: np.ndarray):
        r"""Preprocesses the depth image by resizing, adding channel dimension, and scaling values to meters. Optionally
        converts depth from channels last :math:`(H, W, 1)` to channels first :math:`(1, H, W)` representation.

        Args:
            depth (np.ndarray): Raw depth image

        Returns:
            np.ndarray: Preprocessed depth

        Shape:
            - depth: :math:`(H_\text{old}, W_\text{old})`
            - Output: :math:`(H, W, 1)` if `self.channels_first == False`, else :math:`(1, H, W)`.
        """
        depth = cv2.resize(
            depth.astype(float),
            (self.desired_width, self.desired_height),
            interpolation=cv2.INTER_NEAREST,
        )
        depth = np.expand_dims(depth, -1)

        return depth / self.png_depth_scale
    
    def _preprocess_poses(self, poses: torch.Tensor):
        r"""Preprocesses the poses by setting first pose in a sequence to identity and computing the relative
        homogenous transformation for all other poses.

        Args:
            poses (torch.Tensor): Pose matrices to be preprocessed

        Returns:
            Output (torch.Tensor): Preprocessed poses

        Shape:
            - poses: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
            - Output: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
        """
        return relative_transformation(
            poses[0].unsqueeze(0).repeat(poses.shape[0], 1, 1),
            poses,
            orthogonal_rotations=False,
        )
        
    def get_cam_K(self):
        '''
        Return camera intrinsics matrix K
        
        Returns:
            K (torch.Tensor): Camera intrinsics matrix, of shape (3, 3)
        '''
        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        K = torch.from_numpy(K)
        return K
    
    # def read_embedding_from_file(self, embedding_path: str):
    #     '''
    #     Read embedding from file and process it. To be implemented in subclass for each dataset separately.
    #     '''
    #     raise NotImplementedError

    def __getitem__(self, color, depth, pose):

        # color = np.asarray(np.array(Image.open(color_path)), dtype=float)
        color = self._preprocess_color(color)
        color = torch.from_numpy(color)
        # if ".png" in depth_path:
        if type(depth) != np.ndarray:
            # depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depth = np.asarray(np.array(Image.open(depth)), dtype=np.int64)
        elif type(depth) == np.ndarray:
            # print(depth_path)
            # depth = np.loadtxt(depth_path, dtype=np.float32)
            # depth = depth.reshape((144,256))
            depth = depth
        else:
            raise NotImplementedError

        # self.K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        # self.K = torch.from_numpy(self.K)
        if self.distortion is not None:
            # undistortion is only applied on color image, not depth!
            color = cv2.undistort(color, self.K, self.distortion)

        depth = self._preprocess_depth(depth)
        depth = torch.from_numpy(depth)

        
        pose = self.load_poses(pose)
        # pose = self.transformed_poses[index]

        return (
            color.to(self.device).type(self.dtype),
            depth.to(self.device).type(self.dtype),
            self.intrinsics.to(self.device).type(self.dtype),
            pose.to(self.device).type(self.dtype),
            # self.retained_inds[index].item(),
        )
        
    def getItems(self, color, depth, pose):
        
        color = self._preprocess_color(color)
        color = torch.from_numpy(color)
        # if ".png" in depth_path:
        if type(depth) != np.ndarray:
            # depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depth = np.asarray(np.array(Image.open(depth)), dtype=np.int64)
        elif type(depth) == np.ndarray:
            # print(depth_path)
            # depth = np.loadtxt(depth_path, dtype=np.float32)
            print(f"Depth shape: {depth.shape}")
            depth = depth.reshape((144,256))
            print(f"Depth shape after reshape: {depth.shape}")  
            depth = depth
        else:
            raise NotImplementedError

        if self.distortion is not None:
            # undistortion is only applied on color image, not depth!
            color = cv2.undistort(color, self.K, self.distortion)

        depth = self._preprocess_depth(depth)
        depth = torch.from_numpy(depth)
        print(f"Depth shape after preprocess: {depth.shape}")


        pose = self.load_poses(pose)
        # pose = self.transformed_poses[index]

        return (
            color.to(self.device).type(self.dtype),
            depth.to(self.device).type(self.dtype),
            self.intrinsics.to(self.device).type(self.dtype),
            pose.to(self.device).type(self.dtype),
            # self.retained_inds[index].item(),
        )