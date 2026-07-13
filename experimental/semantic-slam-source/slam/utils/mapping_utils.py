# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mapping support utilities for semantic SLAM operations."""

import os
import time
from pathlib import Path

import numpy as np
import omegaconf
import torch
from omegaconf import DictConfig
from PIL import Image, ImageDraw, ImageFont

from slam.datasets import ipad, replica, scannet


from slam.core.utils import (
    gobs_to_detection_list,
    gobs_to_detection_list_optimized,   
)

BG_CLASSES = ["wall", "floor", "ceiling"]
ASYNC_IO=True
DEBUG_PRINT = False
def debug_print(*args, **kwargs):
    if DEBUG_PRINT:
        print(*args, **kwargs)


def get_dataset(datasetClass, config_dict, desired_height, desired_width, device, dtype, scene_name=None, test_depth_downsampling=1):
    if datasetClass.lower() == "ipad":
        dataset = ipad.IPADDataset(
            config_dict = config_dict,
            desired_height = desired_height,
            desired_width = desired_width,
            device = device,
            dtype=dtype,
            test_depth_downsampling=test_depth_downsampling,
        )
    elif datasetClass.lower() == "replica":
        dataset = replica.ReplicaDataset(
            config_dict = config_dict,
            desired_height = desired_height,
            desired_width = desired_width,
            device = device,
            dtype=dtype,
            test_depth_downsampling=test_depth_downsampling,
        )
    elif datasetClass.lower() == "scannet":
        # Get ScanNet root directory from environment
        scannet_root = os.environ.get('SCANNET_ROOT')
        
        # If we have both scene name and root, and scene name looks like a real scene,
        # then set up for file-based loading to get proper intrinsics
        if (scene_name and scannet_root and 
            scene_name not in ['scannet', 'replica', 'ipad']):
            # For ScanNet scenes, the structure is: SCANNET_ROOT/scene_name
            # This matches the logic in factory.py _get_scannet_paths
            sequence = f"{scene_name}"
            
            dataset = scannet.ScanNetDataset(
                config_dict = config_dict,
                basedir = scannet_root,
                sequence = sequence,
                desired_height = desired_height,
                desired_width = desired_width,
                device = device,
                dtype=dtype,
            )
        else:
            # For cases without proper scene name, create dataset without file paths
            dataset = scannet.ScanNetDataset(
                config_dict = config_dict,
                basedir = None,
                sequence = None,
                desired_height = desired_height,
                desired_width = desired_width,
                device = device,
                dtype=dtype,
            )
    else:
        raise NotImplementedError(f"Dataset class {datasetClass} not supported")
    return dataset


def process_cfg(cfg: DictConfig, useDetector, datasetClass):
    # cfg.dataset_root = Path(cfg.dataset_root)
    if datasetClass.lower() == "replica":
        cfg.dataset_config = "./REPLICA.yaml"
    elif datasetClass.lower() == "scannet":
        cfg.dataset_config = "./ScanNet.yaml"
        
    cfg.dataset_config = Path(cfg.dataset_config)
    
    if cfg.dataset_config.name != "multiscan.yaml":
        # For datasets whose depth and RGB have the same resolution
        # Set the desired image heights and width from the dataset config
        # Get path to datasets directory relative to this file
        datasets_path = os.path.join(os.path.dirname(__file__), "..", "datasets")
        cfg.dataset_config = omegaconf.OmegaConf.load(os.path.join(datasets_path, cfg.dataset_config))
        if cfg.image_height is None:
            cfg.image_height = cfg.dataset_config.camera_params.image_height
        if cfg.image_width is None:
            cfg.image_width = cfg.dataset_config.camera_params.image_width
        print(f"Setting image height and width to {cfg.image_height} x {cfg.image_width}")
        if useDetector:
            cfg.mask_conf_threshold = 0.25
            cfg.skip_bg = False
        
    else:
        # For dataset whose depth and RGB have different resolutions
        assert cfg.image_height is not None and cfg.image_width is not None, \
            "For multiscan dataset, image height and width must be specified"

    return cfg
    
# @hydra.main(version_base=None, config_path="../utilsSLAM/", config_name="base")
def setup(useDetector, datasetClass):
    # Get path to utilsSLAM directory relative to this file  
    datasets_path = os.path.join(os.path.dirname(__file__), "..", "datasets")
    cfg = omegaconf.OmegaConf.load(os.path.join(datasets_path, "base.yaml"))
    cfg = process_cfg(cfg, useDetector, datasetClass)
    return cfg


def create_pcd_parallel(image_np, depth_array, pose, frameNumber, dataset, cfg, classes, gobs, output_receiver_list, pipelined_mapping, datasetClass, time_dict):
    start = time.perf_counter_ns()  
    color_tensor, depth_tensor, intrinsics, unt_pose = dataset.getItems(image_np, depth_array, pose)
    assert not pipelined_mapping, "pipelined_mapping must be False to reach here"
        
    color_np = color_tensor.cpu().numpy() # (H, W, 3)
    image_rgb = (color_np).astype(np.uint8) # (H, W, 3)
    
    # Get the depth image
    depth_tensor = depth_tensor[..., 0]
    depth_array = depth_tensor.cpu().numpy()

    # Get the intrinsics matrix
    cam_K = intrinsics.cpu().numpy()[:3, :3]
    
    unt_pose = unt_pose.cpu().numpy()
    # Don't apply any transformation otherwise
    adjusted_pose = unt_pose
    
    
    fg_detection_list, bg_detection_list, idx_to_keep = gobs_to_detection_list_optimized(
        cfg = cfg,
        image = image_rgb,
        depth_array = depth_array,
        cam_K = cam_K,
        idx = frameNumber,
        gobs = gobs,
        trans_pose = adjusted_pose,
        class_names = classes,
        BG_CLASSES = BG_CLASSES,
        color_path = None,
        pipelined_mapping=pipelined_mapping,
        dataset_type=datasetClass,
        time_dict=time_dict,
    )
    output_receiver_list.append(fg_detection_list)
    output_receiver_list.append(bg_detection_list)
    output_receiver_list.append(idx_to_keep)
    time_to_get_data = time.perf_counter_ns() 
    # print(f"Create GOBS: {(time_to_get_data - start)/1e6} ms")
    time_dict['mapping_time'] = (time.perf_counter_ns() - start)/1e6
    time_dict['gobs_creation_time'] = (time.perf_counter_ns() - start)/1e6

