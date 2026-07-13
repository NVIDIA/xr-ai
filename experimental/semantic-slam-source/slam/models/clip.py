# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLIP model implementation for semantic SLAM operations."""

import asyncio
import cv2
import gzip
import io
import os
import pickle
import sys
import threading
import time
from PIL import Image

import numpy as np
import open_clip
import torch

# Import utilities using proper package imports
from slam.utils.vis import vis_result_fast, vis_result_slow_caption
from slam.utils.model_utils import (
    compute_clip_features,
    compute_clip_features_batched
)

try: 
    from groundingdino.util.inference import Model
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
except ImportError as e:
    print("Import Error: Please install Grounded Segment Anything following the instructions in README.")
    raise e

# Set up some path used in this script
# Assuming all checkpoint files are downloaded as instructed by the original GSA repo
# Import configuration system  
from config.settings import get_config

# Get GSA path from config
config = get_config()
GSA_PATH = str(config.gsa_path)




class clipModel():
    def __init__(self, device="cuda:0", batched_clip=False, trt_clip=False, precision="fp16", batch_size=8, clip_model_name="ViT-H-14", pretrained="laion2b_s32b_b79k"):    
        self.trt_clip = trt_clip
        self.batched_clip = batched_clip
        self.device = device
        self.precision = precision
        self.batch_size = batch_size
        self.clip_model_name = clip_model_name
        self.pretrained = pretrained
        # Set dtype based on precision
        if self.precision == "fp16":
            self.dtype = torch.float16
        elif self.precision == "fp32" or self.precision == "float32":
            self.dtype = torch.float32
        else:
            raise ValueError(f"Unsupported precision: {self.precision}. Choose 'fp16', 'fp32', or 'float32'.")
        
        # Initialize the CLIP model
        if self.trt_clip:
            import utils.model_trt_utils as trt_utils
            device = int(device.split(":")[-1])
            self.clip_model = trt_utils.CLIP_TRT(precision=self.precision, batch_size=self.batch_size, clip_model=self.clip_model_name, device=device)
            
        else:
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                self.clip_model_name, self.pretrained
            )
            self.clip_model = self.clip_model.to(self.device)
            
            # Apply precision setting to the model when not using TRT
            self.clip_model = self.clip_model.to(dtype=self.dtype)
            self.clip_tokenizer = open_clip.get_tokenizer(self.clip_model_name)
    
    
    def get_clip_features(self, image_pil, image_rgb,detections, classes = ['item']):
        
        if len(detections.class_id) > 0:
            # Compute and save the clip features of detections  
            # clip_start = time.perf_counter_ns()
            if self.trt_clip:
                image_crops, image_feats, text_feats = self.clip_model.run_inference(
                    image_pil, detections, classes)
            else:
                # Use autocast for mixed precision inference when not using TRT
                autocast_enabled = self.precision == "fp16"
                with torch.autocast(device_type='cuda', dtype=self.dtype, enabled=autocast_enabled):
                    if self.batched_clip:
                        image_crops, image_feats, text_feats = compute_clip_features_batched(
                            image_rgb, detections, self.clip_model, self.clip_preprocess, self.clip_tokenizer, classes, self.device)
                    else:
                        image_crops, image_feats, text_feats = compute_clip_features(
                            image_rgb, detections, self.clip_model, self.clip_preprocess, self.clip_tokenizer, classes, self.device)
            # delta_time = (time.perf_counter_ns() - clip_start)
            # clip_time += delta_time
            # print(f"Detected {len(detections.class_id)} objects. CLIP TIME = {delta_time / 1e6} ms")
            # num_detections += len(detections.class_id)
        else:
            image_crops, image_feats, text_feats = [], [], []
        return image_crops, image_feats, text_feats
        
        