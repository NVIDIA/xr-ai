# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Segmentation module for semantic SLAM operations."""

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
import torch

# Import utilities using proper package imports
from slam.utils.vis import vis_result_fast, vis_result_slow_caption
from slam.utils.model_utils import (
    compute_clip_features,
    compute_clip_features_batched,
    get_sam_segmentation_from_xyxy_batched,
    get_sam_segmentation_from_xyxy
)

# Import external dependencies with proper error handling
try: 
    from groundingdino.util.inference import Model
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
except ImportError as e:
    print("Import Error: Please install Grounded Segment Anything following the instructions in README.")
    raise e

# Import configuration system
from config.settings import get_config

# Get GSA paths from config
config = get_config()
GSA_PATH = str(config.gsa_path)
TAG2TEXT_PATH = str(config.gsa_path)
EFFICIENTSAM_PATH = str(config.gsa_path / "EfficientSAM")

# Add GSA paths to sys.path for external dependencies
# Note: This is still needed for external GSA modules that aren't properly packaged
if GSA_PATH not in sys.path:
    sys.path.append(GSA_PATH)
if TAG2TEXT_PATH not in sys.path:
    sys.path.append(TAG2TEXT_PATH)
if EFFICIENTSAM_PATH not in sys.path:
    sys.path.append(EFFICIENTSAM_PATH)


# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = str(config.sam_checkpoint_path)



class SegmentationModel():
    def __init__(self, config=None, device=None, sam_variant=None, batched_sam=None, trt_sam=None, useDetector=None):
        """Initialize segmentor with configuration.
        
        Args:
            config: Configuration object (preferred)
            device, sam_variant, batched_sam, trt_sam: Override values (for backward compatibility)
            useDetector: Whether detection is being used (affects logic)
        """
        # Import config if not provided
        if config is None:
            from config.settings import get_config
            config = get_config()
        
        # Use config values or parameter overrides
        self.device = device if device is not None else config.model.segmentation.device
        self.sam_variant = sam_variant if sam_variant is not None else config.model.segmentation.sam_variant
        self.batched_sam = batched_sam if batched_sam is not None else config.model.segmentation.batched_sam
        self.trt_sam = trt_sam if trt_sam is not None else config.model.segmentation.trt_sam
        self.useDetector = useDetector if useDetector is not None else config.model.detection.enabled
        
        assert self.sam_variant in ['mobilesam', "lighthqsam", "sam"], f"Invalid SAM variant: {self.sam_variant}"
        self.trt_sam = trt_sam
        if self.useDetector:
            self.sam_predictor = self.get_sam_predictor()
        else:
            self.sam_predictor = self.get_sam_mask_generator()
        
    def get_sam_mask_generator(self):
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(self.device)
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=12,
            points_per_batch=144,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            crop_n_layers=0,
            min_mask_region_area=100,
        )
        return mask_generator
        
    def get_sam_predictor(self):

        if self.sam_variant == "sam":
            sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
            sam.to(self.device)
            sam_predictor = SamPredictor(sam)
            return sam_predictor
        
        if self.sam_variant == "mobilesam":
            from MobileSAM.setup_mobile_sam import setup_model
            MOBILE_SAM_CHECKPOINT_PATH = str(config.mobile_sam_checkpoint_path)
            checkpoint = torch.load(MOBILE_SAM_CHECKPOINT_PATH)
            mobile_sam = setup_model()
            mobile_sam.load_state_dict(checkpoint, strict=True)
            mobile_sam.to(device=self.device)
            
            sam_predictor = SamPredictor(mobile_sam)
            return sam_predictor

        elif self.sam_variant == "lighthqsam":
            from LightHQSAM.setup_light_hqsam import setup_model
            HQSAM_CHECKPOINT_PATH = str(config.hqsam_checkpoint_path)
            checkpoint = torch.load(HQSAM_CHECKPOINT_PATH)
            light_hqsam = setup_model()
            light_hqsam.load_state_dict(checkpoint, strict=True)
            light_hqsam.to(device=self.device)
            
            sam_predictor = SamPredictor(light_hqsam)
            return sam_predictor
            
        elif self.sam_variant == "fastsam":
            raise NotImplementedError
        else:
            raise NotImplementedError
        
        
    def run_segmentation(self, image_rgb, detections=None):
        
        if self.useDetector:
            if len(detections.class_id) > 0:
                ### Segment Anything ###
                # sam_start = time.thread_time_ns()
                if self.batched_sam:
                    xyxy_tensor = torch.tensor(detections.xyxy).to(self.device)
                    mask = get_sam_segmentation_from_xyxy_batched(
                        sam_predictor=self.sam_predictor,
                        image=image_rgb,
                        xyxy_tensor=xyxy_tensor
                    )
                else:
                    mask = get_sam_segmentation_from_xyxy(
                        sam_predictor=self.sam_predictor,
                        image=image_rgb,
                        xyxy=detections.xyxy
                    )
                # seg_time += (time.thread_time_ns() - sam_start)
                return mask, [], []
            else:
                image_crops, image_feats, text_feats = [], [], []
                return image_crops, image_feats, text_feats 

        else:
            results = self.sam_predictor.generate(image_rgb)
            mask = []
            xyxy = []
            conf = []
            for r in results:
                mask.append(r["segmentation"])
                r_xyxy = r["bbox"].copy()
                # Convert from xyhw format to xyxy format
                r_xyxy[2] += r_xyxy[0]
                r_xyxy[3] += r_xyxy[1]
                xyxy.append(r_xyxy)
                conf.append(r["predicted_iou"])
            mask = np.array(mask)
            xyxy = np.array(xyxy)
            conf = np.array(conf)
            return mask, xyxy, conf
