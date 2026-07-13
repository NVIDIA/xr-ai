# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detection module for semantic SLAM operations."""

import asyncio
import cv2
import gzip
import io
import os
import pickle
import sys
import threading
import time
from typing import Any, List
from PIL import Image

import numpy as np
import torch
import torchvision

# Set up GSA path from environment
# Import configuration system
from config.settings import get_config

# Get GSA path from config
config = get_config()
GSA_PATH = str(config.gsa_path)

# Add GSA path for external dependencies
if GSA_PATH not in sys.path:
    sys.path.append(GSA_PATH)
try: 
    from groundingdino.util.inference import Model
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
except ImportError as e:
    print("Import Error: Please install Grounded Segment Anything following the instructions in README.")
    raise e

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = str(config.grounding_dino_config_path)
GROUNDING_DINO_CHECKPOINT_PATH = str(config.grounding_dino_checkpoint_path)



class detector():
    def __init__(self, config=None, detector=None, device=None, box_threshold=None, text_threshold=None, nms_threshold=None, precision=None):
        """Initialize detector with configuration.
        
        Args:
            config: Configuration object (preferred)
            detector, device, box_threshold, text_threshold, nms_threshold, precision: Override values (for backward compatibility)
        """
        # Import config if not provided
        if config is None:
            from config.settings import get_config
            config = get_config()
        
        # Use config values or parameter overrides
        self.detector = detector if detector is not None else config.model.detection.detector
        self.device = device if device is not None else config.model.detection.device
        self.box_threshold = box_threshold if box_threshold is not None else config.model.detection.box_threshold
        self.text_threshold = text_threshold if text_threshold is not None else config.model.detection.text_threshold
        self.nms_threshold = nms_threshold if nms_threshold is not None else config.model.detection.nms_threshold
        self.precision = precision if precision is not None else config.model.detection.precision
        
        ### Initialize the Grounding DINO model ###
        if detector == "dino":
            self.grounding_dino_model = Model(
                model_config_path=GROUNDING_DINO_CONFIG_PATH, 
                model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH, 
                device=self.device
            )
        else:
            raise ValueError(f"Unsupported detector: {self.detector!r}. Only 'dino' is supported.")
    
    def _predict_with_precision(self, image, classes):
        """Custom prediction method that handles float16 precision correctly using autocast."""
        from groundingdino.util.inference import predict, Model
        
        # Replicate the logic from Model.predict_with_classes but with autocast for precision
        caption = ". ".join(classes)
        processed_image = Model.preprocess_image(image_bgr=image).to(self.device)
        
        # Use autocast for mixed precision inference
        autocast_enabled = self.precision == "float16"
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=autocast_enabled):
            boxes, logits, phrases = predict(
                model=self.grounding_dino_model.model,
                image=processed_image,
                caption=caption,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device)
            
        source_h, source_w, _ = image.shape
        detections = Model.post_process_result(
            source_h=source_h,
            source_w=source_w,
            boxes=boxes,
            logits=logits)
        class_id = Model.phrases2classes(phrases=phrases, classes=classes)
        detections.class_id = class_id
        return detections
    
    def get_detections(self, image, classes):
        if self.detector == "dino":
            # print(f"Frame {idx}: Classes to detect: {classes}")
            # detection_start = time.perf_counter_ns()
            # Using GroundingDINO to detect and SAM to segment

            detections = self._predict_with_precision(image, classes)
            # print("Sept30 ",len(detections))
        
            if len(detections.class_id) > 0:
                ### Non-maximum suppression ###
                # print(f"Before NMS: {len(detections.xyxy)} boxes")
                nms_idx = torchvision.ops.nms(
                    torch.from_numpy(detections.xyxy), 
                    torch.from_numpy(detections.confidence), 
                    self.nms_threshold
                ).numpy().tolist()
                # print(f"After NMS: {len(detections.xyxy)} boxes")

                detections.xyxy = detections.xyxy[nms_idx]
                detections.confidence = detections.confidence[nms_idx]
                detections.class_id = detections.class_id[nms_idx]
                
                # Somehow some detections will have class_id=-1, remove them
                valid_idx = detections.class_id != -1
                detections.xyxy = detections.xyxy[valid_idx]
                detections.confidence = detections.confidence[valid_idx]
                detections.class_id = detections.class_id[valid_idx]
                # print("Sept30 ",len(detections), detections.class_id)
            # detection_time += (time.perf_counter_ns() - detection_start)
            return detections
                # # Somehow some detections will have class_id=-None, remove them
                # valid_idx = [i for i, val in enumerate(detections.class_id) if val is not None]
                # detections.xyxy = detections.xyxy[valid_idx]
                # detections.confidence = detections.confidence[valid_idx]
                # detections.class_id = [detections.class_id[i] for i in valid_idx]
                
        else:
            raise ValueError(f"Unsupported detector: {self.detector!r}. Only 'dino' is supported.")
        
    