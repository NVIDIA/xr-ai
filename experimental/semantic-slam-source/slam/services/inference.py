# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference consumer module for semantic SLAM operations."""

import asyncio
import cv2
import gzip
import io
import os
import pickle
import sys
import threading
import time
from time import sleep
from PIL import Image

import numpy as np
import supervision as sv
import torch
import torchvision

# Import SLAM modules using proper package imports
from slam.models.captioning import captioning
from slam.models.detection import detector
from slam.models.segmentation import SegmentationModel
from slam.models.clip import clipModel

# Import configuration system
from config.settings import get_config
from pathlib import Path
from slam.utils.debug_utils import dump_inference_results

DEBUG_PRINT = True
def debug_print(*args, **kwargs):
    if DEBUG_PRINT:
        print(*args, **kwargs)

def inference_consumer(inferenceQueue, 
                       mappingQueue,
                       useDetector=False,
                       config=None,
                       ):
    nms_threshold = 0.5
    # set the models - segmentation and slam
    
    # Get or set config
    if config is None:
        print("WARNING: No config provided, using default config")
        from config.settings import get_config
        config = get_config()
    else:
        # Set the config globally so utils.py can access it
        from config.settings import set_config
        set_config(config)
    
    # Extract values from config
    batched_sam = config.model.segmentation.batched_sam
    trt_sam = config.model.segmentation.trt_sam
    batched_clip = config.model.clip.batched_clip
    trt_clip = config.model.clip.trt_clip
    precision = config.model.clip.precision
    batch_size = config.model.clip.batch_size
    sam_variant = config.model.segmentation.sam_variant
    device = config.model.segmentation.device
    
    if  not useDetector and sam_variant != "sam":
        raise ValueError("If useDetector is False, sam_variant must be sam. ")
    
    if useDetector:
        captioning_model = captioning(
            class_set=config.model.captioning.class_set, 
            device=config.model.captioning.device, 
            add_bg_classes=config.model.captioning.add_bg_classes, 
            accumu_classes=config.model.captioning.accumu_classes
        )
        detection_model = detector(detector="dino", device=device, box_threshold=0.2, text_threshold=0.2, nms_threshold=0.5)
    segmentation_model = SegmentationModel(device=device, sam_variant=sam_variant, batched_sam=batched_sam, trt_sam=trt_sam, useDetector=useDetector)
    clip_model = clipModel(device=device, batched_clip=batched_clip, trt_clip=trt_clip, precision=precision, batch_size=batch_size, clip_model_name=config.model.clip.model_name, pretrained=config.model.clip.pretrained)
    skipped_frames = 0
    print("*****************************************************[Inference Consumer Started]*****************************************************")
    print("Configuration:")
    print(f"[INFERENCE SERVER] useDetector: {useDetector}")
    print(f"[INFERENCE SERVER] batched_sam: {batched_sam}")
    print(f"[INFERENCE SERVER] trt_sam: {trt_sam}")
    print(f"[INFERENCE SERVER] batched_clip: {batched_clip}")
    print(f"[INFERENCE SERVER] trt_clip: {trt_clip}")
    print(f"[INFERENCE SERVER] precision: {precision}")
    print(f"[INFERENCE SERVER] batch_size: {batch_size}")
    print(f"[INFERENCE SERVER] sam_variant: {sam_variant}")
    print(f"[INFERENCE SERVER] device: {device}")
    
    while True:
        # dequeue a frame 
        if inferenceQueue.empty():
            continue
        # dequeue a frame
        queue_item = inferenceQueue.get()
        
        # Check if this is a scene completion signal
        if isinstance(queue_item, dict) and queue_item.get('type') == 'scene_completion':
            scene_name = queue_item['scene_name']
            print(f"🎯 [INFERENCE] Received scene completion signal for {scene_name}")
            print(f"🧹 [INFERENCE] Scene completed - forwarding to mapping (no state to clear here)")
            
            # Forward completion signal to mapping queue
            mappingQueue.put(queue_item)
            continue
            
        # Check if this is a shutdown signal
        if isinstance(queue_item, dict) and queue_item.get('type') == 'shutdown':
            print("🏁 [INFERENCE] Received shutdown signal. Exiting...")
            mappingQueue.put(queue_item)  # Forward to mapping
            break
            
        image_pil, image_cv2_bgr, depth_array, pose_array, frameNumber, starting_timestamp, clientTimeStamps, time_dict = queue_item
        # convert the frame to a numpy array
        # pass the numpy array to the model
        # get the results from the model
    
        # debug_print("[INFERENCE SERVER]\t\tProcessing frame-------: ", frameNumber)
        classes = None
        
        image_rgb = cv2.cvtColor(image_cv2_bgr, cv2.COLOR_BGR2RGB)
        if useDetector: 
            start = time.perf_counter_ns()
            caption, text_prompt = captioning_model.gen_caption(image_pil)
            classes = captioning_model.classes
            caption_end = time.perf_counter_ns()
            # print("[INFERENCE SERVER]\t\t CLASSES --------------------------------------: ",  captioning_model.classes)
            detections = detection_model.get_detections(image_cv2_bgr, captioning_model.classes)
            # print("[INFERENCE SERVER]\t\tNumber of objects detected: ", len(detections.xyxy))
            detecttion_end = time.perf_counter_ns()
            mask, _, _ = segmentation_model.run_segmentation(image_rgb, detections)
            detections.mask = mask
            seg_end = time.perf_counter_ns()
            image_crops, image_feats, text_feats = clip_model.get_clip_features(image_pil, image_rgb, detections, captioning_model.classes)
            end = time.perf_counter_ns()
            time_dict['caption_time'] = (caption_end - start)/1e6
            time_dict['detection_time'] = (detecttion_end - caption_end)/1e6
            time_dict['segmentation_time'] = (seg_end - detecttion_end)/1e6
            time_dict['clip_time'] = (end - seg_end)/1e6
            
        else:
            start = time.perf_counter_ns()
            classes = ['item']
            mask, xyxy, conf = segmentation_model.run_segmentation(image_rgb, None)
            detections = sv.Detections(
                xyxy=xyxy,
                confidence=conf,
                class_id=np.zeros_like(conf).astype(int),
                mask=mask,
            )
            seg_end = time.perf_counter_ns()
            
            image_crops, image_feats, text_feats = clip_model.get_clip_features(image_pil, image_rgb, detections, classes)            
            end = time.perf_counter_ns()
            time_dict['segmentation_time'] = (seg_end - start)/1e6
            time_dict['clip_time'] = (end - seg_end)/1e6
            
        # print(f"[INFERENCE SERVER]\t\tTime taken for frame {frameNumber} for inference: { (end-start)/1e6} ms")
        time_dict['inference_time'] = time_dict['clip_time'] + time_dict['segmentation_time'] + time_dict['detection_time'] + time_dict['caption_time']
        
        # Convert the detections to a dict. The elements are in np.array
        results = {
            "xyxy": detections.xyxy,
            "confidence": detections.confidence,
            "class_id": detections.class_id,
            "mask": detections.mask,
            "classes":  classes,
            "image_crops": image_crops,
            "image_feats": image_feats,
            "text_feats": text_feats,
        }
        
        if useDetector:
            results["tagging_caption"] = caption
            results["tagging_text_prompt"] = text_prompt
            
        # DEBUG: Dump inference results for reproducibility debugging
        if config.debug.dump_inference:
            dump_inference_results(results, frameNumber, image_pil, depth_array, pose_array, classes)
        
        if mappingQueue.full():
            skipped_frames += 1
            print(f"⚠️  [INFERENCE→MAPPING] Queue full! Dropping frame {frameNumber}. Total skipped: {skipped_frames}")
            continue
        queue_sending_item = (results, depth_array, pose_array, frameNumber, image_pil, classes, starting_timestamp, clientTimeStamps, time_dict)
        mappingQueue.put(queue_sending_item)
        
        
        
        
    
        
        
        
        
        

