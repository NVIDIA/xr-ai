# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parallel inference and mapping module for semantic SLAM operations."""

import asyncio
import cv2
import gzip
import grpc
import io
import os
import pickle
import sys
import threading
import time
from concurrent import futures
from pathlib import Path
from time import sleep
from typing import Tuple
from PIL import Image, ImageDraw, ImageFont

import numpy as np
import omegaconf
import supervision as sv
import torch
import torchvision
from omegaconf import DictConfig

# Import vis_proto using proper relative imports
from slam.protocols.vis_proto import vis_pb2, vis_pb2_grpc

# Import SLAM modules using proper package imports
from slam.models.captioning import captioning
from slam.models.detection import detector
from slam.models.segmentation import SegmentationModel
from slam.models.clip import clipModel

from slam.datasets import ipad, replica, scannet
from slam.utils.vis import OnlineObjectRenderer
from slam.utils.ious import compute_2d_box_contained_batch
from slam.utils.general_utils import to_tensor, to_numpy
from slam.core.slam_classes import MapObjectList, DetectionList
from slam.core.utils import (
    merge_obj2_into_obj1, 
    filter_objects,
    merge_objects, 
    denoise_selected_objects,
)
from slam.core.mapping import (
    compute_spatial_similarities,
    compute_visual_similarities,
    aggregate_similarities,
    merge_detections_to_objects
)

from slam.utils.mapping_utils import (
    get_dataset,
    create_pcd_parallel,
    setup,
)
from slam.utils.debug_utils import dump_semantic_map, dump_inference_results

BG_CLASSES = ["wall", "floor", "ceiling"]
ASYNC_IO=True

DEBUG_PRINT = True
def debug_print(*args, **kwargs):
    if DEBUG_PRINT:
        print(*args, **kwargs)



def inference_consumer(inferenceQueue, 
                       visualizationQueue,
                       useDetector=False,
                       config=None,
                       datasetClass="iPad",
                       pipelined_mapping=False,
                       save_map=False,
                       scene_name=None,
                       ):
    # Extract config values or use defaults if config is None
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
    test_depth_downsampling = config.model.mapping.test_depth_downsampling
    
    nms_threshold = 0.5
    # set the models - segmentation and slam
    
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
    
    # ######################################################### MAPPING SERVER ############################################################################################################
    
    cfg = setup(useDetector, datasetClass)
    dataset = get_dataset(
        datasetClass=datasetClass,
        config_dict = cfg.dataset_config,
        desired_height = cfg.image_height,
        desired_width = cfg.image_width,
        device = "cpu",
        dtype=torch.float,
        scene_name=scene_name,
        test_depth_downsampling=test_depth_downsampling,
    )
    
    if not cfg.skip_bg:
        # Handle the background detection separately 
        # Each class of them are fused into the map as a single object
        bg_objects = {
            c: None for c in BG_CLASSES
        }
    else:
        bg_objects = None
    idx = 0
    history_map = {}                # Idx to object mapping. Key: idx, Value: None, or index in the obj list. If an object is removed then the value is set to None. If an object is edited, then this provides a consistent mapping to client; # keys are the indices of the objects when they were first added to the map (uniqueID/'history_idx'), values are the indices of the objects in the current ObjList
    next_index = 0
    objects = MapObjectList(device=config.model.device)
    
    
    if ASYNC_IO:
        made_grpc_call = False
        # For sending the results to mapping server
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        channel = grpc.aio.insecure_channel('localhost:50054')
    else:
        channel = grpc.insecure_channel('localhost:50054')
    
    
    # Create a gRPC channel and stub
    # channel = grpc.insecure_channel('localhost:50054') 
    stub = vis_pb2_grpc.VisualizerServerStub(channel)
    skipped_frames = 0
    print("*****************************************************[Mapping Consumer Started]*****************************************************")
    print("Configuration:")
    print(f"[MAPPING SERVER] useDetector: {useDetector}")
    print(f"[MAPPING SERVER] device: {config.model.device}")
    print(f"[MAPPING SERVER] image_height: {cfg.image_height}")
    print(f"[MAPPING SERVER] image_width: {cfg.image_width}")
    print(f"[MAPPING SERVER] skip_bg: {cfg.skip_bg}")
    print(f"[MAPPING SERVER] mask_conf_threshold: {cfg.mask_conf_threshold}")
    print(f"[MAPPING SERVER] test_depth_downsampling: {test_depth_downsampling}")
    
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
            print(f"🧹 [INFERENCE] Saving map data for {scene_name} (has {len(objects)} objects)")
            
            # Dump semantic map BEFORE clearing anything
            scene_classes = ['item']  # Default for non-detector mode
            if useDetector and captioning_model.global_classes:
                scene_classes = list(captioning_model.global_classes)
            elif useDetector and hasattr(captioning_model, 'classes') and captioning_model.classes:
                scene_classes = captioning_model.classes
            dump_semantic_map(scene_name, objects, bg_objects, cfg, scene_classes, save_map, test_depth_downsampling=test_depth_downsampling)
            
            # Now clear accumulated map data for next scene
            objects.clear()
            if not cfg.skip_bg:
                bg_objects = {c: None for c in BG_CLASSES}
            else:
                bg_objects = None
            history_map = {}
            next_index = 0
            idx = 0
            
            # Clear accumulated classes from captioning model for next scene
            if useDetector:
                captioning_model.global_classes.clear()
                captioning_model.classes = None
                print(f"🧹 [INFERENCE] Cleared all data for next scene")
            
            # Forward completion signal to visualization queue
            visualizationQueue.put(queue_item)
            continue
            
        # Check if this is a shutdown signal
        if isinstance(queue_item, dict) and queue_item.get('type') == 'shutdown':
            print("🏁 [INFERENCE] Received shutdown signal. Exiting...")
            visualizationQueue.put(queue_item)  # Forward to visualization
            break
            
        image_pil, image_cv2_bgr, depth_array, pose_array, frameNumber, starting_timestamp, clientTimeStamps, time_dict = queue_item
        # print("[MAPPING SERvER]\t\t\t Time taken so far (Decoding + Queue time): ", (time.perf_counter_ns() - starting_timestamp)/1e6)
        # convert the frame to a numpy array
        # pass the numpy array to the model
        # get the results from the model
    
        # debug_print("[INFERENCE SERVER]\t\tProcessing frame-------: ", frameNumber)
        classes = None
        
        image_rgb = cv2.cvtColor(image_cv2_bgr, cv2.COLOR_BGR2RGB)
        start_time = time.perf_counter_ns()
        if useDetector: 
            
            caption, text_prompt = captioning_model.gen_caption(image_pil)
            
            classes = captioning_model.classes
            caption_end = time.perf_counter_ns()
            # print("[INFERENCE SERVER]\t\t CLASSES --------------------------------------: ",  captioning_model.classes)
            detections = detection_model.get_detections(image_cv2_bgr, captioning_model.classes)
            detecttion_end = time.perf_counter_ns()
            # print("[INFERENCE SERVER]\t\tNumber of objects detected: ", len(detections.xyxy))
            mask, _, _ = segmentation_model.run_segmentation(image_rgb, detections)
            detections.mask = mask
            seg_end = time.perf_counter_ns()
            time_dict['caption_time'] = (caption_end - start_time)/1e6
            time_dict['detection_time'] = (detecttion_end - caption_end)/1e6
            time_dict['segmentation_time'] = (seg_end - detecttion_end)/1e6
        else:
            classes = ['item']
            mask, xyxy, conf = segmentation_model.run_segmentation(image_rgb, None)
            detections = sv.Detections(
                xyxy=xyxy,
                confidence=conf,
                class_id=np.zeros_like(conf).astype(int),
                mask=mask,
            )
            seg_end = time.perf_counter_ns()
            time_dict['segmentation_time'] = (seg_end - start_time)/1e6
                            
        # Convert the detections to a dict. The elements are in np.array
        results = {
            "xyxy": detections.xyxy,
            "confidence": detections.confidence,
            "class_id": detections.class_id,
            "mask": detections.mask,
            "classes":  classes,
            # "image_crops": None,
            "image_feats": None,
            "text_feats": None,
        }
        
        if useDetector:
            results["tagging_caption"] = caption
            results["tagging_text_prompt"] = text_prompt
            
        image_np =  np.array(image_pil)
        output_receiver_list = [] 
        child_thread = threading.Thread(target=create_pcd_parallel, args=(image_np, depth_array, pose_array, frameNumber, dataset, cfg, classes, results, output_receiver_list, pipelined_mapping ,datasetClass, time_dict))
        child_thread.start()
        
        
        clip_start = time.perf_counter_ns()
        image_crops, image_feats, text_feats = clip_model.get_clip_features(image_pil, image_rgb, detections, classes)
        end_clip = time.perf_counter_ns()
        time_dict['clip_time'] = (end_clip - clip_start)/1e6
        
        # print(f"[INFERENCE SERVER]\t\tTime taken for frame {frameNumber} for inference: { (end-start_time)/1e6} ms")

        child_thread.join()
        time_dict['inference_time'] = time_dict['clip_time'] + time_dict['segmentation_time'] + time_dict['detection_time'] + time_dict['caption_time']
        if config.debug.dump_inference:
            dump_inference_results(results, frameNumber, image_pil, depth_array, pose_array, classes, config)
            continue                  # to avoid running the whole thing, and focus only on the inference part
            
        
        
        # #######################################################################################################################################################################################################################################
        map_start_time = time.perf_counter_ns()
        try:
            fg_detection_list, bg_detection_list, idx_to_keep = output_receiver_list[0], output_receiver_list[1], output_receiver_list[2]
        except:
            print("Error in output_receiver_list -- size = ", len(output_receiver_list))    
            exit()
        
        
        for obj in fg_detection_list:
            obj['clip_ft'] = to_tensor(image_feats[obj['mask_idx'][0]])
            obj['text_ft'] = to_tensor(text_feats[obj['mask_idx'][0]])
            # obj['image_crops'] = image_crops[obj['mask_idx'][0]]
        for obj in bg_detection_list:
            obj['clip_ft'] = to_tensor(image_feats[obj['mask_idx'][0]])
            obj['text_ft'] = to_tensor(text_feats[obj['mask_idx'][0]])
            # obj['image_crops'] = image_crops[obj['mask_idx'][0]]
            
        if len(bg_detection_list) > 0:
            for detected_object in bg_detection_list:
                class_name = detected_object['class_name'][0]
                if bg_objects[class_name] is None:
                    bg_objects[class_name] = detected_object
                else:
                    matched_obj = bg_objects[class_name]
                    matched_det = detected_object
                    bg_objects[class_name] = merge_obj2_into_obj1(cfg, matched_obj, matched_det, run_dbscan=False)
            
        if len(fg_detection_list) == 0:
            debug_print(f"Frame {idx}: Detected 0 objects. Total objects: {len(objects)}")
            continue
            
        if cfg.use_contain_number:
            xyxy = fg_detection_list.get_stacked_values_torch('xyxy', 0)
            contain_numbers = compute_2d_box_contained_batch(xyxy, cfg.contain_area_thresh)
            for i in range(len(fg_detection_list)):
                fg_detection_list[i]['contain_number'] = [contain_numbers[i]]
            
        if len(objects) == 0:
            # Add all detections to the map
            for i in range(len(fg_detection_list)):
                objects.append(fg_detection_list[i])
                fg_detection_list[i]["history_idx"] = next_index
                history_map[next_index] = i
                next_index +=1

            # Skip the similarity computation 
            continue
        siimilarity_time = time.perf_counter_ns()        
        spatial_sim = compute_spatial_similarities(cfg, fg_detection_list, objects)
        visual_sim = compute_visual_similarities(cfg, fg_detection_list, objects)
        agg_sim = aggregate_similarities(cfg, spatial_sim, visual_sim)
        simimlaritty_end = time.perf_counter_ns()
        time_dict['similarity_time'] = (simimlaritty_end - siimilarity_time)/1e6
        # Compute the contain numbers for each detection
        if cfg.use_contain_number:
            # Get the contain numbers for all objects
            contain_numbers_objects = torch.Tensor([obj['contain_number'][0] for obj in objects])
            detection_contained = contain_numbers > 0 # (M,)
            object_contained = contain_numbers_objects > 0 # (N,)
            detection_contained = detection_contained.unsqueeze(1) # (M, 1)
            object_contained = object_contained.unsqueeze(0) # (1, N)                

            # Get the non-matching entries, penalize their similarities
            xor = detection_contained ^ object_contained
            agg_sim[xor] = agg_sim[xor] - cfg.contain_mismatch_penalty
        
        # Threshold sims according to cfg. Set to negative infinity if below threshold
        agg_sim[agg_sim < cfg.dataset_config.mapping.sim_threshold] = float('-inf')
        merging = time.perf_counter_ns()
        objects, edited_objects_idx, new_obj_idx, history_map, next_index = merge_detections_to_objects(cfg, fg_detection_list, objects, agg_sim, history_map, next_index)
        removed_obj_1 = [] 
        removed_object_2 = []
        edited_objects_idx_2 = []
        
        mergin_end = time.perf_counter_ns()
        time_dict['merging_time'] = (mergin_end - merging)/1e6
        # Perform post-processing periodically if told so
        if cfg.denoise_interval > 0 and (idx+1) % cfg.denoise_interval == 0:
            denoise_start = time.perf_counter_ns()
            objects = denoise_selected_objects(cfg, objects, edited_objects_idx)
            denoise_end = time.perf_counter_ns()
            time_dict['post_process_denoise_time'] = (denoise_end - denoise_start)/1e6
        if cfg.filter_interval > 0 and (idx+1) % cfg.filter_interval == 0:
            objects,removed_obj_1, history_map = filter_objects(cfg, objects, history_map)
        if cfg.merge_interval > 0 and (idx+1) % cfg.merge_interval == 0:
            objects, removed_object_2, edited_objects_idx_2, history_map = merge_objects(cfg, objects, history_map)
        
        idx += 1
        
        final_removed_objects = removed_obj_1 + removed_object_2
        
        queue_pcd_points = []
        queue_pcd_bbox = []
        if len(objects) == 0:
            print("\n\n\n\nDEBUG:RAHUL  __--------------------------__objects:: EXITING  ", objects)
            continue
        for i, obj in enumerate(objects):
            queue_pcd_points.append(np.asarray(obj["pcd"].points,  dtype=np.float64))
            queue_pcd_bbox.append(np.asarray(obj["bbox"].get_box_points(),  dtype=np.float64))
        objects_clip_fts = objects.get_stacked_values_torch('clip_ft')
        objects_text_fts = objects.get_stacked_values_torch('text_ft')
        fresh_objects = edited_objects_idx + new_obj_idx + edited_objects_idx_2
        
        pre_serial_time = time.perf_counter_ns()

        results = {
            'queue_pcd_points': queue_pcd_points,
            # 'bg_objects': None if bg_objects is None else bg_objects.to_serializable(),
            'queue_pcd_bbox': queue_pcd_bbox,
            'objects_text_fts': objects_text_fts,
            'objects_clip_fts': objects_clip_fts,
            'fresh_objects': fresh_objects,         #index of all objects that have been touched/edited/added
            'removed_objects': final_removed_objects,  #index of all objects that have been removed
            'history_map': history_map,                     #mapping of idx to object index. If an object is removed then the value is set to None. If an object is edited, then this provides a consistent mapping to client
            'class_names': classes,                 # Current frame classes for visualization
        }
        
        
        result_serialization_time = time.perf_counter_ns()      
        
        if visualizationQueue.full():
            skipped_frames += 1
            print(f"⚠️  [MAPPING→VISUALIZATION] Queue full! Dropping frame {frameNumber}. Total skipped: {skipped_frames}")
            print(f"    Frame details: {len(fg_detection_list)} objects detected, {len(objects)} total objects")
            continue  
        time_dict['post_processing_time'] = (pre_serial_time - mergin_end)/1e6   
        time_dict['mapping_time'] += (result_serialization_time - map_start_time)/1e6   
        time_dict['clip_mapping'] = (result_serialization_time - seg_end)/1e6
        time_dict['total_time']     = (result_serialization_time - start_time)/1e6
        visualizationQueue.put((results,  frameNumber, starting_timestamp, clientTimeStamps, time_dict))
        
        
        async def makeGRPCCall():   
            # Create the request message
            vis_request = vis_pb2.Status(message=True)

            # Send the request to the visualization server
            try:
                response = stub.updateMap(vis_request)
                # print(f"Visualization server response: {response.message}")
            except grpc.RpcError as e:
                print(f"gRPC error: {e}")
        
        if ASYNC_IO:
            if not made_grpc_call:
                loop.run_until_complete(makeGRPCCall())
                made_grpc_call = True
        else:
            
            # Create the request message
            vis_request = vis_pb2.Status(message=True)

            # Send the request to the visualization server
            try:
                response = stub.updateMap(vis_request)
                debug_print(f"Visualization server response: {response.message}")
            except grpc.RpcError as e:
                debug_print(f"gRPC error: {e}")
        grpcs_time = time.perf_counter_ns()
        
        # debug_print(f"[MAPPING SERvER] Frame {idx}: Detected {len(fg_detection_list)} objects. Total objects: {len(objects)}, Time taken for computation: {(end_compute - start_time)/1e6} ms, Serialization: {(result_serialization_time - end_compute)/1e6} ms, gRPC: {(grpcs_time - result_serialization_time)/1e6} ms")
        # print(f"[MAPPING SERVER]\t\t\t Frame {frameNumber}: Time taken for bookkeeping computation: {(end_bookkeeping_compute - map_start_time)/1e6} ms, Similarity: {(simimlaritty_end - siimilarity_time)/1e6} ms, Denoise: {(denoise_end - denoise_start)/1e6} ms   Merging time = {(mergin_end - merging)/1e6} ms, Serialization: {(result_serialization_time - pre_serial_time)/1e6} ms, Point cloud creation: {time_dict['create_pcd_time']} ms")
        
        

        
        
        
        
        

        
       
   
       
       
        

        


        
        
        
        
        

