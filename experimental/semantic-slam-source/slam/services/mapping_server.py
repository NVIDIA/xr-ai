# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Mapping server for semantic SLAM operations."""

import asyncio
import grpc
import os
import sys
import time
from concurrent import futures
from pathlib import Path
from typing import Tuple

import numpy as np
import omegaconf
import torch
from omegaconf import DictConfig
from PIL import Image, ImageDraw, ImageFont

# Import vis_proto using proper relative imports
from slam.protocols.vis_proto import vis_pb2, vis_pb2_grpc

# Import SLAM utilities using proper package imports
from slam.utils.vis import OnlineObjectRenderer
from slam.utils.ious import compute_2d_box_contained_batch
from slam.utils.general_utils import to_tensor, to_numpy
from slam.utils.debug_utils import dump_semantic_map
from slam.core.slam_classes import MapObjectList, DetectionList
from slam.core.utils import (
    merge_obj2_into_obj1, 
    filter_objects,
    merge_objects, 
    gobs_to_detection_list,
    gobs_to_detection_list_optimized,
    denoise_selected_objects,
)
from slam.core.mapping import (
    compute_spatial_similarities,
    compute_visual_similarities,
    aggregate_similarities,
    merge_detections_to_objects
)

from slam.utils.mapping_utils import get_dataset, setup

BG_CLASSES = ["wall", "floor", "ceiling"]
ASYNC_IO=True
DEBUG_PRINT = False
def debug_print(*args, **kwargs):
    if DEBUG_PRINT:
        print(*args, **kwargs)
   

def mapping_consumer(mappingQueue, visualizationQueue, useDetector=False, datasetClass="iPad", save_map=False, experiment_config=None, scene_name=None):
    # Set the config globally so utils.py can access it
    if experiment_config is not None:
        from config.settings import set_config
        set_config(experiment_config)
    
    cfg = setup(useDetector, datasetClass)
    dataset = get_dataset(
        datasetClass=datasetClass,
        config_dict = cfg.dataset_config,
        desired_height = cfg.image_height,
        desired_width = cfg.image_width,
        device = "cpu",
        dtype=torch.float,
        scene_name=scene_name,
        test_depth_downsampling=experiment_config.model.mapping.test_depth_downsampling,
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
    objects = MapObjectList(device=cfg.device)
    
    
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
    print(f"[MAPPING SERVER] device: {cfg.device}")
    print(f"[MAPPING SERVER] image_height: {cfg.image_height}")
    print(f"[MAPPING SERVER] image_width: {cfg.image_width}")
    print(f"[MAPPING SERVER] skip_bg: {cfg.skip_bg}")
    print(f"[MAPPING SERVER] mask_conf_threshold: {cfg.mask_conf_threshold}")
    print(f"[MAPPING SERVER] test_depth_downsampling: {experiment_config.model.mapping.test_depth_downsampling}")
    
    while True:
            
        # TODO - Dequeue the results from the queue
        # get image_rgb
        if mappingQueue.empty():
            continue
        queue_item = mappingQueue.get()
        
        # Check if this is a scene completion signal
        if isinstance(queue_item, dict) and queue_item.get('type') == 'scene_completion':
            scene_name = queue_item['scene_name']
            print(f"🎯 [MAPPING] Received scene completion signal for {scene_name}")
            print(f"🧹 [MAPPING] Saving map data for {scene_name} (has {len(objects)} objects)")
            
            # Dump semantic map BEFORE clearing anything
            # For mapping server, we don't have accumulated classes like inference pipeline
            # Use a default class list for consistency
            scene_classes = ['item']  # Default for mapping server
            dump_semantic_map(scene_name, objects, bg_objects, cfg, scene_classes, save_map, test_depth_downsampling=experiment_config.model.mapping.test_depth_downsampling)
            
            # Now clear accumulated map data for next scene
            objects.clear()
            if not cfg.skip_bg:
                bg_objects = {c: None for c in BG_CLASSES}
            else:
                bg_objects = None
            history_map = {}
            next_index = 0
            idx = 0
            
            # Forward completion signal to visualization queue
            visualizationQueue.put(queue_item)
            continue
            
        # Check if this is a shutdown signal
        if isinstance(queue_item, dict) and queue_item.get('type') == 'shutdown':
            print("🏁 [MAPPING] Received shutdown signal. Exiting...")
            visualizationQueue.put(queue_item)  # Forward to visualization
            break
            
        results, depth_np_array, poseString, frameNumber, image_pil, global_classes, starting_timestamp, clientTimeStamps, time_dict = queue_item
        debug_print("[MAPPING SERVER]\t\tProcessing frame-------: ", frameNumber, "  Lagging by : ", mappingQueue.qsize(), " frames")
        start_time = time.perf_counter_ns()

        
        gobs = results    #results from previous steps
        classes = global_classes
        # idx = frameNumber
        image_np =  np.array(image_pil)
        pose = poseString
        depth_array = depth_np_array
        
        color_tensor, depth_tensor, intrinsics, unt_pose = dataset.getItems(image_np, depth_array, pose)
        # print("Got the tensors", self.idx, color_tensor.shape, depth_tensor.shape, intrinsics.shape, unt_pose.shape)
          
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
        time_to_get_data = time.perf_counter_ns() 
        
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
        )
        
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
        end_compute = time.perf_counter_ns()
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
            
        }
        
        
        result_serialization_time = time.perf_counter_ns()      
        
        if visualizationQueue.full():
            skipped_frames += 1
            print(f"⚠️  [MAPPING→VISUALIZATION] Queue full! Dropping frame {frameNumber}. Total skipped: {skipped_frames}")
            print(f"    Frame details: {len(fg_detection_list)} objects detected, {len(objects)} total objects")
            continue  
        time_dict['mapping_time'] = (result_serialization_time - start_time)/1e6
        time_dict['post_processing_time'] = (pre_serial_time - mergin_end)/1e6   

        visualizationQueue.put((results, frameNumber, starting_timestamp, clientTimeStamps, time_dict))
        
        
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
        # print(f"[MAPPING SERvER] Frame {frameNumber}: Time taken for computation: {(end_compute - start_time)/1e6} ms, Similarity: {(simimlaritty_end - siimilarity_time)/1e6} ms, Denoise: {(denoise_end - denoise_start)/1e6} ms   Merging time = {(mergin_end - merging)/1e6} ms Data Access = {(time_to_get_data - start_time)/1e6} ms, Serialization: {(result_serialization_time - end_compute)/1e6} ms, gRPC: {(grpcs_time - result_serialization_time)/1e6} ms")
        
        

        
        
        
        
        