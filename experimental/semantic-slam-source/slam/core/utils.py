# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SLAM utility functions for object detection and mapping."""

import copy
import cv2
import json
import time
from collections import Counter

import faiss
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from slam.utils.general_utils import to_tensor, to_numpy, Timer
from .slam_classes import MapObjectList, DetectionList

from slam.utils.ious import (
    compute_3d_iou,
    compute_3d_iou_accuracte_batch,
    mask_subtract_contained,
    compute_iou_batch
)
from slam.datasets.base import from_intrinsics_matrix
from config.settings import get_config

experiment_config = None

def get_classes_colors(classes):
    class_colors = {}

    # Generate a random color for each class
    for class_idx, class_name in enumerate(classes):
        # Generate random RGB values between 0 and 255
        r = np.random.randint(0, 256)/255.0
        g = np.random.randint(0, 256)/255.0
        b = np.random.randint(0, 256)/255.0

        # Assign the RGB values as a tuple to the class in the dictionary
        class_colors[class_idx] = (r, g, b)

    class_colors[-1] = (0, 0, 0)

    return class_colors

def create_or_load_colors(cfg, filename="gsa_classes_tag2text"):
    
    # get the classes, should be saved when making the dataset
    classes_fp = cfg['dataset_root'] / cfg['scene_id'] / f"{filename}.json"
    classes  = None
    with open(classes_fp, "r") as f:
        classes = json.load(f)
    
    # create the class colors, or load them if they exist
    class_colors  = None
    class_colors_fp = cfg['dataset_root'] / cfg['scene_id'] / f"{filename}_colors.json"
    if class_colors_fp.exists():
        with open(class_colors_fp, "r") as f:
            class_colors = json.load(f)
        print("Loaded class colors from ", class_colors_fp)
    else:
        class_colors = get_classes_colors(classes)
        class_colors = {str(k): v for k, v in class_colors.items()}
        with open(class_colors_fp, "w") as f:
            json.dump(class_colors, f)
        print("Saved class colors to ", class_colors_fp)
    return classes, class_colors

def create_object_pcd(depth_array, mask, cam_K, image, obj_color=None, frameNumer = None, time_dict=None, cfg=None) -> o3d.geometry.PointCloud:
    fx, fy, cx, cy = from_intrinsics_matrix(cam_K)
    pre_open3d_start = time.perf_counter_ns()
    # Also remove points with invalid depth values
    mask = np.logical_and(mask, depth_array > 0)

    if mask.sum() == 0:
        pcd = o3d.geometry.PointCloud()
        return pcd
        
    height, width = depth_array.shape
    x = np.arange(0, width, 1.0)
    y = np.arange(0, height, 1.0)
    u, v = np.meshgrid(x, y)
    
    # Apply the mask, and unprojection is done only on the valid points
    masked_depth = depth_array[mask] # (N, )
    u = u[mask] # (N, )
    v = v[mask] # (N, )

    # Convert to 3D coordinates
    x = (u - cx) * masked_depth / fx
    y = (v - cy) * masked_depth / fy
    z = masked_depth

    convert_to_3d = time.perf_counter_ns()
    # time_dict['convert_to_3d_time'] += (convert_to_3d - pre_open3d_start)/1e6

    # Stack x, y, z coordinates into a 3D point cloud
    points = np.stack((x, y, z), axis=-1)
    points = points.reshape(-1, 3)

    stacking_time = time.perf_counter_ns()
    # time_dict['stacking_time'] += (stacking_time - convert_to_3d)/1e6
    
    # Perturb the points a bit to avoid colinearity (using cheap fixed pattern)
    
    # Use pre-computed perturbation pattern instead of expensive random generation
    # This is ~99% faster than np.random.normal() while still avoiding colinearity
    perturbation_pattern = np.array([
        [1e-3, -2e-3, 1e-3],
        [-2e-3, 1e-3, -1e-3], 
        [1e-3, 1e-3, -2e-3],
        [-1e-3, -1e-3, 1e-3],
        [2e-3, -1e-3, -1e-3]
    ], dtype=np.float32)
    
    # Apply pattern cyclically to points
    pattern_idx = np.arange(points.shape[0]) % len(perturbation_pattern)
    points += perturbation_pattern[pattern_idx]
        
    perturbing_time = time.perf_counter_ns()
    # time_dict['perturbing_time'] += (perturbing_time - stacking_time)/1e6
    

    if obj_color is None: # color using RGB
        # # Apply mask to image
        colors = image[mask] / 255.0
    else: # color using group ID
        # Use the assigned obj_color for all points
        colors = np.full(points.shape, obj_color)

    post_color_time = time.perf_counter_ns()
    # time_dict['color_time'] += (post_color_time - perturbing_time)/1e6
    
    if points.shape[0] == 0:
        import pdb; pdb.set_trace()
    pre_open3d_end = time.perf_counter_ns()

    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    post_open3d_start = time.perf_counter_ns()
    # time_dict['pre_open3d_time'] += (pre_open3d_end - pre_open3d_start)/1e6
    # time_dict['open3d_time'] += (post_open3d_start - pre_open3d_end)/1e6
    
    return pcd

def pcd_denoise_dbscan(pcd: o3d.geometry.PointCloud, eps=0.02, min_points=10) -> o3d.geometry.PointCloud:
    ### Remove noise via clustering
    pcd_clusters = pcd.cluster_dbscan(
        eps=eps,
        min_points=min_points,
    )
    
    # Convert to numpy arrays
    obj_points = np.asarray(pcd.points)
    obj_colors = np.asarray(pcd.colors)
    pcd_clusters = np.array(pcd_clusters)

    # Count all labels in the cluster
    counter = Counter(pcd_clusters)

    # Remove the noise label
    if counter and (-1 in counter):
        del counter[-1]

    if counter:
        # Find the label of the largest cluster
        most_common_label, _ = counter.most_common(1)[0]
        
        # Create mask for points in the largest cluster
        largest_mask = pcd_clusters == most_common_label

        # Apply mask
        largest_cluster_points = obj_points[largest_mask]
        largest_cluster_colors = obj_colors[largest_mask]
        
        # If the largest cluster is too small, return the original point cloud
        if len(largest_cluster_points) < 5:
            return pcd

        # Create a new PointCloud object
        largest_cluster_pcd = o3d.geometry.PointCloud()
        largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
        largest_cluster_pcd.colors = o3d.utility.Vector3dVector(largest_cluster_colors)
        
        pcd = largest_cluster_pcd
        
    return pcd

def process_pcd(pcd, cfg, run_dbscan=True, frameNumer = None, caller=None, dataset_type='replica', time_dict=None):  
    voxel_size = cfg.downsample_voxel_size
    downsample_start = time.perf_counter_ns()
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    

    # Approximate the number of points to be around 2000. 
    # Done soleley for performance. Added to improve the dataset performance. Shouldn't affect the iPad performance. 
    # Quality has not degraded becausee of this approximation.
    if experiment_config is None:
        raise ValueError("Experiment config is not set")
    object_based_downsampling = experiment_config.model.mapping.object_based_downsampling
    if dataset_type in ['replica', 'scannet'] and object_based_downsampling:
        while len(pcd.points) > 2000:
            voxel_size *= cfg.dataset_config.mapping.object_based_downsampling
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pre_denoise_time = time.perf_counter_ns()
    # This is a step to remove the noise points in the point cloud. Quite expensive, so make sure that the number of points is not too large.
    if cfg.dbscan_remove_noise and run_dbscan:
        # print("Before dbscan:", len(pcd.points))
        pcd = pcd_denoise_dbscan(
            pcd, 
            eps=cfg.dbscan_eps, 
            min_points=cfg.dbscan_min_points
        )
        # print("After dbscan:", len(pcd.points))
    post_denoise_time = time.perf_counter_ns()
    # if time_dict is not None:
        # time_dict['denoise_time'] += (post_denoise_time - pre_denoise_time) / 1e6
        # time_dict['downsample_time'] += (pre_denoise_time - downsample_start) / 1e6
    return pcd

def get_bounding_box(cfg, pcd):
    if ("accurate" in cfg.spatial_sim_type or "overlap" in cfg.spatial_sim_type) and len(pcd.points) >= 4:
        try:
            return pcd.get_oriented_bounding_box(robust=True)
        except RuntimeError as e:
            print(f"Met {e}, use axis aligned bounding box instead")
            return pcd.get_axis_aligned_bounding_box()
    else:
        return pcd.get_axis_aligned_bounding_box()

def merge_obj2_into_obj1(cfg, obj1, obj2, run_dbscan=True):
    '''
    Merge the new object to the old object
    This operation is done in-place
    '''
    n_obj1_det = obj1['num_detections']
    n_obj2_det = obj2['num_detections']
    
    for k in obj1.keys():
        if k in ['caption']:
            # Here we need to merge two dictionaries and adjust the key of the second one
            for k2, v2 in obj2['caption'].items():
                obj1['caption'][k2 + n_obj1_det] = v2
        elif k == "history_idx":
                obj1[k] = obj1[k]  # Keep the initial history index
        elif k not in ['pcd', 'bbox', 'clip_ft', "text_ft"]:
            if isinstance(obj1[k], list) or isinstance(obj1[k], int):
                obj1[k] += obj2[k]
            elif k == "inst_color":
                obj1[k] = obj1[k] # Keep the initial instance color
            else:
                # TODO: handle other types if needed in the future
                raise NotImplementedError
        else: # pcd, bbox, clip_ft, text_ft are handled below
            continue

    # merge pcd and bbox
    obj1['pcd'] += obj2['pcd']
    obj1['pcd'] = process_pcd(obj1['pcd'], cfg, run_dbscan=run_dbscan, caller="merge_obj2_into_obj1")
    obj1['bbox'] = get_bounding_box(cfg, obj1['pcd'])
    obj1['bbox'].color = [0,1,0]
    
    # merge clip ft
    obj1['clip_ft'] = (obj1['clip_ft'] * n_obj1_det +
                       obj2['clip_ft'] * n_obj2_det) / (
                       n_obj1_det + n_obj2_det)
    obj1['clip_ft'] = F.normalize(obj1['clip_ft'], dim=0)

    # merge text_ft
    obj2['text_ft'] = to_tensor(obj2['text_ft'], cfg.device)
    obj1['text_ft'] = to_tensor(obj1['text_ft'], cfg.device)
    obj1['text_ft'] = (obj1['text_ft'] * n_obj1_det +
                       obj2['text_ft'] * n_obj2_det) / (
                       n_obj1_det + n_obj2_det)
    obj1['text_ft'] = F.normalize(obj1['text_ft'], dim=0)
    
    return obj1

def compute_overlap_matrix(cfg, objects: MapObjectList):
    '''
    compute pairwise overlapping between objects in terms of point nearest neighbor. 
    Suppose we have a list of n point cloud, each of which is a o3d.geometry.PointCloud object. 
    Now we want to construct a matrix of size n x n, where the (i, j) entry is the ratio of points in point cloud i 
    that are within a distance threshold of any point in point cloud j. 
    '''
    n = len(objects)
    overlap_matrix = np.zeros((n, n))
    
    # Convert the point clouds into numpy arrays and then into FAISS indices for efficient search
    point_arrays = [np.asarray(obj['pcd'].points, dtype=np.float32) for obj in objects]
    indices = [faiss.IndexFlatL2(arr.shape[1]) for arr in point_arrays]
    
    # Add the points from the numpy arrays to the corresponding FAISS indices
    for index, arr in zip(indices, point_arrays):
        index.add(arr)

    # Compute the pairwise overlaps
    for i in range(n):
        for j in range(n):
            if i != j:  # Skip diagonal elements
                box_i = objects[i]['bbox']
                box_j = objects[j]['bbox']
                
                # Skip if the boxes do not overlap at all (saves computation)
                iou = compute_3d_iou(box_i, box_j)
                if iou == 0:
                    continue
                
                # # Use range_search to find points within the threshold
                # _, I = indices[j].range_search(point_arrays[i], threshold ** 2)
                D, I = indices[j].search(point_arrays[i], 1)

                # # If any points are found within the threshold, increase overlap count
                # overlap += sum([len(i) for i in I])

                overlap = (D < cfg.downsample_voxel_size ** 1.7).sum() # D is the squared distance

                # Calculate the ratio of points within the threshold
                overlap_matrix[i, j] = overlap / len(point_arrays[i])

    return overlap_matrix

def compute_overlap_matrix_2set(cfg, objects_map: MapObjectList, objects_new: DetectionList) -> np.ndarray:
    '''
    compute pairwise overlapping between two set of objects in terms of point nearest neighbor. 
    objects_map is the existing objects in the map, objects_new is the new objects to be added to the map
    Suppose len(objects_map) = m, len(objects_new) = n
    Then we want to construct a matrix of size m x n, where the (i, j) entry is the ratio of points 
    in point cloud i that are within a distance threshold of any point in point cloud j.
    '''
    m = len(objects_map)
    n = len(objects_new)
    overlap_matrix = np.zeros((m, n))
    
    # Convert the point clouds into numpy arrays and then into FAISS indices for efficient search
    points_map = [np.asarray(obj['pcd'].points, dtype=np.float32) for obj in objects_map] # m arrays
    indices = [faiss.IndexFlatL2(arr.shape[1]) for arr in points_map] # m indices
    
    # Add the points from the numpy arrays to the corresponding FAISS indices
    for index, arr in zip(indices, points_map):
        index.add(arr)
        
    points_new = [np.asarray(obj['pcd'].points, dtype=np.float32) for obj in objects_new] # n arrays
        
    bbox_map = objects_map.get_stacked_values_torch('bbox')
    bbox_new = objects_new.get_stacked_values_torch('bbox')
    
    # try:
        # iou = compute_3d_iou_accuracte_batch(bbox_map, bbox_new) # (m, n)
    # except ValueError:
    bbox_map = []
    bbox_new = []
    for pcd in objects_map.get_values('pcd'):
        bbox_map.append(np.asarray(
            pcd.get_axis_aligned_bounding_box().get_box_points()))
    for pcd in objects_new.get_values('pcd'):
        bbox_new.append(np.asarray(
            pcd.get_axis_aligned_bounding_box().get_box_points()))
    bbox_map = torch.from_numpy(np.stack(bbox_map))
    bbox_new = torch.from_numpy(np.stack(bbox_new))
    
    iou = compute_iou_batch(bbox_map, bbox_new) # (m, n)
            

    # Compute the pairwise overlaps
    for i in range(m):
        for j in range(n):
            if iou[i,j] < 1e-6:
                continue
            
            D, I = indices[i].search(points_new[j], 1) # search new object j in map object i

            overlap = (D < cfg.downsample_voxel_size ** 1.7).sum() # D is the squared distance

            # Calculate the ratio of points within the threshold
            overlap_matrix[i, j] = overlap / len(points_new[j])

    return overlap_matrix

def merge_overlap_objects(cfg, objects: MapObjectList, overlap_matrix: np.ndarray, history_map: dict):
    x, y = overlap_matrix.nonzero()
    overlap_ratio = overlap_matrix[x, y]
    removed_objects = []
    edited_objects = []

    sort = np.argsort(overlap_ratio)[::-1]
    x = x[sort]
    y = y[sort]
    overlap_ratio = overlap_ratio[sort]

    kept_objects = np.ones(len(objects), dtype=bool)
    for i, j, ratio in zip(x, y, overlap_ratio):
        visual_sim = F.cosine_similarity(
            to_tensor(objects[i]['clip_ft']),
            to_tensor(objects[j]['clip_ft']),
            dim=0
        )
        text_sim = F.cosine_similarity(
            to_tensor(objects[i]['text_ft']),
            to_tensor(objects[j]['text_ft']),
            dim=0
        )
        if ratio > cfg.merge_overlap_thresh:
            if visual_sim > cfg.merge_visual_sim_thresh and \
                text_sim > cfg.merge_text_sim_thresh:
                if kept_objects[j]:
                    # Then merge object i into object j --- remove object i and edit object j
                    history_map[objects[i]['history_idx']] = None
                    removed_objects.append(objects[i]['history_idx'])
                    edited_objects.append(objects[j]['history_idx'])
                    objects[j] = merge_obj2_into_obj1(cfg, objects[j], objects[i], run_dbscan=True)
                    kept_objects[i] = False
        else:
            break
    
    # Remove the objects that have been merged
    new_objects = []
    for obj, keep in zip(objects, kept_objects):
        if keep:
            if obj['history_idx'] not in removed_objects:
                if history_map[obj['history_idx']] != len(new_objects):
                    history_map[obj['history_idx']] = len(new_objects)  
                    
            new_objects.append(obj)
        else:
            assert history_map[obj['history_idx']] == None, "The object should have been removed"
    # new_objects = [obj for obj, keep in zip(objects, kept_objects) if keep]
    objects = MapObjectList(new_objects)
    
    return objects, removed_objects, edited_objects, history_map

def denoise_objects(cfg, objects: MapObjectList):
    for i in range(len(objects)):
        og_object_pcd = objects[i]['pcd']
        objects[i]['pcd'] = process_pcd(objects[i]['pcd'], cfg, run_dbscan=True, frameNumer=None ,caller="denoise_objects")
        if len(objects[i]['pcd'].points) < 4:
            objects[i]['pcd'] = og_object_pcd
            continue
        objects[i]['bbox'] = get_bounding_box(cfg, objects[i]['pcd'])
        objects[i]['bbox'].color = [0,1,0]
        
    return objects

def denoise_selected_objects(cfg, objects: MapObjectList, edited_objects_indices: list):
    for i in range(len(objects)):
        hist_idx = objects[i]['history_idx']
        if hist_idx not in edited_objects_indices:
            continue
        og_object_pcd = objects[i]['pcd']
        objects[i]['pcd'] = process_pcd(objects[i]['pcd'], cfg, run_dbscan=True, frameNumer=None ,caller="denoise_selected_objects")
        if len(objects[i]['pcd'].points) < 4:
            objects[i]['pcd'] = og_object_pcd
            continue
        objects[i]['bbox'] = get_bounding_box(cfg, objects[i]['pcd'])
        objects[i]['bbox'].color = [0,1,0]
        
    return objects

def filter_objects(cfg, objects: MapObjectList, history_map: dict):
    # Remove the object that has very few points or viewed too few times
    removed_obj = []
    print("Before filtering:", len(objects))
    objects_to_keep = []
    for i,obj in enumerate(objects):
        if len(obj['pcd'].points) >= cfg.obj_min_points and obj['num_detections'] >= cfg.obj_min_detections:
            index_to_keep = len(objects_to_keep)    
            if history_map[obj['history_idx']] != index_to_keep:
                history_map[obj['history_idx']] = index_to_keep
            objects_to_keep.append(obj)
            
        else:
            history_map[obj['history_idx']] = None
            removed_obj.append(obj['history_idx'])
    objects = MapObjectList(objects_to_keep)
    print("After filtering:", len(objects))
    
    return objects, removed_obj, history_map

def merge_objects(cfg, objects: MapObjectList, history_map: dict):
    removed_objects = []
    edited_objects = []
    
    if cfg.merge_overlap_thresh > 0:
        # Merge one object into another if the former is contained in the latter
        overlap_matrix = compute_overlap_matrix(cfg, objects)
        print("Before merging:", len(objects))
        objects,removed_objects, edited_objects, history_map = merge_overlap_objects(cfg, objects, overlap_matrix, history_map=history_map)
        print("After merging:", len(objects))
    
    return objects, removed_objects, edited_objects, history_map

def filter_gobs(
    cfg: DictConfig,
    gobs: dict,
    image: np.ndarray,
    BG_CLASSES = ["wall", "floor", "ceiling"],
    pipelined_mapping=True,
):
    # If no detection at all
    if len(gobs['xyxy']) == 0:
        return gobs
    
    # Filter out the objects based on various criteria
    idx_to_keep = []
    for mask_idx in range(len(gobs['xyxy'])):
        local_class_id = gobs['class_id'][mask_idx]
        class_name = gobs['classes'][local_class_id]
        
        # SKip masks that are too small
        if gobs['mask'][mask_idx].sum() < max(cfg.mask_area_threshold, 10):
            continue
        
        # Skip the BG classes
        if cfg.skip_bg and class_name in BG_CLASSES:
            continue
        
        # Skip the non-background boxes that are too large
        if class_name not in BG_CLASSES:
            x1, y1, x2, y2 = gobs['xyxy'][mask_idx]
            bbox_area = (x2 - x1) * (y2 - y1)
            image_area = image.shape[0] * image.shape[1]
            if bbox_area > cfg.max_bbox_area_ratio * image_area:
                # print(f"Skipping {class_name} with area {bbox_area} > {cfg.max_bbox_area_ratio} * {image_area}")
                continue
            
        # Skip masks with low confidence
        if gobs['confidence'] is not None:
            if gobs['confidence'][mask_idx] < cfg.mask_conf_threshold:
                continue
        
        idx_to_keep.append(mask_idx)
    
    for k in gobs.keys():
        if isinstance(gobs[k], str) or k == "classes": # Captions
            continue
        elif isinstance(gobs[k], list):
            gobs[k] = [gobs[k][i] for i in idx_to_keep]
        elif isinstance(gobs[k], np.ndarray):
            gobs[k] = gobs[k][idx_to_keep]
        else:
            if (k == "image_feats" or k == "text_feats") and(  pipelined_mapping== False):
                continue
            raise NotImplementedError(f"Unhandled type {type(gobs[k])}, where k is {k},pipe = {pipelined_mapping}")
    
    return gobs

def resize_gobs(
    gobs,
    image
):
    n_masks = len(gobs['xyxy'])

    new_mask = []
    
    for mask_idx in range(n_masks):
        # TODO: rewrite using interpolation/resize in numpy or torch rather than cv2
        mask = gobs['mask'][mask_idx]
        if mask.shape != image.shape[:2]:
            # Rescale the xyxy coordinates to the image shape
            x1, y1, x2, y2 = gobs['xyxy'][mask_idx]
            x1 = round(x1 * image.shape[1] / mask.shape[1])
            y1 = round(y1 * image.shape[0] / mask.shape[0])
            x2 = round(x2 * image.shape[1] / mask.shape[1])
            y2 = round(y2 * image.shape[0] / mask.shape[0])
            gobs['xyxy'][mask_idx] = [x1, y1, x2, y2]
            
            # Reshape the mask to the image shape
            mask = cv2.resize(mask.astype(np.uint8), image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            mask = mask.astype(bool)
            new_mask.append(mask)

    if len(new_mask) > 0:
        gobs['mask'] = np.asarray(new_mask)
        
    return gobs
import time
def gobs_to_detection_list(
    cfg, 
    image, 
    depth_array,
    cam_K, 
    idx, 
    gobs, 
    trans_pose = None,
    class_names = None,
    BG_CLASSES = ["wall", "floor", "ceiling"],
    color_path = None,
    pipelined_mapping=True,
    dataset_type='replica',
    time_dict=None,
):
    '''
    Return a DetectionList object from the gobs
    All object are still in the camera frame. 
    '''
    global experiment_config
    if experiment_config is None:
        experiment_config = get_config()
    # cfg = experiment_config.mapping
    
    fg_detection_list = DetectionList()
    bg_detection_list = DetectionList()
    
    pcd_creation_time = []
    pcd_process_time = []
    resize_filter_start = time.perf_counter_ns()
    gobs = resize_gobs(gobs, image)
    gobs = filter_gobs(cfg, gobs, image, BG_CLASSES, pipelined_mapping=pipelined_mapping)
    
    
    if len(gobs['xyxy']) == 0:
        return fg_detection_list, bg_detection_list, None
    
    # Compute the containing relationship among all detections and subtract fg from bg objects
    xyxy = gobs['xyxy']
    mask = gobs['mask']
    gobs['mask'] = mask_subtract_contained(xyxy, mask)
    resize_filter_end = time.perf_counter_ns()
    # time_dict['resize_filter_time'] = (resize_filter_end - resize_filter_start)/1e6
    idx_to_keep = []    
    n_masks = len(gobs['xyxy'])
    time_dict['pre_open3d_time'] = 0
    time_dict['open3d_time'] = 0
    time_dict['convert_to_3d_time'] = 0
    time_dict['stacking_time'] = 0
    time_dict['perturbing_time'] = 0
    time_dict['color_time'] = 0
    
    for mask_idx in range(n_masks):
        local_class_id = gobs['class_id'][mask_idx]
        mask = gobs['mask'][mask_idx]
        class_name = gobs['classes'][local_class_id]
        global_class_id = -1 if class_names is None else class_names.index(class_name)
        pcd_start = time.perf_counter_ns()
        # make the pcd and color it
        camera_object_pcd = create_object_pcd(
            depth_array,
            mask,
            cam_K,
            image,
            obj_color = None,
            frameNumer = idx,
            time_dict=time_dict,
            cfg=cfg,
        )
        pcd_end = time.perf_counter_ns()
        
        # It at least contains 5 points
        if len(camera_object_pcd.points) < max(cfg.min_points_threshold, 5): 
            continue
        
        if trans_pose is not None:
            global_object_pcd = camera_object_pcd.transform(trans_pose)
        else:
            global_object_pcd = camera_object_pcd
        
        # get largest cluster, filter out noise 
        global_object_pcd = process_pcd(global_object_pcd, cfg, frameNumer=idx, caller="gobs_to_detection_list", dataset_type=dataset_type)
        
        pcd_bbox = get_bounding_box(cfg, global_object_pcd)
        pcd_bbox.color = [0,1,0]
        process_pcd_time = time.perf_counter_ns()
        if pcd_bbox.volume() < 1e-6:
            continue
        
        # Treat the detection in the same way as a 3D object
        # Store information that is enough to recover the detection
        detected_object = {
            'image_idx' : [idx],                             # idx of the image
            'mask_idx' : [mask_idx],                         # idx of the mask/detection
            'color_path' : [color_path],                     # path to the RGB image
            'class_name' : [class_name],                         # global class id for this detection
            'class_id' : [global_class_id],                         # global class id for this detection
            'num_detections' : 1,                            # number of detections in this object
            'mask': [mask],
            'xyxy': [gobs['xyxy'][mask_idx]],
            'conf': [gobs['confidence'][mask_idx]],
            'n_points': [len(global_object_pcd.points)],
            'pixel_area': [mask.sum()],
            'contain_number': [None],                          # This will be computed later
            "inst_color": np.random.rand(3),                 # A random color used for this segment instance
            'is_background': class_name in BG_CLASSES,
            
            # These are for the entire 3D object
            'pcd': global_object_pcd,
            'bbox': pcd_bbox,
            'history_idx': None,
        }
        if pipelined_mapping:
            detected_object['clip_ft'] = to_tensor(gobs['image_feats'][mask_idx])
            detected_object['text_ft'] = to_tensor(gobs['text_feats'][mask_idx])
        idx_to_keep.append(mask_idx)
        if class_name in BG_CLASSES:
            bg_detection_list.append(detected_object)
        else:
            fg_detection_list.append(detected_object)
        pcd_creation_time.append((pcd_end - pcd_start)/1e6)
        pcd_process_time.append((process_pcd_time - pcd_end)/1e6)
    # print(f"[MAPPING SERvER]\t\t FrameNumber: {idx} PCD creation time: ", np.sum(pcd_creation_time), "ms  PCD process time = ",  np.sum(pcd_process_time), "ms")
    time_dict['pcd_creation_time'] = np.sum(pcd_creation_time)
    time_dict['pcd_process_time'] = np.sum(pcd_process_time)
    return fg_detection_list, bg_detection_list, idx_to_keep

def transform_detection_list(
    detection_list: DetectionList,
    transform: torch.Tensor,
    deepcopy = False,
):
    '''
    Transform the detection list by the given transform
    
    Args:
        detection_list: DetectionList
        transform: 4x4 torch.Tensor
        
    Returns:
        transformed_detection_list: DetectionList
    '''
    transform = to_numpy(transform)
    
    if deepcopy:
        detection_list = copy.deepcopy(detection_list)
    
    for i in range(len(detection_list)):
        detection_list[i]['pcd'] = detection_list[i]['pcd'].transform(transform)
        detection_list[i]['bbox'] = detection_list[i]['bbox'].rotate(transform[:3, :3], center=(0, 0, 0))
        detection_list[i]['bbox'] = detection_list[i]['bbox'].translate(transform[:3, 3])
        # detection_list[i]['bbox'] = detection_list[i]['pcd'].get_oriented_bounding_box(robust=True)
    
    return detection_list

def precompute_xy_maps(width, height, fx, fy, cx, cy, dtype=np.float32):
    """Cache this per resolution/intrinsics."""
    u = np.arange(width, dtype=dtype)
    v = np.arange(height, dtype=dtype)
    uu, vv = np.meshgrid(u, v)  # (H, W)
    xu = (uu - cx) / fx         # (H, W)
    yv = (vv - cy) / fy         # (H, W)
    return xu, yv

def masks_to_labels(masks, H, W):
    """
    masks: np.bool_ array of shape (N, H, W)
    Returns int32 label image with [-1]=background, [k]=object id.
    Assumes masks are disjoint after your mask_subtract_contained().
    """
    labels = np.full((H, W), -1, dtype=np.int32)
    # If overlaps could still exist, decide priority here (last wins below):
    for k in range(len(masks)):
        m = masks[k]
        if m.dtype != np.bool_:
            m = m.astype(bool, copy=False)
        labels[m] = k
    return labels

def build_objects_points_and_colors(image, depth, xu, yv, labels,
                                    add_noise=True, sigma=4e-3,
                                    obj_color=None,
                                    time_dict=None,
                                    cfg=None,
                                    masks=None):
    """
    Do frame-wide unprojection, color once, optional jitter once.
    Returns:
      - points_by_obj: dict[obj_id] -> np.ndarray (Ni, 3) float32
      - colors_by_obj: dict[obj_id] -> np.ndarray (Ni, 3) float32
    """
    t0 = time.perf_counter_ns()

    valid = (depth > 0) & (labels >= 0)
    if not np.any(valid):
        print("No valid points")
        # if time_dict is not None:
        #     time_dict['convert_to_3d_time'] += 0.0
        #     time_dict['perturbing_time'] += 0.0
        #     time_dict['color_time'] += 0.0
        return {}, {}

    Z = depth[valid].astype(np.float32, copy=False)
    X = xu[valid].astype(np.float32, copy=False) * Z
    Y = yv[valid].astype(np.float32, copy=False) * Z

    pts = np.column_stack((X, Y, Z))  # (N, 3) float32

    t1 = time.perf_counter_ns()
    # if time_dict is not None:
    #     time_dict['convert_to_3d_time'] += (t1 - t0) / 1e6

    # Optional single jitter (tiny; skip unless you need robust OBB frequently)
    
    # n0 = time.perf_counter_ns()
    # noise = np.random.normal(0.0, sigma, size=pts.shape).astype(np.float32)
    # pts += noise
    # n1 = time.perf_counter_ns()
    # if time_dict is not None:
    #     time_dict['perturbing_time'] += (n1 - n0) / 1e6

        # Perturb the points a bit to avoid colinearity (using cheap fixed pattern)
    # if cfg is None or not hasattr(cfg, 'determinism') or not hasattr(cfg.determinism, 'disable_point_noise') or not cfg.determinism.disable_point_noise:
        # Use pre-computed perturbation pattern instead of expensive random generation
        # This is ~99% faster than np.random.normal() while still avoiding colinearity
    perturbation_pattern = np.array([
        [1e-3, -2e-3, 1e-3],
        [-2e-3, 1e-3, -1e-3], 
        [1e-3, 1e-3, -2e-3],
        [-1e-3, -1e-3, 1e-3],
        [2e-3, -1e-3, -1e-3]
    ], dtype=np.float32)
    
    # Apply pattern cyclically to points
    pattern_idx = np.arange(pts.shape[0]) % len(perturbation_pattern)
    pts += perturbation_pattern[pattern_idx]
    
    perturbing_time = time.perf_counter_ns()
    # time_dict['perturbing_time'] += (perturbing_time - t1)/1e6
   

    c0 = time.perf_counter_ns()
    if obj_color is None:
        cols = (image[valid].astype(np.float32, copy=False) / 255.0)
    else:
        cols = np.full((pts.shape[0], 3), obj_color, dtype=np.float32)
    c1 = time.perf_counter_ns()
    # if time_dict is not None:
    #     time_dict['color_time'] += (c1 - c0) / 1e6

    before_binning = time.perf_counter_ns()
    # # Bin points to objects via labels
    # lbl = labels[valid].ravel()
    # order = np.argsort(lbl, kind='stable')
    # lbl_sorted = lbl[order]
    # pts_sorted = pts[order]
    # cols_sorted = cols[order]
    # arg_srot_time = time.perf_counter_ns()
    # if time_dict is not None:
    #     time_dict['arg_sort_time'] = (arg_srot_time - before_binning) / 1e6

    # uniq, counts = np.unique(lbl_sorted, return_counts=True)
    # offsets = np.cumsum(counts)
    # unique_time = time.perf_counter_ns()
    # if time_dict is not None:
    #     time_dict['unique_time'] = (unique_time - arg_srot_time) / 1e6

    # points_by_obj, colors_by_obj = {}, {}
    # start = 0
    # for obj_id, end in zip(uniq.tolist(), offsets.tolist()):
    #     points_by_obj[obj_id] = pts_sorted[start:end]
    #     colors_by_obj[obj_id] = cols_sorted[start:end]
    #     start = end





        # ---- Bin to objects without sort/unique ----
    # Map between full-frame linear indices and rows in pts/cols
    valid_idx = np.flatnonzero(valid.ravel())        # (N,)
    # Avoid dtype churn in the loop
    valid_idx = np.ascontiguousarray(valid_idx, dtype=np.int64)

    points_by_obj, colors_by_obj = {}, {}

    # Tight loop: for each mask, select only among valid pixels (cheap)
    # NOTE: do not create a big (K x N) temporary; index per-mask to keep memory low.
    for k in range(masks.shape[0]):
        m_flat = masks[k].ravel()                    # view, no copy
        sel = m_flat[valid_idx]                      # boolean over N valid pixels
        if not np.any(sel):
            continue
        rows = np.flatnonzero(sel)                   # indices into pts/cols
        # Slice views; Open3D will copy later anyway
        points_by_obj[k] = pts[rows]
        colors_by_obj[k] = cols[rows]

    


    after_binning = time.perf_counter_ns()
    # if time_dict is not None:
    #     time_dict['binning_time'] = (after_binning - before_binning) / 1e6
        # time_dict['loop_time'] = (after_binning - unique_time) / 1e6

    return points_by_obj, colors_by_obj

def frame_to_object_pcds(depth_array, masks, cam_K, image,
                         add_noise=False, sigma=4e-3,
                         precomputed_xy=None,
                         time_dict=None,
                         cfg=None):
    """
    Returns a dict: obj_id -> open3d PointCloud
    - masks: np.bool_ array of shape (N, H, W) for the *filtered* detections
    - Maintains your timing keys in time_dict (convert_to_3d_time, perturbing_time, color_time, pre_open3d_time, open3d_time)
    """
    fx, fy, cx, cy = from_intrinsics_matrix(cam_K)
    H, W = depth_array.shape

    if precomputed_xy is None:
        xu, yv = precompute_xy_maps(W, H, fx, fy, cx, cy)
    else:
        xu, yv = precomputed_xy  # (H, W) each

    labels = masks_to_labels(masks, H, W)

    pre_o3d_start = time.perf_counter_ns()
    # single-pass points/colors
    pts_by_obj, cols_by_obj = build_objects_points_and_colors(
        image=image,
        depth=depth_array,
        xu=xu, yv=yv,
        labels=labels,
        add_noise=add_noise,
        sigma=sigma,
        obj_color=None,
        time_dict=time_dict,
        cfg=cfg,
        masks=masks
    )

    # Wrap in Open3D
    pc_dict = {}
    pre_o3d_t0 = time.perf_counter_ns()
    
    for obj_id, pts in pts_by_obj.items():
        cols = cols_by_obj[obj_id]
        pts  = np.ascontiguousarray(pts,  dtype=np.float64)
        cols = np.ascontiguousarray(cols, dtype=np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols)
        pc_dict[obj_id] = pcd
    pre_o3d_t1 = time.perf_counter_ns()
    # if time_dict is not None:
    #     # amortize pre/post O3D attribution like before
    #     time_dict['open3d_time'] += (pre_o3d_t1 - pre_o3d_t0) / 1e6
    #     time_dict['pre_open3d_time'] += (pre_o3d_t0 - pre_o3d_start) / 1e6


    return pc_dict


def gobs_to_detection_list_optimized(
    cfg, 
    image, 
    depth_array,
    cam_K, 
    idx, 
    gobs, 
    trans_pose = None,
    class_names = None,
    BG_CLASSES = ["wall", "floor", "ceiling"],
    color_path = None,
    pipelined_mapping=True,
    dataset_type='replica',
    time_dict=None,
):
    global experiment_config
    if experiment_config is None:
        experiment_config = get_config()
    # cfg = experiment_config.mapping
    """
    Entry point: now uses a single frame-wide unprojection.
    """
    fg_detection_list = DetectionList()
    bg_detection_list = DetectionList()

    pcd_creation_times_ms = []
    pcd_process_times_ms = []

    resize_filter_start = time.perf_counter_ns()
    gobs = resize_gobs(gobs, image)
    gobs = filter_gobs(cfg, gobs, image, BG_CLASSES, pipelined_mapping=pipelined_mapping)
    resize_filter_end = time.perf_counter_ns()
    # if time_dict is not None:
    #     time_dict['resize_filter_time'] = (resize_filter_end - resize_filter_start)/1e6

    if len(gobs['xyxy']) == 0:
        return fg_detection_list, bg_detection_list, []

    # Subtract containment as before
    xyxy = gobs['xyxy']
    mask = gobs['mask']
    gobs['mask'] = mask_subtract_contained(xyxy, mask)

    # Init timing buckets (same keys you used before)
    if time_dict is not None:
        time_dict['pre_open3d_time'] = 0.0
        time_dict['open3d_time'] = 0.0
        time_dict['convert_to_3d_time'] = 0.0
        time_dict['perturbing_time'] = 0.0
        time_dict['color_time'] = 0.0
        time_dict['denoise_time'] = 0.0
        time_dict['downsample_time'] = 0.0

    n_masks = len(gobs['xyxy'])
    if n_masks == 0:
        return fg_detection_list, bg_detection_list, []

    # ---- Frame-wide unprojection ----
    make_pcds_start = time.perf_counter_ns()
    masks = np.asarray(gobs['mask'], dtype=np.bool_)  # (N, H, W)
    obj_pcds = frame_to_object_pcds(
        depth_array=depth_array,
        masks=masks,
        cam_K=cam_K,
        image=image,
        add_noise=False,            # ← usually safe to skip; set True if you see OBB degeneracy
        sigma=4e-3,
        precomputed_xy=None,
        time_dict=time_dict,
        cfg=cfg,
    )
    make_pcds_end = time.perf_counter_ns()
    pcd_creation_times_ms.append((make_pcds_end - make_pcds_start)/1e6)

    idx_to_keep = []

    # ---- Per-object post-processing (downsample/DBSCAN/OBB etc.) ----
    for mask_idx in range(n_masks):
        if mask_idx not in obj_pcds:
            continue  # e.g., mask produced zero valid depth points

        local_class_id = gobs['class_id'][mask_idx]
        class_name = gobs['classes'][local_class_id]
        global_class_id = -1 if class_names is None else class_names.index(class_name)

        camera_object_pcd = obj_pcds[mask_idx]

        # Minimum points check
        if len(camera_object_pcd.points) < max(cfg.min_points_threshold, 5):
            continue

        # Transform to world if provided
        if trans_pose is not None:
            global_object_pcd = camera_object_pcd.transform(trans_pose)
        else:
            global_object_pcd = camera_object_pcd

        # Denoise/downsample as before
        pcd_post_start = time.perf_counter_ns()
        global_object_pcd = process_pcd(global_object_pcd, cfg, frameNumer=idx,
                                        caller="gobs_to_detection_list_batched",
                                        dataset_type=dataset_type,
                                        time_dict=time_dict)

        # BBox
        pcd_bbox = get_bounding_box(cfg, global_object_pcd)
        pcd_bbox.color = [0, 1, 0]

        pcd_post_end = time.perf_counter_ns()
        pcd_process_times_ms.append((pcd_post_end - pcd_post_start)/1e6)

        # Reject degenerate bbox
        if pcd_bbox.volume() < 1e-6:
            continue

        # Build detection dict (unchanged shape)
        detected_object = {
            'image_idx' : [idx],
            'mask_idx' : [mask_idx],
            'color_path' : [color_path],
            'class_name' : [class_name],
            'class_id' : [global_class_id],
            'num_detections' : 1,
            'mask': [gobs['mask'][mask_idx]],
            'xyxy': [gobs['xyxy'][mask_idx]],
            'conf': [gobs['confidence'][mask_idx]],
            'n_points': [len(global_object_pcd.points)],
            'pixel_area': [gobs['mask'][mask_idx].sum()],
            'contain_number': [None],
            "inst_color": np.random.rand(3),
            'is_background': class_name in BG_CLASSES,

            'pcd': global_object_pcd,
            'bbox': pcd_bbox,
            'history_idx': None,
        }
        if pipelined_mapping:
            detected_object['clip_ft'] = to_tensor(gobs['image_feats'][mask_idx])
            detected_object['text_ft'] = to_tensor(gobs['text_feats'][mask_idx])

        idx_to_keep.append(mask_idx)
        if class_name in BG_CLASSES:
            bg_detection_list.append(detected_object)
        else:
            fg_detection_list.append(detected_object)

    # Log timings like before
    # print(f"[MAPPING SERVER]\t FrameNumber: {idx} PCD creation(batch) = {pcd_creation_times_ms[0]:.2f} ms PCD postprocess = {pcd_process_times_ms[0]:.2f} ms")
    if time_dict is not None:
        time_dict['pcd_creation_time'] = np.sum(pcd_creation_times_ms)
        time_dict['pcd_process_time']  = np.sum(pcd_process_times_ms)

    return fg_detection_list, bg_detection_list, idx_to_keep
