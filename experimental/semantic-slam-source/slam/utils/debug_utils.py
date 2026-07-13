# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Debug and persistence utilities for semantic SLAM operations."""

import gzip
import os, sys
import pickle
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import cv2
import numpy as np
import supervision as sv
from PIL import Image
from supervision.draw.color import ColorPalette

from slam.utils.general_utils import to_numpy
from slam.core.utils import get_classes_colors, denoise_objects
from slam.core.slam_classes import MapObjectList
from slam.utils.vis import vis_result_fast, vis_result_slow_caption

def dump_inference_results(results: Dict[str, Any], frame_number: int, image_pil, depth_array: np.ndarray, 
                          pose_array: np.ndarray, classes, config=None):
    """Dump inference results and annotated visualizations for reproducibility debugging.
    
    This function saves:
    1. Pickle files with inference data (detections, features, etc.) in ./debug_dumps/{scene}/inference/
    2. Annotated images with detection visualizations in ./debug_dumps/{scene}/visualizations/
    
    Configuration options (in config.debug):
    - dump_inference: Enable/disable the dumping (default: false)
    - dump_dir: Base directory for debug dumps (default: "./debug_dumps")
    - use_slow_vis: Enable high-quality caption visualization (default: false, slower)
    
    Environment variables:
    - DEBUG_DUMP_INFERENCE: Override dump_inference setting
    - DEBUG_DUMP_DIR: Override dump_dir setting
    - DEBUG_USE_SLOW_VIS: Override use_slow_vis setting
    - SLAM_SCENE_NAME: Scene name for organized file naming
    - SLAM_CONFIG_NAME: Config name for organized file naming
    
    Example usage:
    To enable debug dumps with visualizations:
    export DEBUG_DUMP_INFERENCE=true
    export DEBUG_USE_SLOW_VIS=true
    python server/main.py --dataset_type replica --localDataset --sceneName room0
    """
    try:
        # Get config if not provided
        if config is None:
            from config.settings import get_config
            config = get_config()
        
        # Create filename with scene info
        scene_name = getattr(config.debug, 'current_scene', None) or os.environ.get('SLAM_SCENE_NAME', 'unknown_scene')
        config_name = getattr(config.debug, 'config_name', None) or os.environ.get('SLAM_CONFIG_NAME', 'unknown_config')
        
        # Create debug dump directories
        if hasattr(config.debug, 'dump_dir'):
            base_dump_dir = Path(config.debug.dump_dir) / scene_name
        else:
            base_dump_dir = Path(os.environ.get('SLAM_DEBUG_DUMP_DIR', './debug_dumps')) / scene_name
            
        # Create subdirectories for different types of debug data
        inference_dump_dir = base_dump_dir / "inference"
        vis_dump_dir = base_dump_dir / "visualizations"
        inference_dump_dir.mkdir(parents=True, exist_ok=True)
        vis_dump_dir.mkdir(parents=True, exist_ok=True)
            
        # Prepare data to dump (similar to generate_gsa_results.py format)
        dump_data = {
            "xyxy": results["xyxy"],
            "confidence": results["confidence"], 
            "class_id": results["class_id"],
            "mask": results["mask"],
            "classes": classes,
            "image_feats": results["image_feats"],
            "text_feats": results["text_feats"],
            # Additional debug info
            "frame_number": frame_number,
            "scene_name": scene_name,
            "config_name": config_name,
            "pose_array": pose_array,
            "depth_shape": depth_array.shape,
            "image_size": image_pil.size,
        }
        
        if "tagging_caption" in results:
            dump_data["tagging_caption"] = results["tagging_caption"]
            dump_data["tagging_text_prompt"] = results["tagging_text_prompt"]
        
        # Save pickle data
        dump_filename = f"{scene_name}_{config_name}_frame{frame_number:06d}.pkl.gz"
        dump_path = inference_dump_dir / dump_filename
        with gzip.open(dump_path, "wb") as f:
            pickle.dump(dump_data, f)
        
        # Create annotated visualizations
        _dump_annotated_images(results, frame_number, image_pil, classes, vis_dump_dir, scene_name, config_name, config)
            
        if frame_number % 50 == 0:  # Print every 50th frame to avoid spam
            print(f"💾 [DEBUG DUMP] Inference results and visualizations saved for frame {frame_number}")
            
    except Exception as e:
        print(f"❌ [DEBUG DUMP] Failed to dump inference results for frame {frame_number}: {e}")


def _dump_annotated_images(results: Dict[str, Any], frame_number: int, image_pil, classes, 
                          vis_dump_dir: Path, scene_name: str, config_name: str, config):
    """Create and save annotated images using vis.py functions."""
    try:
        # Convert PIL image to numpy array for visualization
        image_rgb = np.array(image_pil)
        
        # Create supervision detections object from results
        detections = sv.Detections(
            xyxy=results["xyxy"],
            confidence=results.get("confidence"),
            class_id=results.get("class_id"),
            mask=results.get("mask"),
        )
        
        # Check if we have valid detections
        if len(detections) == 0:
            # Save original image if no detections
            vis_save_path = vis_dump_dir / f"{scene_name}_{config_name}_frame{frame_number:06d}_no_detections.jpg"
            image_pil.save(vis_save_path)
            return
        
        # Get visualization timing
        vis_start = time.perf_counter_ns()
        
        # Create fast annotated image with default color palette
        try:
            color_palette = ColorPalette.DEFAULT
        except:
            color_palette = ColorPalette.default()
            
        annotated_image, labels = vis_result_fast(image_rgb, detections, classes, color_palette)
        
        # Save the fast annotated image
        vis_save_path = vis_dump_dir / f"{scene_name}_{config_name}_frame{frame_number:06d}_annotated.jpg"
        cv2.imwrite(str(vis_save_path), annotated_image)
        
        # Create slow caption visualization if we have caption data and config allows it
        use_slow_vis = getattr(config.debug, 'use_slow_vis', False)
        if ("tagging_caption" in results and "tagging_text_prompt" in results and use_slow_vis):
            try:
                caption = results["tagging_caption"]
                text_prompt = results["tagging_text_prompt"]
                
                # For slow visualization, we need to convert masks and boxes to the expected format
                masks = results["mask"] if results["mask"] is not None else []
                boxes_filt = results["xyxy"] if results["xyxy"] is not None else []
                
                annotated_image_caption = vis_result_slow_caption(
                    image_rgb, masks, boxes_filt, labels, caption, text_prompt
                )
                
                # Save the slow caption annotated image
                caption_save_path = vis_dump_dir / f"{scene_name}_{config_name}_frame{frame_number:06d}_caption.jpg"
                Image.fromarray(annotated_image_caption).save(caption_save_path)
                
            except Exception as e:
                print(f"⚠️ [DEBUG VIS] Failed to create slow caption visualization for frame {frame_number}: {e}")
        
        vis_end = time.perf_counter_ns()
        
        # Log timing occasionally
        if frame_number % 100 == 0:
            vis_time_ms = (vis_end - vis_start) / 1e6
            print(f"🎨 [DEBUG VIS] Frame {frame_number} visualization took {vis_time_ms:.1f}ms")
            
    except Exception as e:
        print(f"❌ [DEBUG VIS] Failed to create annotated images for frame {frame_number}: {e}")


def dump_semantic_map(scene_name: str, objects, bg_objects: Optional[Dict], cfg, classes, save_map: bool = False, test_depth_downsampling: int = 1):
    """Dump semantic map data for a completed scene."""
    
    # Check if we should save maps 
    if not save_map:
        print(f"💾 Scene {scene_name} completed but --save_map not specified, skipping map dump")
        return
        
    if not objects or len(objects) == 0:
        print(f"⚠️  No map data to save for scene {scene_name}")
        return
        
    print(f"💾 [DUMP SEMANTIC MAP] Dumping semantic map for scene {scene_name}...")
    
    # Generate class colors from classes
    class_colors = get_classes_colors(classes) if classes else {}
    if bg_objects is not None:
        bg_objects = MapObjectList([_ for _ in bg_objects.values() if _ is not None])
        bg_objects = denoise_objects(cfg, bg_objects)
    # Create results dictionary in the expected format
    results = {
        'objects': objects.to_serializable(),
        'bg_objects': None if bg_objects is None else bg_objects.to_serializable(),
        'cfg': cfg,
        'class_names': classes,
        'class_colors': class_colors,
        'scene_name': scene_name,
        'timestamp': datetime.now().isoformat(),
    }
    
    # Create save path
    dataset_root = Path(os.environ.get('REPLICA_ROOT', './output'))
    config_name = os.environ.get('SLAM_CONFIG_NAME', 'default')
    save_dir = dataset_root / scene_name / 'pcd_saves' / 'real_time_sys_indi'   
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"semantic_map_{config_name}_test_depth_downsampling_{test_depth_downsampling}.pkl.gz"
    
    # Save the map with signal protection
    def signal_handler(signum, frame):
        print(f"⚠️  Signal {signum} received during map save - waiting for completion...")
        
    # Block signals during file write to prevent corruption
    old_handler = signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        print(f"💾 Writing semantic map file...")
        with gzip.open(save_path, "wb") as f:
            pickle.dump(results, f)
            f.flush()  # Ensure data is written to disk
            os.fsync(f.fileno())  # Force OS to write to disk
            # File is automatically closed when exiting 'with' block
        
        print(f"✅ Saved semantic map to {save_path}")
        print(f"   - Config: {config_name}")
        print(f"   - Scene: {scene_name}")
        print(f"   - {len(objects)} objects")
        print(f"   - {len(classes)} classes: {classes}")
    except Exception as e:
        print(f"❌ Error saving semantic map: {e}")
    finally:
        # Restore original signal handler
        signal.signal(signal.SIGTERM, old_handler)
