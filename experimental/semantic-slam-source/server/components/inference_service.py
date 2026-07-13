# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference service component for SLAM server."""

import time
import multiprocessing
from typing import Optional, Tuple, Any
from queue import Queue
import numpy as np
from PIL import Image

from config.settings import Config, get_config


class InferenceService:
    """Handles inference coordination and queue management."""
    
    def __init__(self, 
                 inference_queue: multiprocessing.Queue,
                 config: Optional[Config] = None):
        """Initialize inference service.
        
        Args:
            inference_queue: Queue for sending inference requests
            config: Configuration object, defaults to global config
        """
        self.inference_queue = inference_queue
        self.config = config if config else get_config()
        
        # Frame tracking from config
        self.last_frame_index = self.config.frame_processing.initial_frame_index
        self.tick_num = 0
        
        # FPS control from config
        self.target_fps = self.config.server.target_fps
        self.client_fps = self.config.frame_processing.default_client_fps
        
    def should_process_frame(self, frame_number: int) -> bool:
        """Determine if frame should be processed based on FPS target.
        
        Args:
            frame_number: Current frame number
            
        Returns:
            True if frame should be processed
        """
        # Skip frames that are too close to last processed frame
        frame_diff = frame_number - self.last_frame_index
        min_frame_gap = max(1, int(self.client_fps / self.target_fps))
        
        return frame_diff >= min_frame_gap
    
    def submit_inference_request(self, 
                                image_pil: Image.Image,
                                image_array: np.ndarray,
                                depth_array: np.ndarray,
                                pose: list,
                                client_frame_number: int,
                                client_timestamp: int,
                                server_timestamp: int,
                                metadata: dict) -> bool:
        """Submit inference request to processing queue.
        
        Args:
            image_pil: PIL Image object
            image_array: Image as numpy array
            depth_array: Depth data as numpy array  
            pose: Camera pose information
            client_frame_number: Frame number from client
            client_timestamp: Timestamp from client
            server_timestamp: Server-side timestamp
            metadata: Additional metadata
            
        Returns:
            True if request was successfully queued
        """
        if self.inference_queue.full():
            print(f"⚠️  [GRPC→INFERENCE] Queue full! Dropping frame {client_frame_number}")
            return False
        
        try:
            inference_data = (
                image_pil,
                image_array, 
                depth_array,
                pose,
                client_frame_number,
                client_timestamp,
                server_timestamp,
                metadata
            )
            
            self.inference_queue.put(inference_data)
            self.last_frame_index = client_frame_number
            self.tick_num += 1
            
            return True
            
        except Exception as e:
            print(f"Error submitting inference request: {e}")
            return False
    
    def get_processing_stats(self) -> dict:
        """Get current processing statistics.
        
        Returns:
            Dictionary with processing stats
        """
        return {
            'last_frame_index': self.last_frame_index,
            'tick_num': self.tick_num,
            'target_fps': self.target_fps,
            'queue_size': self.inference_queue.qsize() if hasattr(self.inference_queue, 'qsize') else -1
        }
    
    def wait_for_next_frame(self) -> None:
        """Wait appropriate time before processing next frame."""
        time.sleep(1.0 / self.target_fps)