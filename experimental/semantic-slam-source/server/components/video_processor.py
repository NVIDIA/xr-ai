# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video processing component for SLAM server."""

import cv2
import os
import tempfile
from typing import Tuple, Optional
import numpy as np
from PIL import Image

from config.settings import Config, get_config
from ..video_decoders import VideoDecoder


class VideoProcessor:
    """Handles video processing operations with clean separation of concerns."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize video processor.
        
        Args:
            config: Configuration object, defaults to global config
        """
        self.config = config if config else get_config()
        
        # Video processing settings from config
        self.image_width = self.config.video.image_width
        self.image_height = self.config.video.image_height
        self.depth_width = self.config.video.depth_width
        self.depth_height = self.config.video.depth_height
        self.dataset_depth_height = self.config.video.dataset_depth_height
        self.dataset_depth_width = self.config.video.dataset_depth_width
        self.sharpness_threshold = self.config.video.sharpness_threshold
        
        # Initialize video decoder
        self.h264_decoder = VideoDecoder('h264')
        
        # Setup temp directory from config
        self.temp_output_dir = self.config.video.temp_output_dir
        if not os.path.exists(self.temp_output_dir):
            os.makedirs(self.temp_output_dir)
        self._cleanup_temp_directory()
    
    def _cleanup_temp_directory(self) -> None:
        """Clean up temporary output directory."""
        if os.path.exists(self.temp_output_dir):
            for file in os.listdir(self.temp_output_dir):
                file_path = os.path.join(self.temp_output_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
    
    def calculate_sharpness(self, image: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Sharpness value (higher = sharper)
        """
        # Convert to BGR if needed for cv2
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
            
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def is_frame_sharp_enough(self, image: np.ndarray) -> bool:
        """Check if frame meets sharpness threshold.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            True if frame is sharp enough for processing
        """
        return self.calculate_sharpness(image) >= self.sharpness_threshold

    def save_frame_data(self,
                       client_frame_number: int,
                       image_pil: Image.Image, 
                       depth_array: np.ndarray, 
                       pose_string: str) -> Tuple[str, str]:
        """Save frame data to temporary files.
        
        Args:
            client_frame_number: Frame number from client
            image_pil: PIL Image object
            depth_array: Depth data as numpy array
            pose_string: Pose information as string
            
        Returns:
            Tuple of (image_path, depth_path)
        """
        # Save image
        image_filename = f"frame-{client_frame_number}.jpg"
        image_path = os.path.join(self.temp_output_dir, image_filename)
        image_pil.save(image_path)
        
        # Save depth data
        depth_filename = f"depth-{client_frame_number}.npy"
        depth_path = os.path.join(self.temp_output_dir, depth_filename)
        np.save(depth_path, depth_array)
        
        return image_path, depth_path
    
    def process_video_frame(self, frame_data: bytes) -> Tuple[Optional[np.ndarray], bool]:
        """Process raw video frame data.
        
        Args:
            frame_data: Raw video frame bytes
            
        Returns:
            Tuple of (processed_frame, is_valid)
        """
        try:
            # Decode frame using h264 decoder
            decoded_frames = self.h264_decoder.decode_frame(frame_data)
            
            if not decoded_frames:
                return None, False
            
            # Get the first frame (numpy array, RGB HWC, from NVDEC)
            decoded_frame = decoded_frames[0]
            
            # Check if frame is sharp enough
            if not self.is_frame_sharp_enough(decoded_frame):
                return decoded_frame, False
                
            return decoded_frame, True
            
        except Exception as e:
            print(f"Error processing video frame: {e}")
            return None, False