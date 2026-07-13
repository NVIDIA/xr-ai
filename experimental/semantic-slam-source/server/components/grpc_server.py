# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""gRPC server component for SLAM system."""

import os
import time
import numpy as np
from typing import Optional, Iterator, Any
from PIL import Image
import grpc

from config.settings import Config, get_config
from .video_processor import VideoProcessor
from .inference_service import InferenceService

# Import generated protobuf classes from parent package
from server import xr_service_pb2, xr_service_pb2_grpc


class SLAMGRPCServer(xr_service_pb2_grpc.XrServiceServicer):
    """Clean, focused gRPC server for SLAM operations."""
    
    def __init__(self, 
                 inference_queue,
                 config: Optional[Config] = None):
        """Initialize SLAM gRPC server.
        
        Args:
            inference_queue: Queue for inference requests
            config: Configuration object
        """
        self.config = config if config else get_config()
        
        # Initialize components
        self.video_processor = VideoProcessor(self.config)
        self.inference_service = InferenceService(inference_queue, self.config)
        
        # Dataset saving configuration from config
        self.data_dump_enabled = self.config.dataset.enabled
        self._setup_dataset_saving()
    
    def _setup_dataset_saving(self) -> None:
        """Setup dataset saving if enabled."""
        if not self.data_dump_enabled:
            return
            
        # Use configured base directory
        base_dir = self.config.dataset.output_directory
        
        if self.config.dataset.auto_increment_dirs:
            # Find next available dataset directory
            if os.path.exists(base_dir):
                dataset_dirs = [d for d in os.listdir(base_dir) 
                               if d.startswith('dataset_')]
                if dataset_dirs:
                    dataset_numbers = [int(d.split('_')[-1]) for d in dataset_dirs]
                    dataset_number = max(dataset_numbers) + 1
                else:
                    dataset_number = 0
                
                self.dataset_output_dir = os.path.join(base_dir, f'dataset_{dataset_number}')
            else:
                os.makedirs(base_dir, exist_ok=True)
                self.dataset_output_dir = os.path.join(base_dir, 'dataset_0')
        else:
            self.dataset_output_dir = base_dir
            
        os.makedirs(self.dataset_output_dir, exist_ok=True)
        
        # Create subdirectories based on config
        results_subdir = self.config.dataset.results_subdir
        self.image_folder_path = os.path.join(self.dataset_output_dir, results_subdir)
        self.depth_folder_path = os.path.join(self.dataset_output_dir, results_subdir)
        self.pose_folder_path = self.dataset_output_dir
        
        os.makedirs(self.image_folder_path, exist_ok=True)
        os.makedirs(self.depth_folder_path, exist_ok=True)
        print(f"\n\n\n\n\n\n\n Following directories were created: {self.image_folder_path}, {self.depth_folder_path}, {self.pose_folder_path}")
    
    def _save_dataset_files(self, 
                           client_frame_number: int,
                           image_pil: Image.Image,
                           depth_array: np.ndarray, 
                           pose_string: str,
                           is_dataset: bool = False) -> None:
        """Save frame data for dataset if enabled.
        
        Args:
            client_frame_number: Frame number from client
            image_pil: PIL Image object
            depth_array: Depth data
            pose_string: Pose information
        """
        if not self.data_dump_enabled:
            return
            
        try:
            # Save image
            image_filename = f"frame{client_frame_number:06d}.jpg"
            image_path = os.path.join(self.image_folder_path, image_filename)
            image_pil.save(image_path)
            
            # Save depth
            if is_dataset:
                depth_filename = f"depth{client_frame_number:06d}.png"
                depth_path = os.path.join(self.depth_folder_path, depth_filename)
                depth_image = Image.fromarray(depth_array.astype(np.uint16))
                depth_image.save(depth_path)
            else:
                depth_filename_npy = f"depth{client_frame_number:06d}.npy"
                depth_filename_png = f"depth{client_frame_number:06d}.png"
                depth_array = depth_array.reshape((144,256))
                has_nonzero_depth = np.any(depth_array != 0)
                if not has_nonzero_depth:
                    print(f"************************** Depth array is all zeros for frame {client_frame_number}, skipping save **************************")
                    return
                depth_path = os.path.join(self.depth_folder_path, depth_filename_npy)
                np.save(depth_path, depth_array)
                depth_image = Image.fromarray(depth_array.astype(np.uint16))
                depth_path_png = os.path.join(self.depth_folder_path, depth_filename_png)
                depth_image.save(depth_path_png)
            
            # Save pose to trajectory file
            pose_file = os.path.join(self.pose_folder_path, "traj.txt")
            with open(pose_file, 'a') as f:
                f.write(f"{pose_string}\n")
                
        except Exception as e:
            print(f"Error saving dataset files: {e}")
    
    def _process_frame_request(self, request, is_dataset: bool = False) -> Optional[xr_service_pb2.VideoStatus]:
        """Process a single frame request.
        
        Args:
            request: gRPC request object
            is_dataset: Whether this is a dataset request
            
        Returns:
            Upload response or None if processing failed
        TODO:: check if this is correct!!!!!!!!!!!!! Very last minute change!!!!!!!!!!!!! (Seems ok for now? -- Jan 6th 2026)
        """
        try:
            # Extract frame data (same for both message types)
            client_frame_number = request.image.frame_number
            client_timestamp = getattr(request, 'timestamp_ns', request.image.timestamp_us * 1000)  # Convert µs to ns if needed
            server_timestamp = time.perf_counter_ns()
            
            # Check if frame should be processed
            if not self.inference_service.should_process_frame(client_frame_number):
                return xr_service_pb2.VideoStatus(success=True)
            
            # Process H264 encoded video frame
            h264_data = None
            if hasattr(request.image, 'data_h265') and request.image.data_h265:
                h264_data = request.image.data_h265  # Note: despite the name, this is H264 data
            
            # Decode H264 to get PIL Image and numpy array
            image_pil = None
            image_array = None
            if h264_data:
                # Use video processor to decode H264 and get image data
                processed_frame, is_valid = self.video_processor.process_video_frame(h264_data)
                
                if not is_valid:
                    print(f"Frame {client_frame_number} rejected (quality check)")
                    return xr_service_pb2.VideoStatus(success=False)
                
                # processed_frame should be the decoded image
                if processed_frame is not None:
                    if isinstance(processed_frame, Image.Image):
                        image_pil = processed_frame
                        image_array = np.array(processed_frame)
                    elif isinstance(processed_frame, np.ndarray):
                        image_array = processed_frame
                        image_pil = Image.fromarray(processed_frame)
            
            # Convert depth data - different formats for dataset vs regular
            if is_dataset:
                # Dataset format: bytes depth
                depth_array = np.frombuffer(request.depth, dtype=np.uint16)

                scaling_factor = request.scaling_factor
                # reshape the depth array to the height and width of the image
                depth_array = depth_array.reshape((image_array.shape[0]//scaling_factor, image_array.shape[1]//scaling_factor))
            else:
                # Regular format: repeated float depthArr - TODO:: check if this is correct!!!!!!!!!!!!!
                depth_array = np.array(request.depthArr, dtype=np.float32)
              
            # Extract pose data - different formats for dataset vs regular
            if is_dataset:
                # Dataset format: repeated float pose
                pose_values = list(request.pose)
                pose_data = [client_frame_number] + pose_values
            else:
                # Regular format: PoseData object
                pose_data = [
                    client_frame_number,
                    request.pose.pose0_0, request.pose.pose0_1, request.pose.pose0_2, request.pose.pose0_3,
                    request.pose.pose1_0, request.pose.pose1_1, request.pose.pose1_2, request.pose.pose1_3,
                    request.pose.pose2_0, request.pose.pose2_1, request.pose.pose2_2, request.pose.pose2_3,
                    request.pose.pose3_0, request.pose.pose3_1, request.pose.pose3_2, request.pose.pose3_3
                ]
            
            # Save dataset files if enabled
            pose_string = " ".join(map(str, pose_data[1:]))  # Skip frame number for pose string
            if self.data_dump_enabled:
                self._save_dataset_files(client_frame_number, image_pil, depth_array, pose_string, is_dataset)
            
            # Submit to inference
            success = self.inference_service.submit_inference_request(
                image_pil=image_pil,
                image_array=image_array,
                depth_array=depth_array,
                pose=pose_data,
                client_frame_number=client_frame_number,
                client_timestamp=client_timestamp,
                server_timestamp=server_timestamp,
                metadata={'is_dataset': is_dataset}
            )
            
            if success:
                return xr_service_pb2.VideoStatus(success=True)
            else:
                return xr_service_pb2.VideoStatus(success=False)
                
        except Exception as e:
            error_msg = f"Error processing frame: {e}"
            print(error_msg)
            return xr_service_pb2.VideoStatus(success=False)
    
    def UploadSyncMessage(self, 
                         request_iterator, 
                         context):
        """Handle streaming upload requests from clients.
        
        Args:
            request_iterator: Iterator of upload requests
            context: gRPC context
            
        Returns:
            Single VideoStatus response
        """
        success_count = 0
        total_count = 0
        
        for request in request_iterator:
            total_count += 1
            response = self._process_frame_request(request, is_dataset=False)
            if response and response.success:
                success_count += 1
        
        # Return a single VideoStatus indicating overall success
        overall_success = success_count == total_count and total_count > 0
        return xr_service_pb2.VideoStatus(success=overall_success)
    
    def UploadSyncMessage_dataset(self, 
                                 request_iterator, 
                                 context):
        """Handle streaming upload requests for dataset processing.
        
        Args:
            request_iterator: Iterator of upload requests  
            context: gRPC context
            
        Returns:
            Single VideoStatus response
        """
        print(f"\n\n\n GRPC Server: UploadSyncMessage_dataset called")  
        success_count = 0
        total_count = 0
        
        for request in request_iterator:
            total_count += 1
            print(f"Processing frame {request.image.frame_number}")
            response = self._process_frame_request(request, is_dataset=True)
            if response and response.success:
                success_count += 1
        
        # Return a single VideoStatus indicating overall success
        overall_success = success_count == total_count and total_count > 0
        print(f"Dataset processing completed: {success_count}/{total_count} frames successful")
        return xr_service_pb2.VideoStatus(success=overall_success)
                
    def get_server_stats(self) -> dict:
        """Get current server statistics.
        
        Returns:
            Dictionary with server stats
        """
        inference_stats = self.inference_service.get_processing_stats()
        
        return {
            'inference': inference_stats,
            'video_processor': {
                'temp_dir': self.video_processor.temp_output_dir,
                'sharpness_threshold': self.video_processor.sharpness_threshold
            },
            'dataset_saving': {
                'enabled': self.data_dump_enabled,
                'output_dir': getattr(self, 'dataset_output_dir', None)
            }
        }