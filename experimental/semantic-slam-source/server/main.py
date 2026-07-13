# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Main gRPC server for semantic SLAM operations."""

import argparse
import asyncio
import cv2
import grpc
import io
import multiprocessing
import numpy as np
import os
import signal
import subprocess
import sys
import tempfile
import time
from concurrent import futures
from multiprocessing import Manager
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from server import xr_service_pb2, xr_service_pb2_grpc
from server.video_decoders import VideoDecoder

# Import inference SLAM modules using proper package imports
from slam.services import (
    inference_consumer,
    mapping_consumer,
    inference_pipeline_service as inference_consumer_mapping,
    vis_server
)

# Import configuration system
from config.settings import Config, get_config, set_config


from server.components import SLAMGRPCServer
from server.signal_handlers import install_signal_handlers, get_shutdown_handler

# Configuration constants (will be moved to config)
ASYNC_IO = True
DATA_DUMP = False

    
# import segmentation_pb2_grpc, segmentation_pb2
current_file_location = os.path.dirname(os.path.abspath(__file__))


def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()
def sharpness(numpy_image):
    # Convert numpy image to cv2 image
    image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm


class XRService(xr_service_pb2_grpc.XrServiceServicer):
    """Refactored XR service using component architecture.
    
    This class now delegates to focused components for better maintainability.
    """
    
    def __init__(self, inferenceQueue, target_fps=1, config=None):
        """Initialize XR service with component-based architecture.
        
        Args:
            inferenceQueue: Queue for inference requests
            target_fps: Target processing frame rate
            config: Configuration object (optional, uses get_config() if None)
        """
        self.config = config if config else get_config()
        
        # Initialize the new component-based server
        self.slam_server = SLAMGRPCServer(inferenceQueue, config=self.config)
        
        # Legacy compatibility - expose some attributes for existing code
        self.inferenceQueue = inferenceQueue
        self.target_fps = target_fps
        self.video_processor = self.slam_server.video_processor
        self.inference_service = self.slam_server.inference_service
        
        # def delete_images(folder_path_name):
        #     for dir in ["image", "depth", "pose", "depth/full_depth"]:
        #         folder_path = os.path.join(folder_path_name, dir)
        #         if os.path.exists(folder_path):
        #             for file in os.listdir(folder_path):
        #                 file_path = os.path.join(folder_path, file)
        #                 if os.path.isfile(file_path):
        #                     os.remove(file_path)
        
        # Get the path of the folder containing the files
        # folder_path = os.path.join(current_file_location, "../image_collection/")
        # delete_images(folder_path)
                
        
    # async def makeGRPCRequest(self, image_path, depthFloatArray, poseString, frame_num_image ):
    #     image = Image.open(image_path)
    #     with io.BytesIO() as output:
    #         image.save(output, format="JPEG")
    #         image_bytes = output.getvalue()
        
    #         print("Sending Frame Number = ", frame_num_image)
    #         imageFrame = segmentation_pb2.VideoFrame(data=image_bytes, clientFrameNumber=frame_num_image, width=self.imageWidth, height=self.imageHeight)
    #         depthFrame = segmentation_pb2.DepthFrame(depthArr=depthFloatArray, width=self.depthWidth, height=self.depthheight)
    #         request = segmentation_pb2.SegStream(frames=imageFrame, depth=depthFrame, poseArr=poseString)
    #         response = self.stub.ProcessVideoStream(request)
    #         self.tick_num += 1
            # print("Received response: ", response.message)
    
    
    # def makeGRPCRequestSync(self, image_path, depthFloatArray, poseString, frame_num_image ):
    #     image = Image.open(image_path)
    #     with io.BytesIO() as output:
    #         image.save(output, format="JPEG")
    #         image_bytes = output.getvalue()
        
    #         print("Sending Frame Number = ", frame_num_image)
    #         imageFrame = segmentation_pb2.VideoFrame(data=image_bytes, clientFrameNumber=frame_num_image, width=self.imageWidth, height=self.imageHeight)
    #         depthFrame = segmentation_pb2.DepthFrame(depthArr=depthFloatArray, width=self.depthWidth, height=self.depthheight)
    #         request = segmentation_pb2.SegStream(frames=imageFrame, depth=depthFrame, poseArr=poseString)
    #         response = self.stub.ProcessVideoStream(request)
    #         self.tick_num += 1
    #         print("****************Received response: ", response.message)

    # REMOVED: process_request - now handled by SLAMGRPCServer component
    def UploadSyncMessage_dataset(self, request_iterator, context):
        """Handle dataset upload requests - delegates to component architecture."""
        return self.slam_server.UploadSyncMessage_dataset(request_iterator, context)

    def UploadSyncMessage(self, request_iterator, context):
        """Handle regular upload requests - delegates to component architecture."""
        return self.slam_server.UploadSyncMessage(request_iterator, context)



def serve(inferenceQueue, target_fps, config=None):
    from config.settings import get_config
    if config is None:
        config = get_config()
    
    port = str(config.server.port)
    grpc_options = [
        ('grpc.max_send_message_length', config.grpc.max_send_message_length),
        ('grpc.max_receive_message_length', config.grpc.max_receive_message_length),
        ('grpc.keepalive_time_ms', config.grpc.keepalive_time_ms),
        ('grpc.keepalive_timeout_ms', config.grpc.keepalive_timeout_ms),
        ('grpc.keepalive_permit_without_calls', config.grpc.keepalive_permit_without_calls),
        ('grpc.max_connection_idle_ms', config.grpc.max_connection_idle_ms),
    ]
    
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=config.server.max_workers), 
        options=grpc_options
    )
    xr_service_pb2_grpc.add_XrServiceServicer_to_server(XRService(inferenceQueue, target_fps, config), server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()


def get_parser():
    """Streamlined argument parser - model configs moved to YAML files."""
    parser = argparse.ArgumentParser(description='XR Scene Builder Server')
    
    # Essential runtime parameters (cannot be in config files)
    parser.add_argument('--dataset_type', type=str, choices=['ipad', 'replica', 'scannet'], required=True, 
                       help='Dataset type - determines operational mode')
    parser.add_argument('--config', type=str, default=None, 
                       help='Path to YAML configuration file (e.g., config/production.yaml, config/development.yaml). If not specified, uses config/defaults.yaml')
    
    # Runtime behavior toggles
    parser.add_argument('--clientUpdateMode', action="store_true", 
                       help='Stream objects to client every few frames')
    parser.add_argument('--pipelined_mapping', action="store_true", 
                       help='Separate mapping and inference threads for pipelined processing')
    
    # Runtime connection parameters  
    parser.add_argument('--clientIP', type=str, default="10.193.118.184", 
                       help='Client IP address for streaming mode')
    parser.add_argument('--obejct_update_frequency', type=int, default=1, 
                       help='Frame interval for client object updates')
    
    # Dataset testing parameters
    parser.add_argument('--sceneName', type=str, default=None, 
                       help='Scene name for local dataset testing (e.g., room0, office1). Requires --localDataset.')
    parser.add_argument('--dataset_depth_scaling', type=float, default=1.0, 
                       help='Depth image scaling factor')
    parser.add_argument('--dataset_stride', type=int, default=5, 
                       help='Frame stride for dataset processing')
    parser.add_argument('--save_map', action='store_true', 
                       help='Save semantic maps when scene processing completes')
    parser.add_argument('--localDataset', action='store_true',
                       help='Process local dataset files (replica/scannet only). iPad uses real-time streaming mode.')
    
    # Config overrides - allow command line to override YAML values
    parser.add_argument('--test_depth_downsampling', type=int, default=None,
                       help='Override test_depth_downsampling from config file')
    
    return parser


def apply_config_overrides(config, args):
    """Apply command line argument overrides to config object. Currently only for test_depth_downsample to make it easy for eval (command line sweeps)"""
    if args.test_depth_downsampling is not None:
        config.model.mapping.test_depth_downsampling = args.test_depth_downsampling
        print(f"🔧 Command line override: test_depth_downsampling = {args.test_depth_downsampling}")
    
    return config


def useDataset(inferenceQueue, args, dataset_Name = None):
    """Modular dataset processing function that supports multiple dataset types."""
    import os
    import signal
    import glob
    from slam.datasets.factory import DatasetProcessor
    
    # Get config once for all datasets
    from config.settings import get_config
    config = get_config()
    
    # Get dataset-specific information
    dataset_type = args.dataset_type.lower()
    
    # Determine which scenes to process
    if dataset_Name is not None:
        datasets_to_process = [dataset_Name]
    else:
        # Get default scenes for the dataset type
        datasets_to_process = DatasetProcessor.get_default_scenes(dataset_type)
        if not datasets_to_process:
            print(f"⚠️ No default scenes defined for {dataset_type}. Please specify --sceneName")
            return
    
    print(f"🎯 Processing {dataset_type} dataset with scenes: {datasets_to_process}")
    
    for datasetName in datasets_to_process:
        try:
            # Get dataset-specific paths
            paths = DatasetProcessor.get_dataset_paths(dataset_type, datasetName)
            print(f"📁 Processing scene: {datasetName}")
            print(f"   Dataset path: {paths['dataset_path']}")
            
            # Wait before processing (for system stability)
            time.sleep(15)
            
            # Process dataset based on type
            if dataset_type == 'replica':
                _process_replica_dataset(inferenceQueue, args, config, datasetName, paths)
            elif dataset_type == 'scannet':
                _process_scannet_dataset(inferenceQueue, args, config, datasetName, paths)
            else:
                # This should never be reached due to validation above, but just in case
                raise ValueError(f"Unsupported dataset type for local processing: {dataset_type}")
                
        except Exception as e:
            print(f"❌ Error processing scene {datasetName}: {e}")
            continue
    
    # After all datasets are processed, send shutdown signal to workers
    print("🏁 All datasets processed. Sending shutdown signal...")
    shutdown_signal = {'type': 'shutdown'}
    inferenceQueue.put(shutdown_signal)


def _process_replica_dataset(inferenceQueue, args, config, datasetName, paths):
    """Process Replica dataset."""
    imageDataPath = paths['image_path']
    pose_path = paths['pose_path']
    
    # Calculate how many frames will be processed for this dataset
    frame_range = range(0, 2000, args.dataset_stride)
    num_frames = len(list(frame_range))

    # Process frames
    for frame_number in tqdm(frame_range, desc=f"Processing {datasetName}"):
        # Load image, depth, and pose data from the dataset
        image_path = os.path.join(imageDataPath, f"frame{frame_number:06d}.jpg")
        depth_path = os.path.join(imageDataPath, f"depth{frame_number:06d}.png")
        
        if not os.path.exists(image_path) or not os.path.exists(depth_path):
            continue

        imagePIL = Image.open(image_path)
        depthPng = Image.open(depth_path)
        depthArray = np.array(depthPng)

        with open(pose_path, 'r') as f:
            pose_lines = f.readlines()
        poseString = pose_lines[frame_number].strip()
        pose = [frame_number] + [float(x) for x in poseString.split()]

        clientFrameNumber = frame_number
        clientTimeStamps = time.perf_counter_ns()

        inferenceQueue.put((imagePIL, np.array(imagePIL), depthArray, pose, clientFrameNumber, clientTimeStamps, clientTimeStamps, {}))
        time.sleep(1 / config.server.target_fps)
    
    _send_completion_signal(inferenceQueue, datasetName, num_frames)


def _process_scannet_dataset(inferenceQueue, args, config, datasetName, paths):
    """Process ScanNet dataset."""
    import glob
    from natsort import natsorted
    
    # Get all image and depth files
    image_files = natsorted(glob.glob(os.path.join(paths['image_path'], "*.jpg")))
    depth_files = natsorted(glob.glob(os.path.join(paths['depth_path'], "*.png")))
    pose_files = natsorted(glob.glob(os.path.join(paths['pose_path'], "*.txt")))


    if not image_files or not depth_files or not pose_files:
        print(f"⚠️ Missing files in {datasetName}. Skipping.")
        return
    
    # Apply stride
    image_files = image_files[::args.dataset_stride]
    depth_files = depth_files[::args.dataset_stride]
    pose_files = pose_files[::args.dataset_stride]
    
    num_frames = len(image_files)
    
    for i, (image_path, depth_path, pose_path) in enumerate(tqdm(zip(image_files, depth_files, pose_files), 
                                                                   total=num_frames, desc=f"Processing {datasetName}")):
        if not os.path.exists(image_path) or not os.path.exists(depth_path) or not os.path.exists(pose_path):
            continue

        imagePIL = Image.open(image_path)
        depthPng = Image.open(depth_path)
        depthArray = np.array(depthPng)

        # Load pose from file
        pose_matrix = np.loadtxt(pose_path)
        pose = [i] + pose_matrix.flatten().tolist()

        clientFrameNumber = i
        clientTimeStamps = time.perf_counter_ns()

        inferenceQueue.put((imagePIL, np.array(imagePIL), depthArray, pose, clientFrameNumber, clientTimeStamps, clientTimeStamps, {}))
        time.sleep(1 / config.server.target_fps)
    
    _send_completion_signal(inferenceQueue, datasetName, num_frames)


def _send_completion_signal(inferenceQueue, datasetName, num_frames):
    """Send completion signal for a processed dataset."""
    print(f"⏳ Scene {datasetName}: {num_frames} frames enqueued. Waiting for queue to drain...")
    
    # Wait for inference queue to drain
    while not inferenceQueue.empty():
        time.sleep(0.1)  # Check every 100ms
 
    # Signal scene completion
    print(f"✅ Scene {datasetName} processing completed. Sending completion signal...")
    completion_signal = {
        'type': 'scene_completion',
        'scene_name': datasetName,
        'timestamp': time.perf_counter_ns()
    }
    inferenceQueue.put(completion_signal)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        print(f"📋 Loading configuration from: {args.config}")
        config = Config.from_yaml(args.config)
    else:
        print("📋 Loading default configuration (config/defaults.yaml + environment variables)")
        config = get_config()
    
    # Apply command line overrides
    config = apply_config_overrides(config, args)
    set_config(config)
    
    # print(f"🎯 Configuration Summary:")
    # print(f"   Server: {config.server.host}:{config.server.port} (FPS: {config.server.target_fps})")
    # print(f"   Detection device: {config.model.detection.device}")
    # print(f"   Segmentation device: {config.model.segmentation.device}")
    # print(f"   CLIP device: {config.model.clip.device}")
    
    # Scene name logic - for local datasets, determine the actual scene that will be processed
    if args.sceneName:
        scene_name = args.sceneName
    elif args.localDataset:
        # For local dataset mode, get the first scene that will be processed
        from slam.datasets.factory import DatasetProcessor
        default_scenes = DatasetProcessor.get_default_scenes(args.dataset_type)
        if default_scenes:
            scene_name = default_scenes[0]  # Use the first default scene
            print(f"🎯 No --sceneName provided, using first default scene: {scene_name}")
        else:
            scene_name = args.dataset_type
    else:
        # For streaming mode, use dataset type as fallback
        scene_name = args.dataset_type
    
    print(f"🎯 Scene name for consumers: {scene_name}")
    
    # Extract config file name for 3-level directory structure
    if args.config:
        config_name = Path(args.config).stem  # e.g., "replica_config" from "replica_config.yaml"
    else:
        config_name = "default"
    
    # Set environment variables so all child processes use the same names
    os.environ['SLAM_CONFIG_NAME'] = config_name
    os.environ['SLAM_SCENE_NAME'] = scene_name
    print(f"🎯 Performance logging: logs_performance/{config_name}/{scene_name}/")
    
    shutdown_handler = install_signal_handlers()
    
    # serve()
    multiprocessing.set_start_method('spawn')
    if args.clientUpdateMode:
        assert args.clientIP is not None, "Please provide the client IP address"
        assert args.dataset_type != "ipad", "Client update mode is only supported for replica or ScanNet dataset/Jetson"
    print(f"Launching server for dataset type: {args.dataset_type}")
    
    
    inferenceQueue = multiprocessing.Queue(maxsize=config.server.inference_queue_size)
    mappingQueue = multiprocessing.Queue(maxsize=config.server.mapping_queue_size)    
    visualizationQueue = multiprocessing.Queue(maxsize=config.server.visualization_queue_size)
    
    print(f"Queue sizes: Inference={config.server.inference_queue_size}, Mapping={config.server.mapping_queue_size}, Visualization={config.server.visualization_queue_size}")

    # Create producer thread based on dataset type and mode
    if args.localDataset:
        # Local dataset processing (Replica, ScanNet)
        if args.dataset_type == "ipad":
            raise ValueError("iPad is a real-time streaming dataset, not a local dataset. Remove --localDataset flag for iPad.")
        
        if args.dataset_type not in ["replica", "scannet"]:
            raise ValueError(f"Local dataset processing only supports 'replica' and 'scannet', got '{args.dataset_type}'")
        
        if args.sceneName:
            print(f"🗂️ Processing local {args.dataset_type} dataset, scene: {args.sceneName}")
        else:
            print(f"🗂️ Processing all scenes from local {args.dataset_type} dataset")
            
        producer_thread = multiprocessing.Process(target=useDataset, args=(inferenceQueue, args, args.sceneName))
    else:
        # Real-time streaming mode (iPad, or gRPC server for Replica/ScanNet)
        if args.dataset_type == "ipad":
            print("📱 Starting iPad real-time streaming mode")
        else:
            print(f"🌐 Starting gRPC server for {args.dataset_type} dataset")
            
        producer_thread = multiprocessing.Process(target=serve, args=(inferenceQueue, config.server.target_fps, config))
    if not args.pipelined_mapping:
        inference_consumer_thread =  multiprocessing.Process(target=inference_consumer_mapping, args=(inferenceQueue, 
                                                                                    visualizationQueue,
                                                                                    config.model.detection.enabled,  # Use config instead of args
                                                                                    config,  # Pass entire config instead of individual args
                                                                                    args.dataset_type,
                                                                                    False,
                                                                                    args.save_map,
                                                                                    scene_name))
        
    else:
        
        inference_consumer_thread =  multiprocessing.Process(target=inference_consumer, args=(inferenceQueue, 
                                                                                            mappingQueue,
                                                                                            config.model.detection.enabled,  # Use config instead of args
                                                                                            config,  # Pass entire config instead of individual args
                                                                                            ))
    
        mapping_consumer_thread =  multiprocessing.Process(target=mapping_consumer, args=(mappingQueue, 
                                                                                        visualizationQueue,
                                                                                        config.model.detection.enabled,  # Use config instead of args
                                                                                        args.dataset_type,
                                                                                        args.save_map,
                                                                                        config,  # Pass full config
                                                                                        scene_name))
    vis_server_thread =  multiprocessing.Process(target=vis_server, 
                                                args=(visualizationQueue,
                                                config,  # Pass config instead of individual vis args
                                                args.dataset_type, 
                                                args.clientUpdateMode, 
                                                args.obejct_update_frequency, 
                                                args.clientIP,
                                                args,))

        
    # Register all processes with shutdown handler
    processes = [producer_thread, inference_consumer_thread, vis_server_thread]
    if args.pipelined_mapping:
        processes.append(mapping_consumer_thread)
    shutdown_handler.register_processes(processes)
    
    # Start the threads
    producer_thread.start()
    inference_consumer_thread.start()
    if args.pipelined_mapping:
        mapping_consumer_thread.start()
    vis_server_thread.start()
    
    try:
        # Wait for producer (dataset processing) to complete
        producer_thread.join()
        print("✅ Dataset processing completed")
        
        # Workers will naturally exit when they process shutdown signals
        print("⏳ Waiting for workers to complete gracefully...")
        inference_consumer_thread.join()
        if args.pipelined_mapping:
            mapping_consumer_thread.join()
        vis_server_thread.join()
        
        print("✅ All workers completed successfully")
            
    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt detected in main thread")
        shutdown_handler.shutdown()
    except Exception as e:
        print(f"\n❌ Unexpected error in main thread: {e}")
        shutdown_handler.shutdown()
    finally:
        print("📊 Final cleanup completed")