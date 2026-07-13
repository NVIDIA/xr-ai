# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visualization gRPC server for semantic SLAM operations."""

import asyncio
import copy
import csv
import cv2
import gzip
import grpc
import io
import logging
import os
import pickle
import struct
import sys
import threading
import time
from concurrent import futures
from multiprocessing import Queue
from typing import Tuple

import distinctipy
import matplotlib
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import resize

# Import SLAM utilities using proper package imports
from slam.core.slam_classes import MapObjectList
from slam.utils.vis import LineMesh
from slam.utils.performance_manager import get_performance_manager
from slam.core.utils import filter_objects, merge_objects

# Import vis_proto using proper relative imports
from slam.protocols.vis_proto import vis_pb2, vis_pb2_grpc

# Configuration constants
AUDIO_INPUT_USE_OPENAI = False
if AUDIO_INPUT_USE_OPENAI:
    openai_key = os.getenv("OPENAI_API_KEY")
ASYNC_IO = False


def saveToWavFile(data, file_path):
    with open(file_path, "wb") as f:
        f.write(data)
    return file_path
class VisualizationServicer(vis_pb2_grpc.VisualizerServerServicer):
    def __init__(self, visualizationQueue, device="cuda:0", dataset_type="ipad", useTRT=False, clipModel="ViT-H-14", precision="fp16", batch_size=1, clientUpdateMode=False, client_update_frameInt=1, clientIP=None, args=None, config=None, grpc_server=None):
        self.clientUpdateMode = clientUpdateMode   
        self.firstUpdate = True
        self.clientIP = clientIP 
        self.client_update_frameInt = client_update_frameInt
        self.batch_size = batch_size
        self.precision = precision
        self.dataset_type = dataset_type
        self.device = device
        self.trt_clip = useTRT
        self.config = config  # Store config for later use
        self.clipModelType = clipModel
        self.grpc_server = grpc_server  # Store server reference for shutdown

        
        # self.log_directory = "/home/rahul/Documents/connor_streaming_code/new_clean_repo/main/semantic_slam_server/timing_info/streaming/clientUpdateMode/"   
        # self.queryLog_directory = "/home/rahul/Documents/connor_streaming_code/new_clean_repo/main/semantic_slam_server/timing_info/server_query_logs/clientUpdateMode/"
        
        # Initialize centralized performance manager with proper scene name
        import os
        scene_name = os.environ.get('SLAM_SCENE_NAME', 'unknown')
        self.performance_manager = get_performance_manager(scene_name=scene_name)
        
        # Get console log interval for stats printing
        self.console_log_interval = self.config.logging.console_log_interval if self.config else 20
        
        print(f"🎯 VisualizationServicer initialized with centralized performance logging")

        if ASYNC_IO:
            # For sending the results to mapping server
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.channel = grpc.aio.insecure_channel(f'{self.clientIP}:50054')
        else:
            self.channel = grpc.insecure_channel(f'{self.clientIP}:50054')
        self.stub = vis_pb2_grpc.VisualizerServerStub(self.channel)
            
        # if "ViT-H-14" == self.clipModelType:
        #     self.training_data = "laion2b_s32b_b79k"
        # elif "ViT-L-14" == self.clipModelType:
        #     self.training_data = "laion2b_s32b_b82k"
        # elif "MobileCLIP-B" == self.clipModelType:
        #     self.training_data = "datacompdr_lt"
        self.training_data = config.model.visualization.pretrained
        print("Initializing CLIP model...")
        if self.trt_clip:
            import utils.model_trt_utils as trt_utils
            self.device_number = int(device.split(":")[1])
            self.clip_model = trt_utils.CLIP_TRT(precision=self.precision, batch_size=self.batch_size, clip_model=self.clipModelType, device=self.device_number) 
        else:
            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(self.clipModelType, self.training_data)
            self.clip_model = clip_model.to(self.device)
            self.clip_tokenizer = open_clip.get_tokenizer(self.clipModelType)
        print("Done initializing CLIP model.")
        if AUDIO_INPUT_USE_OPENAI:
            self.ASR_client = OpenAI(api_key=openai_key)
        self.asr_mode = "whisper-1"
        self.objects = None
        self.bg_objects = None
        self.class_colors = None
        self.audio_data = []
        self.similarity_threshold = 0.86
        self.pcds = None
        self.bboxes = None
        # self.color = set()
        self.all_queries = ["monitor", "keyboard" ,"mouse", "chair", "laptop"]
        self.query_index = 0
        self.frame_update_index = 0
        self.freshObjects = []          # contains the indices of the fresh objects that have been updated since the last update was sent to client
        self.removed_objects = []       # contains the indices of the objects that have been removed since the last update was sent to client
        self.history_map = None         # keys are the indices of the objects when they were first added to the map (uniqueID/'history_idx'), values are the indices of the objects in the current ObjList
        self.cmap = matplotlib.colormaps.get_cmap("turbo")
        self.file_path = os.path.join( os.path.dirname(os.path.abspath(__file__)) ,"output_audio.wav")
        self.visualizationQueue = visualizationQueue
        self.objects_clip_fts = None
        self.objects_text_fts = None
        self.pcd_points = []
        self.bboxes = []
        

    def SpeechRecognition(self):
        
        print("Transcribing audio...")
        with open(self.file_path, "rb") as audio_file:
            transcript = self.ASR_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcript.text
    
    def update_wav_file_sizes(self, outfile, data_size):
        # Calculate sizes
        Subchunk2Size = data_size
        fileSize = 36 + Subchunk2Size

        # Update fileSize in RIFF header
        outfile.seek(4)
        outfile.write(struct.pack('<I', fileSize))

        # Update Subchunk2Size in data subchunk
        outfile.seek(40)
        outfile.write(struct.pack('<I', Subchunk2Size))
    
    
    def WriteWavHeader(self, outfile):
            # WAV header constants
        NumChannels = 1  # Mono
        CompressionCode = 1  # PCM encoding
        SampleRate = 12000  # Sample rate (Hz)
        BitsPerSample = 16  # Bits per sample
        BytesPerSample = BitsPerSample // 8
        Subchunk1Size = 16  # PCM format subchunk size
        ByteRate = SampleRate * NumChannels * BytesPerSample
        BlockAlign = NumChannels * BytesPerSample
        Subchunk2Size = 0  # Placeholder for audio data size (to be updated later)
        
        # Calculate file size (36 bytes for header without data)
        fileSize = 36

        # Write RIFF chunk descriptor
        outfile.write(b'RIFF')
        outfile.write(struct.pack('<I', fileSize))   # Placeholder for file size (to be updated later)
        outfile.write(b'WAVE')

        # Write fmt subchunk
        outfile.write(b'fmt ')  # Write the ASCII characters 'fmt '
        outfile.write(struct.pack('<I', Subchunk1Size))  # Write the subchunk size (4 bytes)
        outfile.write(struct.pack('<H', CompressionCode))  # Write the compression code (2 bytes)
        outfile.write(struct.pack('<H', NumChannels))  # Write the number of channels (2 bytes)
        outfile.write(struct.pack('<I', SampleRate))  # Write the sample rate (4 bytes)
        outfile.write(struct.pack('<I', ByteRate))  # Write the byte rate (4 bytes)
        outfile.write(struct.pack('<H', BlockAlign))  # Write the block align (2 bytes)
        outfile.write(struct.pack('<H', BitsPerSample))  # Write the bits per sample (2 bytes)

         # Write data subchunk (placeholder)
        outfile.write(b'data')  # Write the ASCII characters 'data'
        outfile.write(struct.pack('<I', Subchunk2Size))  # Write the subchunk size (4 bytes)
    
    def load_result(self, results):
        # with gzip.open(result_path, "rb") as f:
        #     results = pickle.load(f)
    
        if isinstance(results, dict):
            objects = MapObjectList()
            objects.load_serializable(results["objects"])
            
            if results['bg_objects'] is None:
                bg_objects = None
            else:
                bg_objects = MapObjectList()
                bg_objects.load_serializable(results["bg_objects"])

            class_colors = results['class_colors']
        elif isinstance(results, list):
            objects = MapObjectList()
            objects.load_serializable(results)

            bg_objects = None
            if class_colors is not None:
                class_colors = distinctipy.get_colors(len(objects), pastel_factor=0.5)
                class_colors = {str(i): c for i, c in enumerate(class_colors)}
        else:
            raise ValueError("Unknown results type: ", type(results))
            
        return objects, bg_objects, class_colors

    
    def getClipComparison(self, text_query):
        pcd_to_return = []
        color_to_return = []
        text_queries = [text_query]
        start = time.perf_counter_ns()
        if self.trt_clip:
            # print("[[VISUALIZATION SERVER]]****************Using TRT for CLIP]]****************]]****************]]****************]]****************]]****************]]****************")
            text_query_ft = self.clip_model.encode_text(text_queries)
            text_query_ft = torch.tensor(text_query_ft.squeeze(), device=self.device)
        else:
            text_queries_tokenized = self.clip_tokenizer(text_queries).to(self.device)
            text_query_ft = self.clip_model.encode_text(text_queries_tokenized)
            text_query_ft = text_query_ft / text_query_ft.norm(dim=-1, keepdim=True)
            text_query_ft = text_query_ft.squeeze()
        clip_time = time.perf_counter_ns() 
        # print("[VISUALIZATION SERVER]++++++++++++++  Time taken to get text features: ", (clip_time-start)/1e6, "ms", "using TRT: " , self.trt_clip)
        text_query_ft.unsqueeze(0)
        
        # similarities = objects.compute_similarities(text_query_ft)
        # objects_clip_fts = objects.get_stacked_values_torch("clip_ft")
        
        similarities = F.cosine_similarity(
            text_query_ft.unsqueeze(0), self.objects_clip_fts, dim=-1
        )
        max_value = similarities.max()
        if max_value < 0.15:
            # No good matches. Return empty list. 
            # Can't rely solely on normalized similarities, since it always returns 1 for at least one object.
            # This can be relaxed - start from max - check if diff(max-2nd max) is > 0.3, if yes, then return the max object
            # if diff is less than 0.1, then there are similar objects, but we are not sure (since the val is < 0.3). so check if there is a gap of 0.3 between first 1/4th objects.
            # if yes, then return the first k objects out of 1/4th objects. If not, then return empty list.
            print("No good matches found --- Max similarity:---------- ", max_value , " for ", text_query)
            return pcd_to_return, color_to_return
        min_value = similarities.min()
        # print("[VISUALIZATION SERVER]*********** ***********Text:",text_query ," Max similarity: ", max_value, " Min similarity: ", min_value)
        # normalized_similarities - ranges between 0 and 1. 0 is the least similar, 1 is the most similar. Threshold at 0.86 or 0.87
        normalized_similarities = (similarities - min_value) / (max_value - min_value)
        color_sim = (similarities - min_value) / (0.35 - min_value)
        similarity_colors = self.cmap(color_sim.detach().cpu().numpy())[..., :3]
        # Find the indices of pcds with similarity greater than the threshold
        indices = torch.where(normalized_similarities > self.similarity_threshold)[0].cpu().numpy()
        
        for i in indices:
            if self.pcd_points[i].shape[0] > 200:
                self.pcd_points[i] = self.pcd_points[i][np.random.choice(self.pcd_points[i].shape[0], 200, replace=False)]
            pcd_to_return.append(self.pcd_points[i])
            color_to_return.append(similarity_colors[i][0])
            color_to_return.append(similarity_colors[i][1])
            color_to_return.append(similarity_colors[i][2])
        return pcd_to_return, color_to_return
            
        # print("Number of objects with similarity greater than threshold: ", len(indices))
        # print(indices)
        # for i in range(len(self.objects)):
        #     if normalized_similarities[i] > self.similarity_threshold:
        #         print(i)
    
    def createGRPCResponse(self, pcd_list, color_list, serverQueryProcessingTime):
        response = vis_pb2.allPointClouds()
        response.colors.extend(color_list)
        response.numPointClouds = len(pcd_list)
        response.serverQueryProcessing = serverQueryProcessingTime
        for pcd in pcd_list:
            # center = pcd.get_center()
            center = [0,0,0]
            # points = np.asarray(pcd.points, dtype=np.float64)
            points = pcd
            num_points = points.shape[0]
            points_flat_list = points.flatten().tolist()
            pc = vis_pb2.PointCloud(points=points_flat_list, num_points=num_points, centroid=center)
            
            response.pointClouds.append(pc)
        return response
    
    def clientTextQuery(self, request_iterator, context):
        # print("\n[VISUALIZATION SERVER]\t\tReceived clientTextQuery request")
        if AUDIO_INPUT_USE_OPENAI:
            outfile = open(self.file_path, "wb")
            self.WriteWavHeader(outfile)
            
            for i in request_iterator:
                self.audio_data.append(i.chunk_data)
            # Concatenate the chunks into a single byte stream
            combined_audio_data = b''.join(self.audio_data)
            
            outfile.write(combined_audio_data)
            
            self.update_wav_file_sizes(outfile, len(combined_audio_data))
            outfile.close()
    
            transcript = self.SpeechRecognition()
            self.text_query = transcript
            print("Transcript: ", transcript)
        else:
            if self.dataset_type != "ipad":
                for i in request_iterator:
                    self.text_query = i.textQuery
            else:
                self.text_query = self.all_queries[self.query_index % 5]

        if self.pcd_points is not None:
            self.query_index += 1
            print("\n[VISUALIZATION SERVER]\t\t looking for ----------------- ", self.text_query)
            # print("\n\nText query: ", self.text_query, "\n\n")
            start = time.perf_counter_ns()
            pcd_list, color_list = self.getClipComparison(self.text_query)
            end = time.perf_counter_ns()
            
            # print("[[VISUALIZATION SERVER]]++++++++++++++  Time taken to get point clouds: ", (end-start)/1e6, "ms, Number of point clouds: ", len(pcd_list), "  shapes of pcds: ", [pcd.shape for pcd in pcd_list])    
            response = self.createGRPCResponse(pcd_list, color_list, (end-start)/1e6)
            end_grpc = time.perf_counter_ns()
            query_time_dict = {'query_num': self.query_index , 'time_to_get_point_clouds' : (end - start) / 1e6, 'createGRPCResponse_time' : (end_grpc - end) / 1e6, 'num_objects' :len(self.pcd_points)} 
            self.log_query_info(query_time_dict)
            
        else:
            response = vis_pb2.allPointClouds()
            response.numPointClouds = 0
        self.audio_data = []
        return response
    
    # def updateMap(self, request, context):
    #     print("\n[VISUALIZARION SERVER] Received updateMap request")
    #     while self.visualizationQueue.empty():
    #         continue
    #     start = time.perf_counter_ns()
    #     results = self.visualizationQueue.get()
        
    #     self.objects, self.bg_objects, self.class_color = self.load_result(results)
        
    #     self.cfg = results['cfg']
    #     self.class_names = results['class_names']
        
        
    #     # resultPath = "/home/rahul/Documents/scene_graphs/concept-graphs/external/nice-slam/scripts/Datasets/Replica/dataset_2/pcd_saves/full_pcd_none_overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub_post.pkl.gz"
        
    #     # self.pcds = copy.deepcopy(self.objects.get_values("pcd"))
    #     self.pcds = self.objects.get_values("pcd")
        
    #     # self.bboxes = copy.deepcopy(self.objects.get_values("bbox"))
    #     # print("Loaded objects from file --", resultPath)
    #     end = time.perf_counter_ns()
    #     print(f"*********** Updated the map ********************** Time taken = {(end-start)/1e6} \n READY TO RECEIVE QUERIES")
    #     return vis_pb2.Status(message=True)
    
    def _compute_update_message_size_mbits(self, obj_indices, history_map=None):
        """Build objectUpdate for given object indices and return serialized size in Mbits.
        
        Args:
            obj_indices: List of indices into pcd_points/bboxes/objects_clip_fts.
            history_map: Optional dict mapping hist_map_idx -> obj_idx. If provided, used to
                        set Objects.index from obj_idx (inverse lookup). For size, index value is negligible.
        """
        if not obj_indices or self.pcd_points is None or len(self.pcd_points) == 0:
            return 0.0
        update_request = vis_pb2.objectUpdate()
        update_request.num_objects = len(obj_indices)
        clip_fts = self.objects_clip_fts
        rev_map = {v: k for k, v in (history_map or {}).items() if v is not None}
        for i, idx in enumerate(obj_indices):
            obj = vis_pb2.Objects()
            selected_pcd = self.pcd_points[idx]
            if selected_pcd.shape[0] > 200:
                selected_pcd = selected_pcd[np.random.choice(selected_pcd.shape[0], 200, replace=False)]
            obj.pointCloud.extend(selected_pcd.flatten().tolist())
            obj.BBox.extend(self.bboxes[idx].flatten().tolist())
            ft = clip_fts[idx]
            obj.clip_embeddings.extend(ft.cpu().tolist() if hasattr(ft, 'cpu') else ft.tolist())
            obj.num_points = self.pcd_points[idx].shape[0]
            obj.centroid.extend([0, 0, 0])
            obj.index = rev_map.get(idx, i)
            update_request.objects.append(obj)
        return (len(update_request.SerializeToString()) * 8) / (1024 * 1024)

    def _compute_fresh_and_full_scene_update_sizes(self, fresh_obj_indices, history_map=None):
        """Compute fresh-update size and full-scene size in Mbits. Returns (fresh_mbits, full_mbits).
        Uses distinct current object indices to avoid inflating fresh size when multiple history IDs map to same object.
        """
        n = len(self.pcd_points) if self.pcd_points is not None else 0
        if n == 0:
            return 0.0, 0.0
        hm = history_map or self.history_map
        distinct_indices = sorted(set(idx for idx in fresh_obj_indices if idx is not None))
        fresh_mbits = self._compute_update_message_size_mbits(distinct_indices, hm) if distinct_indices else 0.0
        full_mbits = self._compute_update_message_size_mbits(list(range(n)), hm)
        return fresh_mbits, full_mbits

    def generateClientUpdate(self, frameNumber, starting_timestamp, clientTimeStamps, time_dict):  
        start_building_message = time.perf_counter_ns()
        self.freshObjects = set(self.freshObjects)
        # Use distinct current object indices to avoid sending same object multiple times (can happen when
        # multiple history IDs map to same object after merges). Pick min history_idx per object for client tracking.
        fresh_obj_indices = [self.history_map[h] for h in self.freshObjects if self.history_map and self.history_map.get(h) is not None]
        distinct_indices = sorted(set(fresh_obj_indices))
        idx_to_hist = {}
        for h in self.freshObjects:
            if self.history_map and self.history_map.get(h) is not None:
                idx = self.history_map[h]
                if idx not in idx_to_hist:
                    idx_to_hist[idx] = h
        update_request = vis_pb2.objectUpdate()
        update_request.num_objects = len(distinct_indices)
        update_request.remove_indices.extend(self.removed_objects)
        
        update_request.clientTimeStamp = clientTimeStamps
        
        
        for idx in distinct_indices:
            obj = vis_pb2.Objects()
            
            selected_pcd = self.pcd_points[idx]
            if selected_pcd.shape[0] > 200:
                selected_pcd = selected_pcd[np.random.choice(selected_pcd.shape[0], 200, replace=False)]
            
            obj.pointCloud.extend(self.pcd_points[idx].flatten().tolist())
            obj.BBox.extend(self.bboxes[idx].flatten().tolist())
            obj.clip_embeddings.extend(self.objects_clip_fts[idx].tolist())
            obj.num_points = self.pcd_points[idx].shape[0]
            obj.centroid.extend([0,0,0]) 
            obj.index = idx_to_hist[idx]
            update_request.objects.append(obj)
        message_end = time.perf_counter_ns()
        serialized_request = update_request.SerializeToString()
        if self.client_update_frameInt == 1:
            update_request.frameNumber.extend([frameNumber])
            update_request.serverProcessLatency = (time.perf_counter_ns()-starting_timestamp)/1e6
        else:
            update_request.frameNumber.extend([])
            update_request.serverProcessLatency = 0
        
        time_dict['client_update_message_building_time'] = (message_end - start_building_message)/1e6
        fresh_mbits = (len(serialized_request) * 8) / (1024 * 1024)
        time_dict['client_update_message_size (Mbits)'] = fresh_mbits
        time_dict['total objecs'] = len(self.pcd_points)
        
        # Log to separate update_sizes.csv (num_fresh = distinct current objects in update)
        full_mbits = self._compute_update_message_size_mbits(list(range(len(self.pcd_points))))
        num_fresh_distinct = len(distinct_indices)
        self.performance_manager.log_update_sizes(
            fresh_mbits, full_mbits,
            num_fresh_objects=num_fresh_distinct,
            num_total_objects=len(self.pcd_points)
        )
        
        # Log timing information
        self.log_timing_info(frameNumber, time_dict)
        
        print(f"\n[VISUALIZATION SERVER]*********** Updating Client with {num_fresh_distinct} distinct objects - Message size: {len(serialized_request)/1024:.1f} KB (fresh: {fresh_mbits:.2f} Mbits, full-scene: {full_mbits:.2f} Mbits) Time taken to build message = {((time.perf_counter_ns()-start_building_message)/1e6)} ms\n")
        yield update_request
    
    #Once the client sends a updateMap request, the server will continue to check the queue. This queue will be updated by mapping server.
    def updateMap(self, request, context):
        print("\n[VISUALIZARION SERVER] Received updateMap request")
        while True:
            if self.visualizationQueue.empty():
                continue
            start = time.perf_counter_ns()
            queue_item = self.visualizationQueue.get()
            
            # Check if this is a scene completion signal
            if isinstance(queue_item, dict) and queue_item.get('type') == 'scene_completion':
                scene_name = queue_item['scene_name']
                print(f"🎯 [VISUALIZATION] Received scene completion signal for {scene_name}")
                
                # Map dumping now handled in inference pipeline
                
                # Write scene-specific performance summary and reset for next scene
                self.performance_manager.write_scene_summary_and_reset(scene_name)
                
                # Clear visualization data for next scene
                print(f"🧹 [VISUALIZATION] Clearing visualization data for next scene")
                self.pcd_points = []
                self.bboxes = []
                self.objects_clip_fts = None
                self.objects_text_fts = None
                self.freshObjects = []
                self.removed_objects = []
                self.history_map = None
                self.frame_update_index = 0
                
                continue
                
            # Check if this is a shutdown signal
            if isinstance(queue_item, dict) and queue_item.get('type') == 'shutdown':
                print("🏁 [VISUALIZATION] Received shutdown signal. Finalizing performance data...")
                self.performance_manager.cleanup()  # Safe - won't duplicate due to cleanup flag
                
                # Stop the gRPC server to allow graceful shutdown
                if self.grpc_server:
                    print("🛑 [VISUALIZATION] Stopping gRPC server...")
                    self.grpc_server.stop(grace=2.0)  # 5 second grace period
                
                return vis_pb2.Status(message=True)
                
            results,  frameNumber, starting_timestamp, clientTimeStamps, time_dict = queue_item
            self.pcd_points = results['queue_pcd_points']
            self.bboxes = results['queue_pcd_bbox']
            self.objects_clip_fts = results['objects_clip_fts']
            self.objects_text_fts = results['objects_text_fts']
            
            
            
            if self.clientUpdateMode:
                if self.firstUpdate:
                    self.firstUpdate = False
                self.frame_update_index +=1 
                fresh_objects = results['fresh_objects']
                removed_objects = results['removed_objects']
                history_map = results['history_map']    
                self.freshObjects.extend(fresh_objects)
                self.removed_objects.extend(removed_objects)
                # Remove any fresh object that is also in removed objects
                self.freshObjects = [obj for obj in self.freshObjects if obj not in self.removed_objects]
                self.history_map = history_map
                if self.frame_update_index % self.client_update_frameInt == 0:
                    responseFromClient = self.stub.updateDeviceMap(self.generateClientUpdate(frameNumber, starting_timestamp, clientTimeStamps, time_dict))
                    self.freshObjects = []
                    self.removed_objects = []
                    
                # Also log timing information in client update mode
                self.log_timing_info(frameNumber, time_dict, 
                                   server_timestamp=time.perf_counter_ns(),
                                   client_timestamp=clientTimeStamps)
            else:
                # Non-client mode (e.g. localDataset): measure and log update sizes at update frequency
                # Use sequential counter (frame_update_index) since frameNumber can be strided (0,5,10,...)
                self.frame_update_index += 1
                if self.frame_update_index % self.client_update_frameInt == 0:
                    fresh_objects = results.get('fresh_objects', [])
                    history_map = results.get('history_map')
                    fresh_obj_indices = [history_map[h] for h in fresh_objects if history_map and history_map.get(h) is not None]
                    fresh_mbits, full_mbits = self._compute_fresh_and_full_scene_update_sizes(fresh_obj_indices, history_map)
                    num_fresh_distinct = len(set(fresh_obj_indices))
                    self.performance_manager.log_update_sizes(
                        fresh_mbits, full_mbits,
                        num_fresh_objects=num_fresh_distinct,
                        num_total_objects=len(self.pcd_points)
                    )
                # Log timing information with timestamps
                self.log_timing_info(frameNumber, time_dict, 
                                   server_timestamp=time.perf_counter_ns(),
                                   client_timestamp=clientTimeStamps)
            
            # Print stats every N frames using existing frameNumber (for both modes)
            if frameNumber % self.console_log_interval == 0:
                print(f"\n📊 [VISUALIZATION] Current Frame {frameNumber} Stats:")
                print(f"   Total: {time_dict.get('total_time', 0):.1f}ms")
                print(f"   Caption: {time_dict.get('caption_time', 0):.1f}ms | Detection: {time_dict.get('detection_time', 0):.1f}ms | Segmentation: {time_dict.get('segmentation_time', 0):.1f}ms | CLIP: {time_dict.get('clip_time', 0):.1f}ms | Mapping: {time_dict.get('mapping_time', 0):.1f}ms")

            self.objects_clip_fts = self.objects_clip_fts.to(self.device)

            # self.objects, self.bg_objects, self.class_color = self.load_result(results)
            
            # self.cfg = results['cfg']
            # self.class_names = results['class_names']
            
            
            # resultPath = "/home/rahul/Documents/scene_graphs/concept-graphs/external/nice-slam/scripts/Datasets/Replica/dataset_2/pcd_saves/full_pcd_none_overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub_post.pkl.gz"
            
            # self.pcds = copy.deepcopy(self.objects.get_values("pcd"))
            # self.pcds = self.objects.get_values("pcd")
            
            # self.bboxes = copy.deepcopy(self.objects.get_values("bbox"))
            # print("Loaded objects from file --", resultPath)
            end = time.perf_counter_ns()
            # print(f"[VISUALIZATION SERVER]\t\t*********** Frame {frameNumber} Update Time = {(end-start)/1e6} ms, TOTAL LATENCY = {(end-starting_timestamp)/1e6} ms\n")
        return vis_pb2.Status(message=True)

    def log_timing_info(self, frameNumber, time_dict, server_timestamp=None, client_timestamp=None):
        """Log timing info using centralized performance manager."""
        # Use centralized performance manager for logging
        self.performance_manager.log_frame_timing(
            frame_number=frameNumber,
            timing_dict=time_dict,
            server_timestamp=server_timestamp,
            client_timestamp=client_timestamp
        )
            
    def log_query_info(self, query_dict):
        """Log query info - could be extended to use performance manager."""
        # For now, just print query info since it's less critical
        print(f"🔍 Query Info: {query_dict}")

def vis_server(visualizationQueue, config, dataset_type, clientUpdateMode=False, client_update_frameInt=1, clientIP=None, args=None):
    # Extract visualization config values
    if config is None:
        print("WARNING: No config provided, using default config")
        from config.settings import get_config
        config = get_config()
    else:
        # Set the config globally so utils.py can access it
        from config.settings import set_config
        set_config(config)
    
    device = config.model.visualization.device
    useTRT = config.model.visualization.use_trt
    clipModelvis = config.model.visualization.clip_model
    precision = config.model.visualization.precision
    batch_size = config.model.visualization.batch_size
    
    # Create a gRPC server
    port = 50054
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    
    # Create servicer with server reference for graceful shutdown
    servicer = VisualizationServicer(visualizationQueue, 
                                   device, 
                                   dataset_type=dataset_type, 
                                   useTRT=useTRT, 
                                   clipModel=clipModelvis, 
                                   precision=precision, 
                                   batch_size=batch_size, 
                                   clientUpdateMode=clientUpdateMode, 
                                   client_update_frameInt=client_update_frameInt, 
                                   clientIP=clientIP, 
                                   args=args,
                                   config=config,
                                   grpc_server=server)
    
    vis_pb2_grpc.add_VisualizerServerServicer_to_server(servicer, server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f"*****************************************************[Visualization Server Started on Port {port}]*****************************************************")
    server.wait_for_termination()

# if __name__ == '__main__':
#     # Create a gRPC server
#     vis_server()