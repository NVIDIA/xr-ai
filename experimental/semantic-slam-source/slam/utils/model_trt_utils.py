# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from transformers import CLIPTokenizer
from PIL import Image
from torchvision import transforms
import open_clip
import time
import os

if "CLIP_TRT_WEIGHTS" in os.environ:
    CLIP_TRT_WEIGHTS = os.environ["CLIP_TRT_WEIGHTS"]
else:
    raise ValueError("Please set the environment variable CLIP_TRT_WEIGHTS to the directory containing the TensorRT engine files")

class CLIP_TRT:
    def __init__(self, precision="fp32", batch_size=1, clip_model="ViT-H-14", device=0):
        self.engine_file_path = os.path.join(CLIP_TRT_WEIGHTS, f"{clip_model}_{precision}_B{batch_size}_device{device}.engine")
        print("Engine File Path: ", self.engine_file_path)
        
        
        cuda.init()
        
        self.device_number = device
        self.device = cuda.Device(device)
        # cuda.cudaSetDevice(self.device) 
        self.ctx = self.device.make_context()
        
        self.engine = self.load_engine(self.engine_file_path)
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine, batch_size)
        self.context = self.engine.create_execution_context()
        self.batch_size = batch_size
        self.clip_tokenizer = open_clip.get_tokenizer(clip_model)
        self.cuda_stream = cuda.Stream()
        # self.context = self.engine.create_execution_context()
        # self.clip_model, _ = open_clip.load("ViT-B/32")
        # self.device = "cuda"
        # self.clip_model.to(self.device)
        # self.clip_model.eval()
        # self.clip_model.requires_grad_(False)
        # self.padding = 16
        # self.classes = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        
        
    def __del__(self):
        self.ctx.pop()
            
    
    
    # Load the TensorRT engine
    def load_engine(self, engine_file_path):
        with open(engine_file_path, 'rb') as f, trt.Runtime(trt.Logger()) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
            
    # Allocate buffers for inputs and outputs
    def allocate_buffers(self, engine, batch_size):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            if size < 0:
                size = 0 - size
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                print(binding , "---", size, dtype)
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
                print(binding , "---", size, dtype)
        return inputs, outputs, bindings, stream 
    
    def run_inference(self, image_pil, detections, classes):
        # image = Image.fromarray(image_rgb)
        image = image_pil
        padding = 20  # Adjust the padding amount as needed
        
        image_crops = []
        preprocessed_images = []
        text_tokens = []
        all_image_encodings = []
        all_text_encodings = []
        
        # Prepare data for batch processing
        for idx in range(len(detections.xyxy)):
            x_min, y_min, x_max, y_max = detections.xyxy[idx]
            image_width, image_height = image.size
            left_padding = min(padding, x_min)
            top_padding = min(padding, y_min)
            right_padding = min(padding, image_width - x_max)
            bottom_padding = min(padding, image_height - y_max)

            x_min -= left_padding
            y_min -= top_padding
            x_max += right_padding
            y_max += bottom_padding

            cropped_image = image.crop((x_min, y_min, x_max, y_max))
            preprocessed_image = self.preprocess_images(cropped_image).unsqueeze(0)
            preprocessed_images.append(preprocessed_image)

            class_id = detections.class_id[idx]
            text_tokens.append(classes[class_id])
            image_crops.append(cropped_image)

    
        # Convert the list to batches of size self.batch_size
        text_batches = [text_tokens[i:i+self.batch_size] for i in range(0, len(text_tokens), self.batch_size)]
        image_batches = [preprocessed_images[i:i+self.batch_size] for i in range(0, len(preprocessed_images), self.batch_size)]
        
        assert len(text_batches) == len(image_batches), "Text and Image batches are not equal"
        for text_batch, image_batch in zip(text_batches, image_batches):
            num_images = len(image_batch)
            self.inputs[1]['host'] = self.preprocess_texts(text_batch).flatten()

            # TODO:: Handle this part - we already have an image, it just needs conversion to proper format
            self.inputs[0]['host'] = np.concatenate(image_batch, axis=0).flatten()
            
            image_encoding, text_encoding = self.infer()  # dimension is (batch_size,1024)
            
            text_encoding = text_encoding / np.linalg.norm(text_encoding, axis=-1, keepdims=True)
            image_encoding = image_encoding / np.linalg.norm(image_encoding, axis=-1, keepdims=True)
            
            for enc in range(num_images):
                all_image_encodings.append(image_encoding[enc])
                all_text_encodings.append(text_encoding[enc])
                
        # all_image_encodings = np.concatenate(all_image_encodings, axis=0)
        # all_text_encodings = np.concatenate(all_text_encodings, axis=0)
        all_image_encodings_numpy = np.vstack(all_image_encodings)
        all_text_encodings_numpy = np.vstack(all_text_encodings)
        return image_crops, all_image_encodings_numpy, all_text_encodings_numpy
            
    def encode_text(self, text_query:list):
        self.ctx.push()
        text_tokens = text_query
        all_text_encodings = []
    
        # Convert the list to batches of size self.batch_size
        text_batches = [text_tokens[i:i+self.batch_size] for i in range(0, len(text_tokens), self.batch_size)]
        # Create batches of dummy images
        # dummy_image = np.zeros((3, 224, 224), dtype=np.float32)
        # dummy_image_pil = Image.fromarray((dummy_image.transpose(1, 2, 0) * 255).astype(np.uint8))
        
        # dummy_image_preprocessed = self.preprocess_images(dummy_image_pil).unsqueeze(0)
        # print("Dummy image size: ", dummy_image_preprocessed.shape)
        # dummy_image_batch = [dummy_image_preprocessed for _ in range(self.batch_size)]
        # print(np.concatenate(dummy_image_batch, axis=0).flatten().shape)
        # dummy_image_batch = [dummy_image for _ in range(self.batch_size)]
        
        # print("do we reach ehre 1 --- ", self.batch_size, text_batches, len(dummy_image_batch))
        for text_batch in text_batches:
            num_text = len(text_batch)

            self.inputs[1]['host'] = self.preprocess_texts(text_batch).flatten()
            # self.inputs[0]['host'] = np.concatenate(dummy_image_batch, axis=0).flatten()

            text_encoding = self.infer_text()  # dimension is (batch_size,1024)


            text_encoding = text_encoding / np.linalg.norm(text_encoding, axis=-1, keepdims=True)
            for enc in range(num_text):
                all_text_encodings.append(text_encoding[enc])
            # all_text_encodings = np.concatenate(all_text_encodings, axis=0)
            all_text_encodings_numpy = np.vstack(all_text_encodings)
        return all_text_encodings_numpy
        
        
    def preprocess_texts(self, texts):    
        encoded_inputs = self.clip_tokenizer(texts)  
        # print(encoded_inputs.numpy().astype(np.int32).shape)
        return encoded_inputs.numpy().astype(np.int32)

    def preprocess_images(self, image_crops):
        # TODO:: Handle this part - we already have an image, it just needs conversion to proper format
        transform = transforms.Compose([
            transforms.Resize(224, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])
        return transform(image_crops)
        # images = [transform(Image.open(image_path).convert("RGB")).unsqueeze(0) for image_path in image_paths]
        # return np.concatenate(images, axis=0)
    
    
    def infer(self):
        # Copy inputs to device
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)  # For image
        cuda.memcpy_htod_async(self.inputs[1]['device'], self.inputs[1]['host'], self.stream)  # For text
        
        # # Execute inference
        # context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # # Copy outputs from device to host
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)  # For image encoding
        cuda.memcpy_dtoh_async(self.outputs[1]['host'], self.outputs[1]['device'], self.stream)  # For text encoding

        self.stream.synchronize()
        return self.outputs[0]['host'].reshape(self.batch_size, -1), self.outputs[1]['host'].reshape(self.batch_size, -1)
    
    
    def infer_text(self):
        # Copy inputs to device
        cuda.memcpy_htod_async(self.inputs[1]['device'], self.inputs[1]['host'], self.stream)  # For text
        # # Execute inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        # # Copy outputs from device to host
        cuda.memcpy_dtoh_async(self.outputs[1]['host'], self.outputs[1]['device'], self.stream)  # For text encoding        
        self.stream.synchronize()
        return self.outputs[1]['host'].reshape(self.batch_size, -1)