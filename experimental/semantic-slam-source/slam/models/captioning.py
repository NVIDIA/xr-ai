# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Captioning module for semantic SLAM operations."""

import asyncio
import cv2
import gzip
import io
import os
import pickle
import sys
import threading
import time
from typing import Any, List
from PIL import Image

import numpy as np
import torch

# Set up GSA path from environment
# Import configuration system
from config.settings import get_config

# Get GSA paths from config
config = get_config()
GSA_PATH = str(config.gsa_path)
TAG2TEXT_PATH = str(config.gsa_path)
EFFICIENTSAM_PATH = str(config.gsa_path / "EfficientSAM")

# Add GSA paths for external dependencies
if GSA_PATH not in sys.path:
    sys.path.append(GSA_PATH)
if TAG2TEXT_PATH not in sys.path:
    sys.path.append(TAG2TEXT_PATH)
if EFFICIENTSAM_PATH not in sys.path:
    sys.path.append(EFFICIENTSAM_PATH)
import torchvision.transforms as TS
try:
    from ram.models import ram
    from ram.models import tag2text
    from ram import inference_tag2text, inference_ram
except ImportError as e:
    print("RAM sub-package not found. Please check your GSA_PATH. ")
    raise e

TAG2TEXT_CHECKPOINT_PATH = os.path.join(TAG2TEXT_PATH, "./tag2text_swin_14m.pth")
RAM_CHECKPOINT_PATH = os.path.join(TAG2TEXT_PATH, "./ram_swin_large_14m.pth")

class captioning():
    def __init__(self, class_set="ram", device="cuda:0", add_bg_classes=True, accumu_classes=True, precision=None):
        self.device = device
        self.class_set = class_set
        self.add_bg_classes = add_bg_classes
        self.accumu_classes = accumu_classes
        
        # Get config for precision setting
        from config.settings import get_config
        current_config = get_config()
        
        # Set precision from config or parameter
        self.precision = precision if precision is not None else current_config.model.captioning.precision
        
        # Set dtype based on precision
        if self.precision == "float16":
            self.dtype = torch.float16
        elif self.precision == "float32":
            self.dtype = torch.float32
        else:
            raise ValueError(f"Unsupported precision: {self.precision}. Choose 'float32' or 'float16'.")
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your CUDA installation.")
        
        
        if self.class_set == "tag2text":
            # The class set will be computed by tag2text on each image
            # filter out attributes and action categories which are difficult to grounding
            delete_tag_index = []
            for i in range(3012, 3429):
                delete_tag_index.append(i)

            self.specified_tags='None'
            # load model
            self.tagging_model = tag2text.tag2text_caption(pretrained=TAG2TEXT_CHECKPOINT_PATH,
                                                    image_size=384,
                                                    vit='swin_b',
                                                    delete_tag_index=delete_tag_index)
            # threshold for tagging - use config value
            # we reduce the threshold to obtain more tags
            if config is None:
                from config.settings import get_config
                config = get_config()
            self.tagging_model.threshold = config.model.captioning.tagging_threshold 
        elif self.class_set == "ram":
            self.tagging_model = ram(pretrained=RAM_CHECKPOINT_PATH,
                                         image_size=384,
                                         vit='swin_l')
            
        self.tagging_model = self.tagging_model.eval()
        self.tagging_model = self.tagging_model.to(self.device)
        
        # Apply precision setting to the model
        if self.precision == "float16":
            self.tagging_model = self.tagging_model.half()
        elif self.precision == "float32":
            self.tagging_model = self.tagging_model.float()
        
        # initialize Tag2Text
        self.tagging_transform = TS.Compose([
            TS.Resize((384, 384)),
            TS.ToTensor(), 
            TS.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])
        
        # Add "other item" to capture objects not in the tag2text captions. 
        # Remove "xxx room", otherwise it will simply include the entire image
        # Also hide "wall" and "floor" for now...
        self.add_classes = ["other item"]
        self.remove_classes = [
            "room", "kitchen", "office", "house", "home", "building", "corner",
            "shadow", "carpet", "photo", "shade", "stall", "space", "aquarium", 
            "apartment", "image", "city", "blue", "skylight", "hallway", 
            "bureau", "modern", "salon", "doorway", "wall lamp"
        ]
        self.bg_classes = ["wall", "floor", "ceiling"]

        if self.add_bg_classes:
            self.add_classes += self.bg_classes
        else:
            self.remove_classes += self.bg_classes
        
        self.classes = None
        self.global_classes = set()

    def process_tag_classes(self, text_prompt:str) -> list[str]:
        '''
        Convert a text prompt from Tag2Text to a list of classes. 
        '''
        classes = text_prompt.split(',')
        classes = [obj_class.strip() for obj_class in classes]
        classes = [obj_class for obj_class in classes if obj_class != '']
        
        for c in self.add_classes:
            if c not in classes:
                classes.append(c)
        
        for c in self.remove_classes:
            classes = [obj_class for obj_class in classes if c not in obj_class.lower()]
        
        return classes
    
    def gen_caption(self, image_pil):
        raw_image = image_pil.resize((384, 384))
        raw_image = self.tagging_transform(raw_image).unsqueeze(0).to(self.device, dtype=self.dtype)
        
        if self.class_set == "ram":
            # tag_start = time.perf_counter_ns()
            res = inference_ram(raw_image , self.tagging_model)
            # tagging_time += (time.perf_counter_ns() - tag_start)
            caption="NA"
        elif self.class_set == "tag2text":
            res = inference_tag2text.inference(raw_image , self.tagging_model, self.specified_tags)
            caption=res[2]

        # Currently ", " is better for detecting single tags
        # while ". " is a little worse in some case
        text_prompt=res[0].replace(' |', ',')
        # print(f" Sept30Frame {idx}: RAM text_prompt: {text_prompt}")

        self.classes = self.process_tag_classes(
            text_prompt
        )
        # print(f"Sept30 Frame {idx}: RAM classes for GDINO: {len(classes)} {classes}")
        
        # add classes to global classes
        self.global_classes.update(self.classes)
        
        if self.accumu_classes:
            # Use all the classes that have been seen so far
            self.classes = list(self.global_classes)
        return caption, text_prompt
    