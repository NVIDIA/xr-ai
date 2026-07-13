#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dataset_type>"
    exit 1
fi

dataset_type=$1

# for jetson client, if you want to use the clientUpdateMode, you need to specify the clientIP
# python server.py --useDetector --sam_variant mobilesam --batched_sam --trt_clip --target_fps 3 --batched_clip --vis_gpu cuda:1 --dataset_type $dataset_type  --clientUpdateMode --clientIP "10.20.23.32"

# for iPad client. No clientUpdateMode here.
# python server.py --useDetector --sam_variant mobilesam --batched_sam --trt_clip --target_fps 3 --batched_clip --vis_gpu cuda:1 --dataset_type $dataset_type  --vis_useTRT
# python server.py --useDetector --sam_variant mobilesam --batched_sam --trt_clip --target_fps 3 --batched_clip --vis_gpu cuda:1 --dataset_type $dataset_type
# python server.py --useDetector --sam_variant mobilesam --batched_sam --trt_clip --target_fps 3 --batched_clip --vis_gpu cuda:1 --dataset_type $dataset_type --pipelined_mapping


python server.py --useDetector --sam_variant mobilesam --batched_sam --target_fps 2 --batched_clip --vis_gpu cuda:1 --dataset_type $dataset_type


# For testing with dataset
# python server.py --useDetector --sam_variant mobilesam --batched_sam --target_fps 2 --batched_clip --vis_gpu cuda:1 --dataset_type replica --useDataset room0 --dataset_stride 5
