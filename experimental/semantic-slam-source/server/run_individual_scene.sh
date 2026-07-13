#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0




all_scenes=("room0" "room1" "room2" "office0" "office1" "office2" "office3" "office4")

for scene in "${all_scenes[@]}"; do
    # python main.py --dataset_type replica --localDataset --save_map --dataset_stride 5 --config ../config/defaults.yaml --sceneName $scene
    # python main.py --dataset_type replica --localDataset --save_map --dataset_stride 5 --config ../config/noHalf.yaml --sceneName $scene
    # python main.py --dataset_type replica --localDataset --save_map --dataset_stride 5 --config ../config/ClipCaptionHalf.yaml --sceneName $scene
    # python main.py --dataset_type replica --localDataset --save_map --dataset_stride 5 --config ../config/ClipDetHalf.yaml --sceneName $scene
    # python main.py --dataset_type replica --localDataset --save_map --dataset_stride 5 --config ../config/onlyClipHalf.yaml --sceneName $scene
    # python main.py --dataset_type replica --localDataset --save_map --dataset_stride 5 --config ../config/defaults_mobileclip.yaml --sceneName $scene
    # python main.py --dataset_type replica --localDataset --save_map --dataset_stride 5 --config ../config/parallelization.yaml --sceneName $scene
    # python main.py --dataset_type replica --localDataset --save_map --dataset_stride 5 --config ../config/parallelization_mobileclip.yaml --sceneName $scene
    python main.py --dataset_type replica --localDataset --save_map --dataset_stride 5 --config ../config/diff_parallel.yaml --sceneName $scene

done

