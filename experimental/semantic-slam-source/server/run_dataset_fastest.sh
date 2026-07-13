#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


python main.py --dataset_type replica --localDataset --save_map --dataset_stride 5 --config ../config/defaults.yaml
# python main.py --dataset_type replica --localDataset --save_map --dataset_stride 5 --config ../config/defaults_mobileclip.yaml
# python main.py --dataset_type replica --localDataset --save_map --dataset_stride 5 --config ../config/parallelization.yaml
# python main.py --dataset_type replica --localDataset --save_map --dataset_stride 5 --config ../config/parallelization_mobileclip.yaml