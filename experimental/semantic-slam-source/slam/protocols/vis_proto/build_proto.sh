# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

python -m grpc_tools.protoc -I. --python_out=. --pyi_out=. --grpc_python_out=. vis.proto