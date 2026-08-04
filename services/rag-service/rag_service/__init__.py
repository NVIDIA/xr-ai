# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dense document retrieval service."""

from .index import DenseIndex
from .service import RAGService

__all__ = ["DenseIndex", "RAGService"]
