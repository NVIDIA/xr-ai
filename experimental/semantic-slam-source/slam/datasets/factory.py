# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset factory for creating dataset instances dynamically."""

import os
from typing import Dict, Any, Optional

from .replica import ReplicaDataset
from .scannet import ScanNetDataset


class DatasetFactory:
    """Factory class for creating local dataset instances.
    
    Note: iPad is not included as it uses real-time streaming, not local file processing.
    iPad datasets are directly instantiated in the streaming services.
    """
    
    # Registry of available local datasets
    DATASETS = {
        'replica': ReplicaDataset,
        'scannet': ScanNetDataset,
    }
    
    @classmethod
    def create_dataset(cls, dataset_type: str, config_dict: Dict[str, Any], **kwargs):
        """Create a local dataset instance based on the dataset type.
        
        Args:
            dataset_type: Type of local dataset ('replica', 'scannet')
            config_dict: Configuration dictionary for the dataset
            **kwargs: Additional arguments to pass to the dataset constructor
            
        Returns:
            Dataset instance for local file processing
            
        Raises:
            ValueError: If dataset_type is not supported for local processing
        """
        dataset_type = dataset_type.lower()
        
        if dataset_type not in cls.DATASETS:
            available = ', '.join(cls.DATASETS.keys())
            raise ValueError(f"Dataset type '{dataset_type}' not supported. Available: {available}")
        
        dataset_class = cls.DATASETS[dataset_type]
        return dataset_class(config_dict=config_dict, **kwargs)
    
    @classmethod
    def get_available_datasets(cls):
        """Get list of available dataset types."""
        return list(cls.DATASETS.keys())
    
    @classmethod
    def register_dataset(cls, name: str, dataset_class):
        """Register a new dataset type.
        
        Args:
            name: Name of the dataset type
            dataset_class: Dataset class to register
        """
        cls.DATASETS[name.lower()] = dataset_class


class DatasetProcessor:
    """Processor for handling dataset-specific operations."""
    
    @staticmethod
    def get_dataset_paths(dataset_type: str, scene_name: str) -> Dict[str, str]:
        """Get dataset-specific paths for a given scene.
        
        Args:
            dataset_type: Type of dataset
            scene_name: Name of the scene
            
        Returns:
            Dictionary containing dataset paths
        """
        dataset_type = dataset_type.lower()
        
        if dataset_type == 'replica':
            return DatasetProcessor._get_replica_paths(scene_name)
        elif dataset_type == 'scannet':
            return DatasetProcessor._get_scannet_paths(scene_name)
        else:
            raise ValueError(f"Unsupported dataset type for local file processing: {dataset_type}. iPad uses real-time streaming.")
    
    @staticmethod
    def _get_replica_paths(scene_name: str) -> Dict[str, str]:
        """Get Replica dataset paths."""
        replica_root = os.environ.get('REPLICA_ROOT')
        if not replica_root:
            raise ValueError("REPLICA_ROOT environment variable not set")
        
        dataset_path = os.path.join(replica_root, scene_name)
        return {
            'dataset_path': dataset_path,
            'image_path': os.path.join(dataset_path, "results"),
            'pose_path': os.path.join(dataset_path, "traj.txt"),
            'image_format': "frame{:06d}.jpg",
            'depth_format': "depth{:06d}.png"
        }
    
    @staticmethod
    def _get_scannet_paths(scene_name: str) -> Dict[str, str]:
        """Get ScanNet dataset paths."""
        scannet_root = os.environ.get('SCANNET_ROOT')
        if not scannet_root:
            raise ValueError("SCANNET_ROOT environment variable not set")
        
        dataset_path = os.path.join(scannet_root, scene_name)
        assert os.path.exists(dataset_path), f"Dataset path {dataset_path} does not exist"
        return {
            'dataset_path': dataset_path,
            'image_path': os.path.join(dataset_path, "color"),
            'depth_path': os.path.join(dataset_path, "depth"),
            'pose_path': os.path.join(dataset_path, "pose"),
            'intrinsic_path': os.path.join(dataset_path, "intrinsic", "intrinsic_color.txt"),
            'image_format': "*.jpg",
            'depth_format': "*.png"
        }
    

    @staticmethod
    def get_default_scenes(dataset_type: str) -> list:
        """Get default scene names for a dataset type.
        
        Args:
            dataset_type: Type of dataset
            
        Returns:
            List of default scene names
        """
        dataset_type = dataset_type.lower()
        
        if dataset_type == 'replica':
            return ['room0', 'room1', 'room2', 'office0', 'office1', 'office2', 'office3', 'office4']
        elif dataset_type == 'scannet':
            # ScanNet scenes would need to be discovered dynamically or configured
            return ["scene0011_00"]  # Return empty list for now, to be filled based on available scenes
        elif dataset_type == 'ipad':
            return []  # iPad scenes are typically custom recorded scenes
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
