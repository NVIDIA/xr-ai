# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Centralized configuration management for Semantic SLAM Server."""

from pathlib import Path
from typing import Optional
import os
from dataclasses import dataclass, field
import yaml


@dataclass
class VideoConfig:
    """Video processing configuration."""
    # Image dimensions
    image_width: int = 1280
    image_height: int = 720
    
    # Depth dimensions  
    depth_width: int = 256
    depth_height: int = 144
    
    # Dataset-specific dimensions
    dataset_depth_width: int = 1200
    dataset_depth_height: int = 680
    
    # Processing parameters
    sharpness_threshold: float = 100.0
    temp_output_dir: str = "./main_server/temp_output_dir"

@dataclass  
class DatasetConfig:
    """Dataset saving configuration."""
    enabled: bool = False  # Whether to save dataset files
    output_directory: str = "./datasets"
    auto_increment_dirs: bool = True  # Auto-create dataset_0, dataset_1, etc.
    
    # What to save
    save_images: bool = True
    save_depth: bool = True
    save_poses: bool = True
    
    # Directory structure
    results_subdir: str = "results"

@dataclass
class FrameProcessingConfig:
    """Frame processing configuration."""
    default_client_fps: int = 30
    initial_frame_index: int = -1000
    frame_skip_logic: str = "fps_based"  # "fps_based" or "interval_based"

@dataclass
class GRPCConfig:
    """gRPC server configuration."""
    max_send_message_length: int = 100 * 1024 * 1024  # 100MB
    max_receive_message_length: int = 100 * 1024 * 1024  # 100MB
    keepalive_time_ms: int = 30000
    keepalive_timeout_ms: int = 5000
    keepalive_permit_without_calls: bool = True
    max_connection_idle_ms: int = 60000

@dataclass 
class LoggingConfig:
    """Performance logging configuration."""
    enabled: bool = True
    log_directory: str = "logs_performance"
    console_log_interval: int = 20  # Log stats every N frames
    csv_logging: bool = True
    json_summary: bool = True
    keep_recent_frames: int = 100  # For FPS calculation

@dataclass
class DebugConfig:
    """Debug configuration for troubleshooting and development."""
    dump_inference: bool = False  # Enable dumping of inference results (detections, CLIP features)
    dump_dir: str = "./debug_dumps"  # Directory to save debug dumps
    current_scene: Optional[str] = None  # Scene name for organized dumps (auto-detected if null)
    config_name: Optional[str] = None  # Config name for organized dumps (auto-detected if null)
    use_slow_vis: bool = False  # Enable slow high-quality visualization with captions

@dataclass
class ServerConfig:
    """Main server configuration."""
    host: str = "localhost"
    port: int = 50051
    max_workers: int = 4
    target_fps: float = 3.0  # Fixed to match current server.py default
    gpu_id: int = 0
    
    # Queue sizes for pipeline stages
    inference_queue_size: int = 2  # GRPC → Inference
    mapping_queue_size: int = 5    # Inference → Mapping  
    visualization_queue_size: int = 2  # Mapping → Visualization


@dataclass
class DetectionConfig:
    """Object detection model configuration."""
    enabled: bool = True  # Whether to use detection system (replaces --useDetector)
    device: str = "cuda:0"  # Device for detection model
    detector: str = "dino"  # Detection model type
    box_threshold: float = 0.2
    text_threshold: float = 0.2  
    nms_threshold: float = 0.5
    precision: str = "float16"  # Model precision: "float16" or "float32"

@dataclass
class SegmentationConfig:
    """Segmentation model configuration."""
    device: str = "cuda:0"  # Device for segmentation model
    sam_variant: str = "sam"  # Options: sam, mobilesam, lighthqsam
    batched_sam: bool = True
    trt_sam: bool = False

@dataclass
class CLIPConfig:
    """CLIP model configuration."""
    device: str = "cuda:0"  # Device for CLIP model
    model_name: str = "ViT-H-14"
    pretrained: str = "laion2b_s32b_b79k"
    precision: str = "fp16"
    batch_size: int = 8
    batched_clip: bool = False
    trt_clip: bool = False

@dataclass
class CaptioningConfig:
    """Image captioning model configuration."""
    device: str = "cuda:0"  # Device for captioning model
    class_set: str = "ram"
    tagging_threshold: float = 0.64
    add_bg_classes: bool = True
    accumu_classes: bool = True
    precision: str = "float16"  # Precision for model inference: "float32" or "float16"

@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    device: str = "cuda:0"  # Device for visualization
    similarity_threshold: float = 0.86
    clip_model: str = "ViT-H-14"
    pretrained: str = "laion2b_s32b_b79k"
    precision: str = "fp16"
    batch_size: int = 1
    use_trt: bool = False

@dataclass
class MappingConfig:
    """3D mapping configuration."""
    device: str = "cuda:0"  # Device for mapping operations
    mask_conf_threshold: float = 0.25
    sim_threshold: float = 0.5  # Placeholder - need to find actual default
    skip_bg: bool = False
    object_based_downsampling: bool = False
    test_depth_downsampling: int = 1

@dataclass
class ModelConfig:
    """Complete model configuration."""
    device: str = "cuda:0"  # Default device (can be overridden per model)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    clip: CLIPConfig = field(default_factory=CLIPConfig)
    captioning: CaptioningConfig = field(default_factory=CaptioningConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    
    def set_all_devices(self, device: str) -> None:
        """Set device for all models at once."""
        self.device = device
        self.detection.device = device
        self.segmentation.device = device
        self.clip.device = device
        self.captioning.device = device
        self.visualization.device = device
        self.mapping.device = device
        
    def get_device_summary(self) -> dict:
        """Get a summary of device assignments."""
        return {
            "detection": self.detection.device,
            "segmentation": self.segmentation.device,
            "clip": self.clip.device,
            "captioning": self.captioning.device,
            "visualization": self.visualization.device,
            "mapping": self.mapping.device,
        }


@dataclass
class PathConfig:
    """Path configuration."""
    base_dir: Path = Path(__file__).parent.parent
    gsa_path: Optional[Path] = None
    model_weights_dir: Optional[Path] = None
    temp_output_dir: Optional[Path] = None
    
    def __post_init__(self):
        """Set defaults and validate paths."""
        # Convert strings to Path objects if needed
        if isinstance(self.base_dir, str):
            self.base_dir = Path(self.base_dir)
            
        # Set defaults from environment or fallback
        if self.gsa_path is None:
            if "GSA_PATH" in os.environ:
                self.gsa_path = Path(os.environ["GSA_PATH"])
            else:
                self.gsa_path = self.base_dir / "external" / "Grounded-Segment-Anything"
                
        if isinstance(self.gsa_path, str):
            self.gsa_path = Path(self.gsa_path)
            
        # Validate critical paths
        if not self.gsa_path.exists():
            raise ValueError(f"GSA path not found: {self.gsa_path}")
            
        # Set model weights directory
        if self.model_weights_dir is None:
            self.model_weights_dir = self.base_dir / "model_weights"
            
        # Set temp output directory  
        if self.temp_output_dir is None:
            self.temp_output_dir = self.base_dir / "main_server" / "temp_output_dir"


@dataclass
class Config:
    """Main configuration class."""
    server: ServerConfig = field(default_factory=ServerConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    frame_processing: FrameProcessingConfig = field(default_factory=FrameProcessingConfig)
    grpc: GRPCConfig = field(default_factory=GRPCConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Create config with dataclass defaults first
        config = cls()
        
        # Override with YAML values
        if 'server' in data:
            for key, value in data['server'].items():
                if hasattr(config.server, key):
                    setattr(config.server, key, value)
                    
        if 'video' in data:
            for key, value in data['video'].items():
                if hasattr(config.video, key):
                    setattr(config.video, key, value)
                    
        if 'dataset' in data:
            for key, value in data['dataset'].items():
                if hasattr(config.dataset, key):
                    setattr(config.dataset, key, value)
                    
        if 'frame_processing' in data:
            for key, value in data['frame_processing'].items():
                if hasattr(config.frame_processing, key):
                    setattr(config.frame_processing, key, value)
                    
        if 'grpc' in data:
            for key, value in data['grpc'].items():
                if hasattr(config.grpc, key):
                    setattr(config.grpc, key, value)
                    
        if 'logging' in data:
            for key, value in data['logging'].items():
                if hasattr(config.logging, key):
                    setattr(config.logging, key, value)
                    
        if 'model' in data:
            model_data = data['model']
            if 'device' in model_data:
                config.model.device = model_data['device']
                
            # Update each model config section
            for model_type in ['detection', 'segmentation', 'clip', 'captioning', 'visualization', 'mapping']:
                if model_type in model_data:
                    model_config = getattr(config.model, model_type)
                    for key, value in model_data[model_type].items():
                        if hasattr(model_config, key):
                            setattr(model_config, key, value)
                            
        if 'paths' in data:
            for key, value in data['paths'].items():
                if hasattr(config.paths, key) and value is not None:
                    setattr(config.paths, key, Path(value))
        
        if 'debug' in data:
            for key, value in data['debug'].items():
                if hasattr(config.debug, key):
                    setattr(config.debug, key, value)
                    
        return config
    
    @classmethod 
    def from_env(cls) -> 'Config':
        """Load configuration from environment variables."""
        config = cls()
        
        # Override with environment variables
        if os.getenv("GSA_PATH"):
            config.paths.gsa_path = Path(os.getenv("GSA_PATH"))
        
        # Global device setting (applies to all models if not overridden)
        if os.getenv("DEVICE"):
            global_device = os.getenv("DEVICE")
            config.model.device = global_device
            config.model.detection.device = global_device
            config.model.segmentation.device = global_device
            config.model.clip.device = global_device
            config.model.captioning.device = global_device
            config.model.visualization.device = global_device
            config.model.mapping.device = global_device
            
        # Per-model device overrides
        if os.getenv("DETECTION_DEVICE"):
            config.model.detection.device = os.getenv("DETECTION_DEVICE")
        if os.getenv("SEGMENTATION_DEVICE"):
            config.model.segmentation.device = os.getenv("SEGMENTATION_DEVICE")
        if os.getenv("CLIP_DEVICE"):
            config.model.clip.device = os.getenv("CLIP_DEVICE")
        if os.getenv("CAPTIONING_DEVICE"):
            config.model.captioning.device = os.getenv("CAPTIONING_DEVICE")
        if os.getenv("VISUALIZATION_DEVICE"):
            config.model.visualization.device = os.getenv("VISUALIZATION_DEVICE")
        if os.getenv("MAPPING_DEVICE"):
            config.model.mapping.device = os.getenv("MAPPING_DEVICE")
            
        # Other settings
        if os.getenv("SERVER_PORT"):
            config.server.port = int(os.getenv("SERVER_PORT"))
        if os.getenv("SAM_VARIANT"):
            config.model.segmentation.sam_variant = os.getenv("SAM_VARIANT")
        if os.getenv("CLIP_MODEL"):
            config.model.clip.model_name = os.getenv("CLIP_MODEL")
            
        # Debug settings
        if os.getenv("DEBUG_DUMP_INFERENCE"):
            config.debug.dump_inference = os.getenv("DEBUG_DUMP_INFERENCE").lower() in ['true', '1', 'yes']
        if os.getenv("DEBUG_DUMP_DIR"):
            config.debug.dump_dir = os.getenv("DEBUG_DUMP_DIR")
        if os.getenv("DEBUG_CURRENT_SCENE"):
            config.debug.current_scene = os.getenv("DEBUG_CURRENT_SCENE")
        if os.getenv("DEBUG_CONFIG_NAME"):
            config.debug.config_name = os.getenv("DEBUG_CONFIG_NAME")
            
        return config

    @property 
    def gsa_path(self) -> Path:
        """Convenient access to GSA path."""
        return self.paths.gsa_path
        
    @property
    def grounding_dino_config_path(self) -> Path:
        """Path to GroundingDINO config."""
        return self.gsa_path / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
        
    @property
    def grounding_dino_checkpoint_path(self) -> Path:
        """Path to GroundingDINO checkpoint."""
        return self.gsa_path / "groundingdino_swint_ogc.pth"
        
    @property
    def sam_checkpoint_path(self) -> Path:
        """Path to SAM checkpoint."""
        return self.gsa_path / "sam_vit_h_4b8939.pth"
        
    @property
    def mobile_sam_checkpoint_path(self) -> Path:
        """Path to MobileSAM checkpoint."""
        return self.gsa_path / "EfficientSAM" / "mobile_sam.pt"
        
    @property
    def hqsam_checkpoint_path(self) -> Path:
        """Path to HQSAM checkpoint."""
        return self.gsa_path / "EfficientSAM" / "sam_hq_vit_tiny.pth"


# Global instance for easy access
_global_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance.
    
    Loading order:
    1. YAML defaults (config/defaults.yaml) if exists
    2. Dataclass defaults as fallback
    3. Environment variable overrides
    """
    global _global_config
    if _global_config is None:
        # Try to load from YAML first
        defaults_path = Path(__file__).parent / "defaults.yaml"
        if defaults_path.exists():
            try:
                _global_config = Config.from_yaml(str(defaults_path))
            except Exception as e:
                print(f"WARNING: Failed to load {defaults_path}, using dataclass defaults: {e}")
                _global_config = Config.from_env()
        else:
            _global_config = Config.from_env()
            
        # Apply environment variable overrides if we loaded from YAML
        if defaults_path.exists():
            _global_config = _apply_env_overrides(_global_config)
    return _global_config


def _apply_env_overrides(config: Config) -> Config:
    """Apply environment variable overrides to an existing config."""
    import os
    
    # Global device setting (applies to all models if not overridden)
    if os.getenv("DEVICE"):
        global_device = os.getenv("DEVICE")
        config.model.device = global_device
        config.model.detection.device = global_device
        config.model.segmentation.device = global_device
        config.model.clip.device = global_device
        config.model.captioning.device = global_device
        config.model.visualization.device = global_device
        config.model.mapping.device = global_device
        
    # Per-model device overrides
    if os.getenv("DETECTION_DEVICE"):
        config.model.detection.device = os.getenv("DETECTION_DEVICE")
    if os.getenv("DETECTION_ENABLED"):
        config.model.detection.enabled = os.getenv("DETECTION_ENABLED").lower() in ['true', '1', 'yes']
    if os.getenv("SEGMENTATION_DEVICE"):
        config.model.segmentation.device = os.getenv("SEGMENTATION_DEVICE")
    if os.getenv("CLIP_DEVICE"):
        config.model.clip.device = os.getenv("CLIP_DEVICE")
    if os.getenv("CAPTIONING_DEVICE"):
        config.model.captioning.device = os.getenv("CAPTIONING_DEVICE")
    if os.getenv("VISUALIZATION_DEVICE"):
        config.model.visualization.device = os.getenv("VISUALIZATION_DEVICE")
    if os.getenv("MAPPING_DEVICE"):
        config.model.mapping.device = os.getenv("MAPPING_DEVICE")
        
    # Other settings
    if os.getenv("GSA_PATH"):
        config.paths.gsa_path = Path(os.getenv("GSA_PATH"))
    if os.getenv("SERVER_PORT"):
        config.server.port = int(os.getenv("SERVER_PORT"))
    if os.getenv("SAM_VARIANT"):
        config.model.segmentation.sam_variant = os.getenv("SAM_VARIANT")
    if os.getenv("CLIP_MODEL"):
        config.model.clip.model_name = os.getenv("CLIP_MODEL")
        
    # Debug settings
    if os.getenv("DEBUG_DUMP_INFERENCE"):
        config.debug.dump_inference = os.getenv("DEBUG_DUMP_INFERENCE").lower() in ['true', '1', 'yes']
    if os.getenv("DEBUG_DUMP_DIR"):
        config.debug.dump_dir = os.getenv("DEBUG_DUMP_DIR")
    if os.getenv("DEBUG_CURRENT_SCENE"):
        config.debug.current_scene = os.getenv("DEBUG_CURRENT_SCENE")
    if os.getenv("DEBUG_CONFIG_NAME"):
        config.debug.config_name = os.getenv("DEBUG_CONFIG_NAME")
    if os.getenv("DEBUG_USE_SLOW_VIS"):
        config.debug.use_slow_vis = os.getenv("DEBUG_USE_SLOW_VIS").lower() in ['true', '1', 'yes']
        
    return config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _global_config
    _global_config = config