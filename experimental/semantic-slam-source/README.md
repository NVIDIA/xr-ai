<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Semantic SLAM Server

A real-time semantic SLAM system that combines object detection, segmentation, and CLIP-based understanding for interactive 3D scene mapping and querying.

## Overview

This semantic SLAM server provides:
- **Real-time 3D mapping** with object-level semantic understanding
- **Multi-modal AI pipeline** (detection → segmentation → CLIP encoding → mapping)
- **Interactive querying** using natural language descriptions
- **Configurable architecture** supporting multiple datasets and hardware setups
- **Performance monitoring** with detailed logging and analytics

## Clean Python API (`semantic_slam`)

A small, importable façade over the full pipeline. Push RGB/depth/pose frames
and query the resulting semantic map with natural language — no gRPC, no
multiprocess server.

```python
import numpy as np
from semantic_slam import SemanticSLAM

slam = SemanticSLAM(dataset_type="replica", scene_name="room0")  # builds models
slam.push(rgb, depth, pose)          # rgb HxWx3 uint8, depth HxW, pose 4x4 (or flat-16)
hits = slam.query("a brown chair", top_k=5)
for h in hits:
    print(h["score"], h["centroid"], h["class_name"])
slam.save_map("map.pkl.gz")          # also: load_map(), reset()
```

### Install with uv (replaces the old conda env)

Dependencies are managed by [uv](https://docs.astral.sh/uv/). The core stack is
locked in `pyproject.toml` / `uv.lock`; the heavy model stack (pytorch3d,
gradslam, segment-anything — all build CUDA extensions / come from git) plus the
SAM weights are installed by a **one-time, idempotent setup script** that runs
automatically on first launch:

```bash
uv sync                       # core deps into an isolated .venv
scripts/setup_env.sh          # model stack + SAM weights (idempotent; --with-dataset adds Replica)
```

You normally don't need to call the script yourself — the first
`SemanticSLAM(...)` runs it if anything is missing (disable with
`SEMANTIC_SLAM_AUTO_SETUP=0`). The script installs the CUDA build toolchain
(`nvcc`, `gcc-10`/`g++-10`) via `apt` only if it is absent, generates the gRPC
protobuf stubs, and downloads the SAM weights.

> **Note:** the source-built model stack lives in the venv but not the lockfile.
> Use `uv sync --inexact` (as the script does) — a plain `uv sync` would remove
> it. If it ever gets removed, just re-run `scripts/setup_env.sh` (cached builds
> make it fast).

### End-to-end test

```bash
REPLICA_ROOT=/data/replica/Replica .venv/bin/python -m pytest tests/test_replica_e2e.py -q -s
```

The CPU-only smoke test (`tests/test_semantic_slam_smoke.py`) needs none of the
above and runs anywhere.

### Visual validation

To eyeball that it works, `scripts/visualize_e2e.py` pushes a few Replica frames
and writes images to `outputs/semantic_slam_viz/`:

```bash
REPLICA_ROOT=/data/replica/Replica GSA_PATH=$PWD/external/Grounded-Segment-Anything \
  HF_HOME=$HOME/.cache/huggingface SEMANTIC_SLAM_AUTO_SETUP=0 \
  .venv/bin/python scripts/visualize_e2e.py
```

- `frame_XXXXXX_seg.png` — SAM masks overlaid on each input RGB frame.
- `map_bev.png` — bird's-eye scatter of all mapped object centroids.
- `query_<text>.png` — the BEV map with the top-k hits for a text query ringed
  and scored, so you can see where a query localizes in the scene.

### Use as a module in `nvidia/xr-ai`

The package is shaped to drop into `xr-ai`'s `ai-services/` layout (hatchling,
Apache-2.0 SPDX, pinned PyTorch CUDA index, `[project.scripts]` entry point),
matching the existing `vlm-server` / `stt-server`. To consume it from xr-ai,
add it as a path/git dependency and import `semantic_slam`.

## Quick Start

**Prerequisites:** Complete the [Installation](#installation) section first to install all dependencies.

```bash
# 1. After completing installation, install the package in development mode
pip install -e .

# 2. Run server with default settings
python server/grpc_server.py --dataset_type replica --useDataset room0

# 3. Monitor performance (in another terminal)
tail -f logs_performance/default/room0/frame_timing.csv
```

## Project Structure

```
semantic-slam-server/
├── server/                     # gRPC server and components
│   ├── grpc_server.py         # Main server entry point
│   └── components/            # Modular server components
├── slam/                      # Core SLAM functionality
│   ├── models/                # AI models (detection, segmentation, CLIP)
│   ├── services/              # Processing pipelines
│   ├── core/                  # SLAM data structures
│   └── utils/                 # Utilities and performance monitoring
├── config/                    # Configuration files
│   ├── defaults.yaml          # Default settings
│   ├── replica_config.yaml    # Replica dataset config
│   └── settings.py            # Configuration system
├── logs_performance/          # Performance monitoring
│   └── <config>/<scene>/      # Organized by config and scene
└── external/                  # External dependencies
    ├── gradslam/              # Point cloud mapping
    ├── chamferdist/           # Distance metrics
    └── Grounded-Segment-Anything/  # Vision models
```

## Installation

### Dependencies Setup

1. Create a new conda environment with Python 3.10:
    ```
    conda create -n sceneGraphDemo python=3.10
    ```

2. Activate the sceneGraphDemo environment:
    ```
    conda activate sceneGraphDemo
    ```

3. Install the required packages:
    ```
    pip install tyro open_clip_torch wandb h5py openai hydra-core distinctipy timm==0.4.12
    pip install pynvvideocodec
    python -m pip install grpcio
    python -m pip install grpcio-tools
    conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
    ```

    H.264/H.265 video decode is handled by NVIDIA's PyNvVideoCodec (Python bindings, MIT-licensed) backed by the NVIDIA Video Codec SDK. Download the SDK from https://developer.nvidia.com/video-codec-sdk (requires NVIDIA driver / CUDA).

4. Set the CUDA_HOME environment variable to the CUDA installation inside the conda environment:
    ```
    export CUDA_HOME=/home/<USERNAME>/miniconda3/envs/sceneGraphDemo/
    ```

5. Install additional packages:
    ```
    conda install -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl
    conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
    conda install -c fvcore -c conda-forge fvcore
    ```
    Optional
    ```
    conda install pytorch3d -c pytorch3d
    ```

6. Chamferdist - Module to compute Chamfer distance between two pointclouds. Clone the chamferdist repository and install it:
    ```
    git clone https://github.com/krrish94/chamferdist.git
    cd chamferdist
    Commit ID: ee75389 (TODO: need verification)
    pip install .
    ```

7. GradSLAM - Module used in mapping the point clouds. Clone the gradslam repository and switch to the conceptfusion branch:
    ```
    git clone https://github.com/gradslam/gradslam.git
    cd gradslam
    git checkout conceptfusion
    pip install .
    ```

8. Install the supervision package. This package has some datastructures that we use for object based point clouds:
    ```
    pip install supervision==0.14.0
    ```

9. Install the line_profiler package from conda-forge:
    ```
    conda install conda-forge::line_profiler
    ```

10. Clone the Grounded-Segment-Anything repository and switch to a specific commit:
     ```
     git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
     cd Grounded-Segment-Anything
     git checkout a4d76a2
     ```

11. Download the following files from [here](https://github.com/IDEA-Research/Grounded-Segment-Anything#install-without-docker) and place them in Grounding-Segment-Anything directory
     - ram_swin_large_14m.pth
     - groundingdino_swint_ogc.pth
     - sam_vit_h_4b8939.pth
     - Download the mobile_sam weights from [here](https://github.com/ChaoningZhang/MobileSAM/tree/master/weights) and put it in Grounded-SAM-Anything/EfficientSAM directory.

     ```
     python -m pip install -e segment_anything
     pip install --no-build-isolation -e GroundingDINO
     pip install --upgrade diffusers[torch]
     git submodule update --init --recursive
     cd grounded-sam-osx && bash install.sh
     cd ..
     git clone https://github.com/xinyu1205/recognize-anything.git
     Commit ID: 88c2b0c (TODO: need verification)
     pip install --upgrade setuptools
     pip install -r ./recognize-anything/requirements.txt
     pip install -e ./recognize-anything/
     ```
     
    

12. Set the GSA_PATH environment variables:
     ```
     export GSA_PATH=<path to Grounded-Segment-Anything directory>
     ```

13. Additional considerations - 
     - Perhaps need to downgrade scipy to 1.10, and opencv-python to opencv-python==4.9.0.80
     - Numpy must be - 1.24.3 or 1.26.4
     - pip install tensorrt==8.6.1, Native tensorrt should also be the same version.
     - pip install pycuda
     - Transformer version must be: pip install transformer==4.25.1

14. **Install the Semantic SLAM package** (REQUIRED - final step after all dependencies):
    ```bash
    # From the semantic-slam-server root directory
    pip install -e .
    ```
    **This is the critical step that:**
    - Installs the package in development mode
    - Makes all imports work properly (`from slam.models import ...`)
    - Enables the server to run (`python server/grpc_server.py`)
    - Must be done AFTER all dependencies are installed

15. **Verify Installation** (optional but recommended):
    ```bash
    # Test that core imports work
    python tests/test_basic_imports.py
    ```

16. Speech/Audio options and OpenAI key:

    **Option 1:** Use Audio/Speech. 
    
    Set the AUDIO_INPUT_USE_OPENAI flag to "True" in `slam/services/visualization_service.py`
    
    Next, you need to create an OpenAI account and key and set the environment variable with your key:
    ```
    export OPENAI_API_KEY="xxxxxx"
    ```

    **Option 2:** No Audio - go through common workplace objects
    
    The server will look for Monitor, Keyboard, Mouse, Laptop, Chair. And keep iterating through the list every time you send a query. You can change this in `slam/services/visualization_service.py`.

## Usage:

### **Python Package Imports**
After installation, you can import modules cleanly from anywhere:

```python
# Core SLAM functionality
from slam.models import SegmentationModel, DetectionModel, CLIPModel
from slam.core.slam_classes import MapObjectList
from slam.utils.vis import vis_result_fast

# Configuration system
from config.settings import Config, get_config

# Example usage
config = get_config()
segmentation_model = SegmentationModel(config)
detection_model = DetectionModel(config)
```

### **Configuration System**
The server uses a centralized YAML-based configuration system:

```bash
# Use default configuration
python server/grpc_server.py --dataset_type replica --useDataset room0

# Use custom configuration file
python server/grpc_server.py --config config/replica_config.yaml --useDataset office1

# List available options
python server/grpc_server.py --help
```

**Configuration files are located in `config/`:**
- `config/defaults.yaml` - Default settings
- `config/replica_config.yaml` - Replica dataset settings  
- `config/ipad_config.yaml` - iPad/live capture settings

### **Performance Logging**
All performance metrics are automatically logged to:
```
logs_performance/<config_name>/<scene_name>/
├── frame_timing.csv          # Detailed per-frame timing data
└── performance_summary.json  # Session statistics and averages
```

**Examples:**
- `logs_performance/default/room0/` - Default config, room0 scene
- `logs_performance/replica_config/office1/` - Custom config, office1 scene

### **Running the Server**

**Basic usage:**
```bash
# Run with specific dataset
python server/grpc_server.py --dataset_type replica --useDataset room0

# Run with custom configuration
python server/grpc_server.py --config config/replica_config.yaml --useDataset office1

# Run with iPad/live input
python server/grpc_server.py --dataset_type ipad --clientUpdateMode --clientIP 192.168.1.100
```

**Key command-line options:**
- `--config <path>` - Use custom YAML configuration file
- `--useDataset <name>` - Process specific dataset (enables scene-based logging)
- `--dataset_type <type>` - Dataset type: `replica`, `ipad`, etc.
- `--clientUpdateMode` - Enable real-time client updates
- `--clientIP <ip>` - Client IP address for updates
- `--pipelined_mapping` - Enable pipelined mapping mode

**Performance monitoring:**
- Real-time FPS and timing statistics logged to console
- Detailed CSV logs for analysis: `logs_performance/<config>/<scene>/frame_timing.csv`
- Graceful shutdown on Ctrl+C with final performance summary

## Configuration Details

### **Environment Variables**
The system supports environment variable overrides for any configuration value:

```bash
# Override model devices
export SLAM_MODEL_DETECTION_DEVICE="cuda:1"
export SLAM_MODEL_SEGMENTATION_DEVICE="cuda:0"

# Override server settings  
export SLAM_SERVER_TARGET_FPS=15
export SLAM_SERVER_PORT=50051

# Run server with overrides
python server/grpc_server.py --useDataset room0
```

### **Model Configuration**
Each AI model can be configured independently:

```yaml
# config/custom_config.yaml
model:
  detection:
    device: "cuda:0"
    enabled: true
    confidence_threshold: 0.35
  segmentation:
    device: "cuda:1" 
    variant: "mobile_sam"
  clip:
    device: "cuda:0"
    model_name: "ViT-H-14"
    batch_size: 8
```

### **Performance Monitoring**
- **Real-time console output:** FPS, frame timing, queue sizes
- **Detailed CSV logs:** Per-frame breakdown of inference times
- **Summary statistics:** Session averages, percentiles, total frames
- **Graceful shutdown:** Ctrl+C writes final performance summary

### **Troubleshooting**

**Import Issues:**
```bash
# Ensure package is installed
pip install -e .

# Test imports
python -c "from slam.models import SegmentationModel; print('✅ Imports working')"
```

**Performance Issues:**
```bash
# Check GPU utilization
nvidia-smi

# Monitor performance logs
tail -f logs_performance/default/room0/frame_timing.csv
```

**Configuration Issues:**
```bash
# Validate configuration
python -c "from config.settings import get_config; print(get_config())"

# Check environment variables
env | grep SLAM_
```

## Licensing

This repository is being migrated to Apache-2.0. The Python package and every dependency declared in `setup.py` is permissive-licensed (BSD-3, MIT, Apache-2.0, HPND, or NVIDIA proprietary runtime SDK). The repo contains:

- **No AGPL/GPL Python deps** (the previously-declared `ultralytics` has been removed).
- **No LGPL Python deps**, and the default code path no longer links into LGPL native libraries (PyAV → PyNvVideoCodec swap; `imageio` is used only for offline PNG reads via Pillow; the optional MP4 animation export is gated behind an actionable error if `imageio-ffmpeg` isn't installed).
- **NVIDIA proprietary runtime SDKs**: CUDA, cuDNN, NCCL, TensorRT, and the NVIDIA Video Codec SDK are required runtime dependencies but are not redistributed by this repository.

For the full dependency inventory and SPDX identifiers, see [`THIRD_PARTY_LICENSES.md`](./THIRD_PARTY_LICENSES.md).
