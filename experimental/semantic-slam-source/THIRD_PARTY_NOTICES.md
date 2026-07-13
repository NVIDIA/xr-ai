<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Third-Party Notices — semantic-SLAM source snapshot

This directory is a legacy semantic-SLAM source snapshot retained for
evaluation and migration reference. It is governed by the repository-root
Apache-2.0 license. It is not the XR AI MCP integration and it does not include
model weights, third-party model repositories, CUDA, TensorRT, or a Python
environment.

This inventory reconciles the components shared with
`feat/semantic-slam-module` and adds the legacy-only dependency surface. The
listed source licenses do not determine the terms of model checkpoints or
runtime SDKs; review those artifacts separately before deployment.

## Direct Python dependencies

These constraints are declared by this snapshot's `pyproject.toml`; `setup.py`
also declares `torchaudio>=2.0.2` for its older setuptools path.

| Package | Manifest constraint | License | Upstream |
|---|---|---|---|
| `torch` | `==2.0.1` | BSD-3-Clause | https://github.com/pytorch/pytorch |
| `torchvision` | `==0.15.2` | BSD-3-Clause | https://github.com/pytorch/vision |
| `torchaudio` | `>=2.0.2` (`setup.py` only) | BSD-3-Clause | https://github.com/pytorch/audio |
| `numpy` | `>=1.24.3,<2` | BSD-3-Clause; see binary-wheel note | https://github.com/numpy/numpy |
| `opencv-python` | `>=4.9.0,<4.11.0` | Apache-2.0 | https://github.com/opencv/opencv-python |
| `Pillow` | `>=9.5.0` | HPND | https://github.com/python-pillow/Pillow |
| `grpcio`, `grpcio-tools` | `>=1.65.0` | Apache-2.0 | https://github.com/grpc/grpc |
| `supervision` | `>=0.14.0` | MIT | https://github.com/roboflow/supervision |
| `pynvvideocodec` | `>=1.0.0` | MIT; see runtime note | https://developer.nvidia.com/pynvvideocodec |
| `distinctipy` | `>=1.3.0` | MIT | https://github.com/alan-turing-institute/distinctipy |
| `open-clip-torch` | `>=2.26.0` | MIT | https://github.com/mlfoundations/open_clip |
| `transformers` | `>=4.25.1,<4.26.0` | Apache-2.0 | https://github.com/huggingface/transformers |
| `openai` | `>=1.37.0` | Apache-2.0 | https://github.com/openai/openai-python |
| `hydra-core` | `>=1.3.0` | MIT | https://github.com/facebookresearch/hydra |
| `omegaconf` | `>=2.3.0` | BSD-3-Clause | https://github.com/omry/omegaconf |
| `wandb` | `>=0.17.0` | MIT | https://github.com/wandb/wandb |
| `h5py` | `>=3.11.0` | BSD-3-Clause | https://github.com/h5py/h5py |
| `tyro` | `>=0.8.0` | MIT | https://github.com/brentyi/tyro |
| `tqdm` | `>=4.65.0` | MPL-2.0 OR MIT | https://github.com/tqdm/tqdm |
| `matplotlib` | `>=3.9.0` | PSF / BSD-style | https://github.com/matplotlib/matplotlib |
| `scikit-learn` | `>=1.5.0` | BSD-3-Clause | https://github.com/scikit-learn/scikit-learn |
| `scipy` | `>=1.10.0,<1.15.0` | BSD-3-Clause | https://github.com/scipy/scipy |
| `imageio` | `>=2.34.0` | BSD-2-Clause | https://github.com/imageio/imageio |
| `kornia` | `==0.7.3` | Apache-2.0 | https://github.com/kornia/kornia |
| `natsort` | `>=8.4.0` | MIT | https://github.com/SethMMorton/natsort |
| `open3d` | `>=0.18.0` | MIT | https://github.com/isl-org/Open3D |
| `faiss-cpu` | `>=1.7.4` | MIT | https://github.com/facebookresearch/faiss |
| `ftfy` | `>=6.2.0` | Apache-2.0 | https://github.com/rspeer/python-ftfy |
| `fairscale` | `>=0.4.4` | BSD-3-Clause | https://github.com/facebookresearch/fairscale |
| `addict` | `>=2.4.0` | MIT | https://github.com/mewwts/addict |
| `pyyaml` | `>=6.0` | MIT | https://github.com/yaml/pyyaml |
| `fvcore` | `>=0.1.5` | Apache-2.0 | https://github.com/facebookresearch/fvcore |
| `iopath` | `>=0.1.9` | MIT | https://github.com/facebookresearch/iopath |
| `timm` | unpinned | Apache-2.0 | https://github.com/huggingface/pytorch-image-models |
| `yapf` | unpinned | Apache-2.0 | https://github.com/google/yapf |
| `pycocotools` | unpinned | BSD-2-Clause | https://github.com/cocodataset/cocoapi |

## Externally provisioned model components

The source code and setup script expect these components outside this
repository. Versions are intentionally shown as the legacy script specifies
them; an unpinned URL is not a reproducible deployment specification.

| Project | Version or reference | License | Upstream |
|---|---|---|---|
| chamferdist | unpinned PyPI install | MIT | https://github.com/krrish94/chamferdist |
| PyTorch3D | `v0.7.7` | BSD-3-Clause | https://github.com/facebookresearch/pytorch3d |
| gradSLAM | `conceptfusion` branch | MIT | https://github.com/gradslam/gradslam |
| Segment Anything (SAM) | unpinned Git install | Apache-2.0 | https://github.com/facebookresearch/segment-anything |
| GroundingDINO | unpinned Git install | Apache-2.0 | https://github.com/IDEA-Research/GroundingDINO |
| Recognize Anything Model (RAM / Tag2Text) | unpinned Git install | Apache-2.0 | https://github.com/xinyu1205/recognize-anything |
| MobileSAM | source-supported optional variant | Apache-2.0 | https://github.com/ChaoningZhang/MobileSAM |
| SAM-HQ | source-supported optional variant | Apache-2.0 | https://github.com/SysCV/sam-hq |

## Source attributions and runtime notes

- Portions of the semantic-map implementation derive from ConceptGraphs,
  Copyright 2023 concept-graphs, under the MIT License:
  https://github.com/concept-graphs/concept-graphs
- `slam/utils/vis.py` contains the `align_vector_to_another` helper derived
  from Open3D material, under the MIT License.
- `chamferdist` retains a second MIT attribution for AtlasNet in addition to
  its Krishna Murthy copyright notice.
- NumPy Linux wheels can bundle OpenBLAS and a GCC runtime component. Their
  package metadata includes a GPL-3.0-with-GCC-exception notice for that
  runtime; this source tree does not redistribute a wheel. Review the binary
  notice when building a runtime image.
- CUDA Toolkit and TensorRT are separately provisioned NVIDIA SDKs. They are
  not redistributed by this source snapshot. The optional TensorRT import path
  requires a separately reviewed runtime installation.
- PyNvVideoCodec is MIT-licensed Python software but uses NVIDIA Video Codec
  SDK APIs at runtime; the SDK and driver terms are separate from this package.
- Model checkpoints, including SAM, GroundingDINO, RAM, MobileSAM, and SAM-HQ
  weights, are not present in this repository and require separate terms
  review before deployment.
- `sceneGraphDemo_test.yml` is a historical Conda environment capture, not a
  redistributable binary environment. Its CUDA and system-library entries must
  be reviewed if a runtime image is assembled from it.
