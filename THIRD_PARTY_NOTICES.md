<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Third-Party Notices

This file lists the third-party open-source software distributed with or
required by `xr-ai`. Each entry gives the upstream project, version, SPDX
license identifier, and a link to the upstream source repository, where the
canonical license text is available.

`xr-ai` itself is licensed under Apache-2.0 — see [`LICENSE`](LICENSE).

## Python (server-side and AI services)

Used by `server-runtime/`, `agent-sdk/`, `utils/`, `ai-services/`,
`agent-mcp-servers/`, `agent-samples/`, `cloudxr-runtime/`, and `tests/`.
For the per-package dependency mapping, see [`DEPENDENCIES.md`](DEPENDENCIES.md).

| Package        | Version  | License       | Upstream |
|---             |---       |---            |---|
| `msgpack`      | 1.0.0    | Apache-2.0    | https://github.com/msgpack/msgpack-python |
| `pyzmq`        | 27.0.0   | BSD-3-Clause  | https://github.com/zeromq/pyzmq |
| `uvicorn`      | 0.29.0   | BSD-3-Clause  | https://github.com/encode/uvicorn |
| `fastapi`      | 0.111.0  | MIT           | https://github.com/fastapi/fastapi |
| `httpx`        | 0.27.0   | BSD-3-Clause  | https://github.com/encode/httpx |
| `livekit`      | 0.17.0   | Apache-2.0    | https://github.com/livekit/python-sdks |
| `livekit-api`  | 0.7.0    | Apache-2.0    | https://github.com/livekit/python-sdks |
| `numpy`        | 1.24.0   | BSD-3-Clause  | https://github.com/numpy/numpy |
| `websockets`   | 12.0     | BSD-3-Clause  | https://github.com/python-websockets/websockets |

## Python — `ai-services/semantic-slam`

Used by `ai-services/semantic-slam/`. Pinned versions from `uv.lock` in
[`semantic-slam-server`](https://github.com/nvddr/semantic-slam-server).

### PyPI packages

| Package              | Version          | License            | Upstream |
|---                   |---               |---                 |---|
| `torch`              | 2.0.1+cu118      | BSD-3-Clause       | https://github.com/pytorch/pytorch |
| `torchvision`        | 0.15.2+cu118     | BSD-3-Clause       | https://github.com/pytorch/vision |
| `numpy`              | 1.26.4           | BSD-3-Clause       | https://github.com/numpy/numpy |
| `opencv-python`      | 4.10.0.84        | Apache-2.0         | https://github.com/opencv/opencv-python |
| `Pillow`             | 12.2.0           | HPND               | https://github.com/python-pillow/Pillow |
| `grpcio`             | 1.81.0           | Apache-2.0         | https://github.com/grpc/grpc |
| `grpcio-tools`       | 1.81.0           | Apache-2.0         | https://github.com/grpc/grpc |
| `supervision`        | 0.28.0           | MIT                | https://github.com/roboflow/supervision |
| `pynvvideocodec`     | 2.1.0            | MIT                | https://github.com/NVIDIA/PyNvVideoCodec |
| `distinctipy`        | 1.3.4            | MIT                | https://github.com/alan-turing-institute/distinctipy |
| `open-clip-torch`    | 3.3.0            | MIT                | https://github.com/mlfoundations/open_clip |
| `transformers`       | 4.25.1           | Apache-2.0         | https://github.com/huggingface/transformers |
| `openai`             | 2.41.0           | Apache-2.0         | https://github.com/openai/openai-python |
| `hydra-core`         | 1.3.2            | MIT                | https://github.com/facebookresearch/hydra |
| `omegaconf`          | 2.3.0            | BSD-3-Clause       | https://github.com/omry/omegaconf |
| `wandb`              | 0.27.2           | MIT                | https://github.com/wandb/wandb |
| `h5py`               | 3.16.0           | BSD-3-Clause       | https://github.com/h5py/h5py |
| `tyro`               | 1.0.13           | MIT                | https://github.com/brentyi/tyro |
| `tqdm`               | 4.68.2           | MIT / MPL-2.0      | https://github.com/tqdm/tqdm |
| `matplotlib`         | 3.10.9           | PSF / BSD-style    | https://github.com/matplotlib/matplotlib |
| `scikit-learn`       | 1.7.2            | BSD-3-Clause       | https://github.com/scikit-learn/scikit-learn |
| `scipy`              | 1.14.1           | BSD-3-Clause       | https://github.com/scipy/scipy |
| `imageio`            | 2.37.3           | BSD-2-Clause       | https://github.com/imageio/imageio |
| `kornia`             | 0.7.3            | Apache-2.0         | https://github.com/kornia/kornia |
| `natsort`            | 8.4.0            | MIT                | https://github.com/SethMMorton/natsort |
| `open3d`             | 0.19.0           | MIT                | https://github.com/isl-org/Open3D |
| `faiss-cpu`          | 1.14.2           | MIT                | https://github.com/facebookresearch/faiss |
| `ftfy`               | 6.3.1            | Apache-2.0         | https://github.com/rspeer/python-ftfy |
| `fairscale`          | 0.4.13           | BSD-3-Clause       | https://github.com/facebookresearch/fairscale |
| `addict`             | 2.4.0            | MIT                | https://github.com/mewwts/addict |
| `pyyaml`             | 6.0.3            | MIT                | https://github.com/yaml/pyyaml |
| `fvcore`             | 0.1.5            | Apache-2.0         | https://github.com/facebookresearch/fvcore |
| `iopath`             | 0.1.10           | MIT                | https://github.com/facebookresearch/iopath |
| `timm`               | 1.0.27           | Apache-2.0         | https://github.com/huggingface/pytorch-image-models |
| `yapf`               | 0.43.0           | Apache-2.0         | https://github.com/google/yapf |
| `pycocotools`        | 2.0.11           | BSD-2-Clause       | https://github.com/cocodataset/cocoapi |

### Git-installed model components

| Project                  | License     | Upstream |
|---                       |---          |---|
| GroundingDINO            | Apache-2.0  | https://github.com/IDEA-Research/GroundingDINO |
| Segment Anything (SAM)   | Apache-2.0  | https://github.com/facebookresearch/segment-anything |
| MobileSAM                | Apache-2.0  | https://github.com/ChaoningZhang/MobileSAM |
| sam-hq / LightHQSAM      | Apache-2.0  | https://github.com/SysCV/sam-hq |
| EfficientSAM             | Apache-2.0  | https://github.com/yformer/EfficientSAM |
| recognize-anything (RAM) | Apache-2.0  | https://github.com/xinyu1205/recognize-anything |
| gradslam (conceptfusion) | MIT         | https://github.com/gradslam/gradslam |
| chamferdist              | MIT         | https://github.com/krrish94/chamferdist |
| pytorch3d                | BSD-3-Clause | https://github.com/facebookresearch/pytorch3d |

### NVIDIA proprietary runtime SDKs (runtime-only, not redistributed)

| SDK                    | Purpose |
|---                     |---|
| CUDA Toolkit           | GPU compute runtime |
| cuDNN                  | Deep-learning primitives |
| NCCL                   | Multi-GPU collectives |
| TensorRT               | Inference optimizer (optional CLIP path) |
| NVIDIA Video Codec SDK | Hardware H.264/H.265 decode (via `pynvvideocodec`) |

## Swift (iOS / visionOS client)

Used by `client-samples/ios-visionos/`. Resolved via Swift Package Manager.

| Package | Version | License | Upstream |
|---|---|---|---|
| `LiveKitClient` (`livekit/client-sdk-swift`)            | 2.13.0       | Apache-2.0   | https://github.com/livekit/client-sdk-swift |
| `livekit/webrtc-xcframework`                            | 144.7559.01  | MIT          | https://github.com/livekit/webrtc-xcframework |
| `livekit/livekit-uniffi-xcframework`                    | 0.0.5        | Apache-2.0   | https://github.com/livekit/livekit-uniffi-xcframework |
| `swift-protobuf` (`apple/swift-protobuf`)               | 1.36.1       | Apache-2.0   | https://github.com/apple/swift-protobuf |

## License texts

The full text of each SPDX license identifier referenced above is available at:

- **Apache-2.0**: https://www.apache.org/licenses/LICENSE-2.0 — also bundled
  with this repository as [`LICENSE`](LICENSE).
- **BSD-2-Clause**: https://opensource.org/license/bsd-2-clause
- **BSD-3-Clause**: https://opensource.org/license/bsd-3-clause
- **HPND**: https://opensource.org/license/historical-permission-notice-and-disclaimer (Pillow / PIL)
- **MIT**: https://opensource.org/license/mit
- **MPL-2.0**: https://www.mozilla.org/en-US/MPL/2.0/ (tqdm dual-license)
- **PSF**: https://docs.python.org/3/license.html (matplotlib)

Each upstream project repository linked above includes its own canonical
license file (typically `LICENSE`, `LICENSE.txt`, or `COPYING`).

## Updating this file

When adding, removing, or upgrading a third-party dependency:

1. Update the relevant manifest — `pyproject.toml` (Python),
   `Package.swift` (Swift), `gradle/libs.versions.toml` (Android), or the
   web client's manifest.
2. Update [`DEPENDENCIES.md`](DEPENDENCIES.md) — the internal/external
   dependency map.
3. Update this file with the new package name, version, license, and upstream
   URL.

All three changes belong in the same commit.
