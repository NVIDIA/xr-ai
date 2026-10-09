<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# XR AI

Agentic AI for XR — an open-source foundation for multimodal, real-time
conversational AI in the NVIDIA CloudXR ecosystem.

XR AI connects web, Android, iOS/visionOS, and native clients to a shared media
hub, GPU-accelerated AI services, tool-using agents, and optional CloudXR remote
rendering. Agents can see and hear what a participant experiences, call native
tools, and return audio or data to the same participant.

This project is a public beta. APIs and behavior may change as it evolves.

## Get started

Refer to the [versioned documentation](https://nvidia.github.io/xr-ai/) for
setup, requirements, credentials, networking, architecture, and
troubleshooting. The landing page opens the newest complete documentation set;
a release takes precedence once it contains the current entry points. Start
with:

- [Set up with a coding agent](https://nvidia.github.io/xr-ai/latest/getting_started/skills.html)
- [Build your own application](https://nvidia.github.io/xr-ai/latest/guides/building-your-app.html)
- [Manual quickstart](https://nvidia.github.io/xr-ai/latest/getting_started/quickstart.html)
- [System requirements](https://nvidia.github.io/xr-ai/latest/getting_started/requirements.html)
- [Architecture](https://nvidia.github.io/xr-ai/latest/overview/architecture.html)

The site also publishes the current `main` branch and release-tagged versions.

## Build your own app

Build an application in the XR AI source tree so its unpublished SDK
dependencies resolve from their repository paths. Start from
`agent-samples/simple-vlm-example/`, copy it to `apps/<your-app>/`, keep only
the SDK layers and services the application needs, and follow the
[build-your-application guide](https://nvidia.github.io/xr-ai/latest/guides/building-your-app.html)
through scaffolding, startup, and hardware-free verification.

## Samples

| Sample | Purpose |
|---|---|
| [`model-servers`](model-server-samples/model-servers/README.md) | Start and persist the shared model stack |
| [`model-servers-nim`](model-server-samples/model-servers-nim/README.md) | Start the shared model stack using NVIDIA NIM |
| [`simple-vlm-example`](agent-samples/simple-vlm-example/README.md) | Voice and text questions about the current camera frame |
| [`lab-instrument-monitoring`](agent-samples/lab-instrument-monitoring/README.md) | Marker-associated visual monitoring with a foreground voice agent |
| [`tea-making-sample`](agent-samples/tea-making-sample/README.md) | Guided workflow with visual evidence and background observations |
| [`xr-render-demo`](agent-samples/xr-render-demo/README.md) | Voice-driven CloudXR scene manipulation |

Each sample README gives the shortest runnable command sequence. The linked
documentation contains architecture, behavior, configuration, output contracts,
evaluation, and adaptation guidance.

## Repository map

| Directory | Contents |
|---|---|
| `client-samples/` | Platform clients and shared StreamKit implementations |
| `agent-sdk/` | Hub IPC, model clients, runtime, tools, voice, and web events |
| `agent-samples/` | Runnable agent stacks |
| `model-server-samples/` | Shared model-server launch samples |
| `apps/` | Application workspaces outside repository sample tooling |
| `services/` | Hub, model servers, and typed capability services |
| `utils/` | Launcher, logging, VAD, vLLM, and voice-gate utilities |
| `tests/` | Cross-package and integration tests |
| `docs/source/` | Canonical user and contributor documentation |
| `skills/` | Setup skills for coding agents |

For repository constraints and dependency boundaries, refer to
[`AGENTS.md`](AGENTS.md) and [`DEPENDENCIES.md`](DEPENDENCIES.md).

<!-- Compatibility anchors for headings consolidated into the documentation. -->
<a id="public-beta-notice"></a><a id="what-is-xr-ai"></a>
<a id="requirements"></a><a id="architecture"></a><a id="quickstart"></a>
<a id="model-servers-shared-ai-services"></a>
<a id="simple-vlm-example-vision-qa-over-voice--text"></a>
<a id="step-1--start-the-server"></a><a id="step-2--connect-a-client"></a>
<a id="lab-instrument-monitoring-marker-associated-readings--foreground-voice"></a>
<a id="xr-render-demo-voice-driven-sphere-in-cloudxr"></a>
<a id="step-1--start-model-servers-once"></a><a id="step-2--start-the-demo"></a>
<a id="hub-only-standalone"></a><a id="clients"></a><a id="web"></a>
<a id="android"></a><a id="ios-and-visionos"></a><a id="networking"></a>
<a id="tests"></a><a id="deeper-docs"></a><a id="project-meta"></a>
<a id="ios--visionos"></a>

## Contributing

Refer to [`CONTRIBUTING.md`](CONTRIBUTING.md) for the contribution process and
[`tests/README.md`](tests/README.md) for the shortest test commands. Refer to
[`SECURITY.md`](SECURITY.md) to report security issues.

## License and third-party disclosures

XR AI is licensed under [Apache-2.0](LICENSE). The table below discloses the
components, license terms, and all 60 retained license files under
`third_party_licenses/`, including notices for bundled libraries, fonts, and
other upstream assets. Component-specific terms apply to those components;
the retained upstream texts are unchanged.

Refer to [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for dependency
usage, corresponding source, additional attributions, and license elections.
The table records upstream alternatives where present; XR AI's license
elections are documented in the notices.

| Component | License terms | Retained license file |
|---|---|---|
| certifi-2026.7.22 | MPL-2.0 | [`LICENSE`](third_party_licenses/certifi-2026.7.22/LICENSE) |
| flashinfer-python-0.6.18.post1 | Apache-2.0 | [`LICENSE`](third_party_licenses/flashinfer-python-0.6.18.post1/LICENSE) |
| huggingface-hub-1.32.0 | Apache-2.0 | [`LICENSE`](third_party_licenses/huggingface-hub-1.32.0/LICENSE) |
| humming-kernels-0.1.12 | Apache-2.0 | [`LICENSE`](third_party_licenses/humming-kernels-0.1.12/LICENSE) |
| instanttensor-0.2.0 | Apache-2.0 | [`LICENSE`](third_party_licenses/instanttensor-0.2.0/LICENSE) |
| libsndfile-1.2.2 | LGPL-2.1-or-later | [`COPYING`](third_party_licenses/libsndfile-1.2.2/COPYING) |
| llvmlite-0.47.0 | BSD-2-Clause | [`LICENSE`](third_party_licenses/llvmlite-0.47.0/LICENSE) |
| LLVM code in llvmlite 0.47.0 | Apache-2.0 WITH LLVM-exception | [`LICENSE.thirdparty`](third_party_licenses/llvmlite-0.47.0/LICENSE.thirdparty) |
| FreeType (Matplotlib 3.11.1) | FTL | [`FTL.TXT`](third_party_licenses/matplotlib-3.11.1/FTL.TXT) |
| Matplotlib 3.11.1 | Matplotlib license agreement (PSF-based) | [`LICENSE`](third_party_licenses/matplotlib-3.11.1/LICENSE) |
| AMS fonts (Matplotlib 3.11.1) | OFL-1.1 | [`LICENSE_AMSFONTS`](third_party_licenses/matplotlib-3.11.1/LICENSE_AMSFONTS) |
| BaKoMa fonts (Matplotlib 3.11.1) | BaKoMa Fonts Licence | [`LICENSE_BAKOMA`](third_party_licenses/matplotlib-3.11.1/LICENSE_BAKOMA) |
| ColorBrewer (Matplotlib 3.11.1) | Apache-2.0 | [`LICENSE_COLORBREWER`](third_party_licenses/matplotlib-3.11.1/LICENSE_COLORBREWER) |
| Courier 10 Pitch fonts (Matplotlib 3.11.1) | Bitstream Charter and Courier font permission terms | [`LICENSE_COURIERTEN`](third_party_licenses/matplotlib-3.11.1/LICENSE_COURIERTEN) |
| DejaVu fonts (Matplotlib 3.11.1) | Bitstream Vera and Arev font terms; DejaVu changes public domain | [`LICENSE_DEJAVU`](third_party_licenses/matplotlib-3.11.1/LICENSE_DEJAVU) |
| Font Awesome SVG icons (Matplotlib 3.11.1) | CC-BY-4.0 | [`LICENSE_FONT_AWESOME`](third_party_licenses/matplotlib-3.11.1/LICENSE_FONT_AWESOME) |
| FreeType license alternatives (Matplotlib 3.11.1) | FTL (elected; GPL alternative retained in upstream notice) | [`LICENSE_FREETYPE`](third_party_licenses/matplotlib-3.11.1/LICENSE_FREETYPE) |
| HarfBuzz (Matplotlib 3.11.1) | Old MIT (permission and disclaimer terms) | [`LICENSE_HARFBUZZ`](third_party_licenses/matplotlib-3.11.1/LICENSE_HARFBUZZ) |
| JSXTools Resize Observer (Matplotlib 3.11.1) | CC0-1.0 | [`LICENSE_JSXTOOLS_RESIZE_OBSERVER`](third_party_licenses/matplotlib-3.11.1/LICENSE_JSXTOOLS_RESIZE_OBSERVER) |
| Last Resort font (Matplotlib 3.11.1) | OFL-1.1 | [`LICENSE_LAST_RESORT_FONT`](third_party_licenses/matplotlib-3.11.1/LICENSE_LAST_RESORT_FONT) |
| libraqm (Matplotlib 3.11.1) | MIT | [`LICENSE_LIBRAQM`](third_party_licenses/matplotlib-3.11.1/LICENSE_LIBRAQM) |
| formlayout Qt editor (Matplotlib 3.11.1) | MIT | [`LICENSE_QT4_EDITOR`](third_party_licenses/matplotlib-3.11.1/LICENSE_QT4_EDITOR) |
| SheenBidi (Matplotlib 3.11.1) | Apache-2.0 | [`LICENSE_SHEENBIDI`](third_party_licenses/matplotlib-3.11.1/LICENSE_SHEENBIDI) |
| Solarized colors (Matplotlib 3.11.1) | MIT | [`LICENSE_SOLARIZED`](third_party_licenses/matplotlib-3.11.1/LICENSE_SOLARIZED) |
| STIX fonts (Matplotlib 3.11.1) | OFL-1.1 | [`LICENSE_STIX`](third_party_licenses/matplotlib-3.11.1/LICENSE_STIX) |
| Gist and Yorick colormaps (Matplotlib 3.11.1) | UC LLNL BSD-style permission terms | [`LICENSE_YORICK`](third_party_licenses/matplotlib-3.11.1/LICENSE_YORICK) |
| Anti-Grain Geometry 2.4 (Matplotlib 3.11.1) | Anti-Grain Geometry permission terms | [`copying`](third_party_licenses/matplotlib-3.11.1/copying) |
| num2words-0.5.14 | LGPL-2.1-or-later | [`COPYING`](third_party_licenses/num2words-0.5.14/COPYING) |
| nvidia-cutlass-dsl-4.7.1 | NVIDIA Software License Agreement (CUTLASS DSLs) | [`LICENSE`](third_party_licenses/nvidia-cutlass-dsl-4.7.1/LICENSE) |
| nvtx-0.2.15 | Apache-2.0 WITH LLVM-exception | [`LICENSE.txt`](third_party_licenses/nvtx-0.2.15/LICENSE.txt) |
| OpenCV 5.0.0.93 wheel bundled components, including FFmpeg | LGPL-2.1-or-later (FFmpeg) and component-specific terms in the notice | [`LICENSE-3RD-PARTY.txt`](third_party_licenses/opencv-python-headless-5.0.0.93/LICENSE-3RD-PARTY.txt) |
| opencv-python-headless-5.0.0.93 | MIT (Python packaging) | [`LICENSE.txt`](third_party_licenses/opencv-python-headless-5.0.0.93/LICENSE.txt) |
| protobuf-7.36.0 | BSD-3-Clause | [`LICENSE`](third_party_licenses/protobuf-7.36.0/LICENSE) |
| pycountry-26.2.16 | LGPL-2.1-only | [`LICENSE.txt`](third_party_licenses/pycountry-26.2.16/LICENSE.txt) |
| pyjwt-2.15.1 | MIT | [`LICENSE`](third_party_licenses/pyjwt-2.15.1/LICENSE) |
| pyzmq-27.2.0 | BSD-3-Clause | [`LICENSE`](third_party_licenses/pyzmq-27.2.0/LICENSE) |
| libsodium 1.0.22 in PyZMQ 27.2.0 | ISC | [`LICENSE.libsodium`](third_party_licenses/pyzmq-27.2.0/LICENSE.libsodium) |
| Tornado code in PyZMQ 27.2.0 | Apache-2.0 | [`LICENSE.tornado`](third_party_licenses/pyzmq-27.2.0/LICENSE.tornado) |
| libzmq 4.3.5 in PyZMQ 27.2.0 | MPL-2.0 | [`LICENSE.zeromq`](third_party_licenses/pyzmq-27.2.0/LICENSE.zeromq) |
| quack-kernels-0.6.5 | Apache-2.0 | [`LICENSE`](third_party_licenses/quack-kernels-0.6.5/LICENSE) |
| regex-2026.7.19 | Apache-2.0 and CNRI Python 1.6 terms | [`LICENSE.txt`](third_party_licenses/regex-2026.7.19/LICENSE.txt) |
| soundfile-0.14.0 | BSD-3-Clause | [`LICENSE`](third_party_licenses/soundfile-0.14.0/LICENSE) |
| Python-SoXR 1.0.0 and libsoxr 0.1.3 | LGPL-2.1-or-later | [`COPYING.LGPL`](third_party_licenses/soxr-1.0.0/COPYING.LGPL) |
| soxr-1.0.0 | LGPL-2.1-or-later | [`LICENSE`](third_party_licenses/soxr-1.0.0/LICENSE) |
| PFFFT in Python-SoXR 1.0.0 | FFTPACK permissive license | [`LICENSE-PFFFT`](third_party_licenses/soxr-1.0.0/LICENSE-PFFFT) |
| libsoxr 0.1.3 in Python-SoXR 1.0.0 | LGPL-2.1-or-later | [`LICENSE-libsoxr`](third_party_licenses/soxr-1.0.0/LICENSE-libsoxr) |
| supervisor-4.3.0 | BSD-derived Supervisor terms, BSD-3-Clause, and Medusa permission terms | [`LICENSES.txt`](third_party_licenses/supervisor-4.3.0/LICENSES.txt) |
| text-unidecode-1.3 | Artistic-1.0-Perl (elected; GPL alternative retained in upstream notice) | [`LICENSE`](third_party_licenses/text-unidecode-1.3/LICENSE) |
| dav1d in TorchCodec 0.16.0 | BSD-2-Clause | [`COPYING.dav1d`](third_party_licenses/torchcodec-0.16.0/COPYING.dav1d) |
| torchcodec-0.16.0 | BSD-3-Clause | [`LICENSE`](third_party_licenses/torchcodec-0.16.0/LICENSE) |
| libavif in TorchCodec 0.16.0 | BSD-2-Clause | [`LICENSE.libavif`](third_party_licenses/torchcodec-0.16.0/LICENSE.libavif) |
| libjpeg-turbo in TorchCodec 0.16.0 | IJG, BSD-3-Clause, and zlib terms | [`LICENSE.libjpeg-turbo`](third_party_licenses/torchcodec-0.16.0/LICENSE.libjpeg-turbo) |
| libnvjpeg in TorchCodec 0.16.0 | NVIDIA CUDA Toolkit EULA | [`LICENSE.libnvjpeg-NVIDIA-CUDA-EULA.txt`](third_party_licenses/torchcodec-0.16.0/LICENSE.libnvjpeg-NVIDIA-CUDA-EULA.txt) |
| libpng in TorchCodec 0.16.0 | PNG Reference Library License version 2 | [`LICENSE.libpng`](third_party_licenses/torchcodec-0.16.0/LICENSE.libpng) |
| libwebp in TorchCodec 0.16.0 | BSD-3-Clause | [`LICENSE.libwebp`](third_party_licenses/torchcodec-0.16.0/LICENSE.libwebp) |
| libyuv in TorchCodec 0.16.0 | BSD-3-Clause | [`LICENSE.libyuv`](third_party_licenses/torchcodec-0.16.0/LICENSE.libyuv) |
| zlib in TorchCodec 0.16.0 | Zlib | [`LICENSE.zlib`](third_party_licenses/torchcodec-0.16.0/LICENSE.zlib) |
| tqdm-4.70.0 | MPL-2.0 AND MIT | [`LICENCE`](third_party_licenses/tqdm-4.70.0/LICENCE) |
| urllib3-2.8.0 | MIT | [`LICENSE.txt`](third_party_licenses/urllib3-2.8.0/LICENSE.txt) |
| vllm-0.30.0 | Apache-2.0 | [`LICENSE`](third_party_licenses/vllm-0.30.0/LICENSE) |
