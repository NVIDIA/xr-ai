<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo compatibility source license provenance

The services consume Apache-2.0 NeMo source with a narrow Lightning 2.6
Neptune import compatibility change and a regression test. Inference model
implementations and model revisions retain their upstream definitions.

| Service source | Upstream base | Patched revision |
|---|---|---|
| STT and Magpie TTS | [`fd6a877539710e2b98f28c43272ff81312f83417`](https://github.com/NVIDIA-NeMo/Speech/tree/fd6a877539710e2b98f28c43272ff81312f83417) | [`f047d87f41761e55bf2baf61967cbc687eeb1754`](https://github.com/yanziz-nvidia/Speech/tree/f047d87f41761e55bf2baf61967cbc687eeb1754) |
| Speaker STT | [`1688cc3d6a9ade854f544987810c53f605dc86fc`](https://github.com/NVIDIA-NeMo/Speech/tree/1688cc3d6a9ade854f544987810c53f605dc86fc) | [`d7939d46c647d64db180e9f291e058855cd75d3a`](https://github.com/yanziz-nvidia/Speech/tree/d7939d46c647d64db180e9f291e058855cd75d3a) |

The following upstream license files are copied byte for byte. They contain
the same Apache-2.0 terms and differ in their final newline.

| Copied file | SHA-256 |
|---|---|
| [LICENSE.3.0.0](LICENSE.3.0.0) | `43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1` |
| [LICENSE.speaker](LICENSE.speaker) | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |
