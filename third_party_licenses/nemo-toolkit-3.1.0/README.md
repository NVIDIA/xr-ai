<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo toolkit 3.1.0 source provenance

STT, Magpie TTS, and speaker STT consume the unmodified upstream NeMo Speech
3.1 release-branch commit `dec3f0d39d869193c00b49a198fe13613be914dc`, which includes the
Hydra/Lightning compatibility update from Speech #16504 via #16578.

- Source: [upstream commit archive](https://github.com/NVIDIA-NeMo/Speech/archive/dec3f0d39d869193c00b49a198fe13613be914dc.tar.gz)
- Archive SHA-256: `452e78957581f9d522f9e4fa7a38e43c8413cdf6582db71fb834253cb7d621ad`
- [LICENSE](LICENSE) SHA-256: `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`

The archive and license were read directly and hashed with SHA-256. The license
is copied byte for byte. NeMo is Apache-2.0; no local source modifications apply.
Default OneLogger registration is disabled upstream, and Neptune logging is
unsupported. Model and checkpoint compatibility require service qualification.
