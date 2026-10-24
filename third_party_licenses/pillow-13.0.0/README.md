<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# pillow 13.0.0 license provenance

Unmodified license and attribution texts copied from SHA-256-verified upstream
artifacts:

- Artifact: [pillow-13.0.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl](https://files.pythonhosted.org/packages/e6/99/fdf65cc539e266ab93546c261bdf98411fe165c6e4e13f9dbf597097192b/pillow-13.0.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl)
- SHA-256: `42d63be3af4f9b951b199b0a9d23365d37ab14687e6d77b74940432c8515c4ca`
- Artifact: [pillow-13.0.0-cp312-cp312-macosx_11_0_arm64.whl](https://files.pythonhosted.org/packages/39/4b/f2ed53d83aa44ac19599ac0a54897f867fafb95f406980d1e24083f8cbe3/pillow-13.0.0-cp312-cp312-macosx_11_0_arm64.whl)
- SHA-256: `fc481f742ccda20e9ed760b8f978dff50f2a4e9bc253ac167fb0a05b0783ef52`

| Copied file | Artifact member | SHA-256 |
|---|---|---|
| [LICENSE](LICENSE) | `pillow-13.0.0.dist-info/licenses/LICENSE` (both artifacts) | `dda12a98c1979cf3d94df1cff45d27a4cb3f04a60c76f76902ac54cac03ec0ce` |

Pillow uses MIT-CMU. The complete license file contains bundled native-library
notices and is identical in the inspected macOS arm64 and Linux x86_64 and
aarch64 wheels. It is retained in full for redistributors.

For FreeType, XR AI elects the **FTL** option, including its acknowledgment:
portions of this software are copyright © The FreeType Project
(www.freetype.org). All rights reserved.

The inspected CPython 3.12 official macOS arm64 and Linux x86_64 and aarch64
wheels do not bundle libimagequant or FriBiDi. Their optional GPL and LGPL
entries in Pillow's SBOM do not describe libraries included in these inspected
wheels. Custom builds enabling either library require a separate license review.
The Pillow-authored `fribidi-shim` is MIT-CMU.

Linux aarch64 artifact: `pillow-13.0.0-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl`, SHA-256 `8bf295fe459343bc6093c8e5e0df60bfa189e0fcb775a22e2c2488fdeccc6c0d`.
