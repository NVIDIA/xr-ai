<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# pyopenxr 1.1.5301 license provenance

Unmodified license and attribution texts copied from the upstream artifact and
from the upstream source tag matching the bundled native library:

- Artifact: [pyopenxr-1.1.5301-py3-none-any.whl](https://files.pythonhosted.org/packages/a8/0f/3703e541dc6e4163b8ea1c5a562cfb5c32515ffb1bb0f9f24d4c7db90c19/pyopenxr-1.1.5301-py3-none-any.whl)
- SHA-256: `83da1ea6534e55cf5e88860870fa7cc48ec977a58bf173b3d4bfbc273926e3b2`

The wheel bundles `xr/library/libopenxr_loader.so`, built from
KhronosGroup/OpenXR-SDK-Source at OpenXR 1.1.53, which statically links
jsoncpp (the binary contains `Json::` symbols and
`src/external/jsoncpp/src/lib_json/*.cpp` source paths). The OpenXR SDK and
jsoncpp licenses are copied from the `release-1.1.53` tag. The wheel's Linux
API layers (`xr/api_layer/linux/libXrApiLayer_*.so`) are built from the same
OpenXR SDK sources and do not contain jsoncpp:

- [LICENSE](https://raw.githubusercontent.com/KhronosGroup/OpenXR-SDK-Source/release-1.1.53/LICENSE)
- [src/external/jsoncpp/LICENSE](https://raw.githubusercontent.com/KhronosGroup/OpenXR-SDK-Source/release-1.1.53/src/external/jsoncpp/LICENSE)

| Copied file | SHA-256 |
|---|---|
| [LICENSE](LICENSE) (wheel, `pyopenxr-1.1.5301.dist-info/licenses/LICENSE`, Apache-2.0) | `037ac52db97cad860289acc754685d3fdf6085d620a8c15c370fe03dfdc510f3` |
| [LICENSE.openxr-sdk](LICENSE.openxr-sdk) (OpenXR-SDK-Source `LICENSE`) | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| [LICENSE.jsoncpp](LICENSE.jsoncpp) (OpenXR-SDK-Source `src/external/jsoncpp/LICENSE`) | `cec0db5f6d7ed6b3a72647bd50aed02e13c3377fd44382b96dc2915534c042ad` |
