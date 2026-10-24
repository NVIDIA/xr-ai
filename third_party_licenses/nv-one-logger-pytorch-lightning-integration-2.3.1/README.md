<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OneLogger PyTorch Lightning integration 2.3.1 license provenance

The inference services use Apache-2.0 source from NVIDIA OneLogger at upstream
revision `c382dd5193ae987233c7a61de33fb16f91724b7d`, with compatibility and packaging changes pinned at
[`665328ee4697b1c488ef58d4597dfad34d33f759`](https://github.com/yanziz-nvidia/nv-one-logger/tree/665328ee4697b1c488ef58d4597dfad34d33f759).

The fork broadens `save_checkpoint.weights_only` to `Optional[bool]`, preserves
signature validation and forwarding behavior, corrects the Poetry URL table,
and adds checkpoint-policy regression tests. Modified files carry change notices.

The upstream `LICENSE` is a short license notice. The complete Apache-2.0 terms
are retained in the [repository license](../../LICENSE).

The bundled `LICENSE` and repository `NOTICE.md` are unmodified. The latter
describes the full upstream project, including optional integrations; its
third-party license references are relative to the
[upstream repository](https://github.com/NVIDIA/nv-one-logger/tree/c382dd5193ae987233c7a61de33fb16f91724b7d/third_party_licenses).

| Copied file | SHA-256 |
|---|---|
| [LICENSE](LICENSE) | `485a7818a8b75b6f9f992d42f7cb4aab46994e7eba276787b98baa220bf32389` |
| [NOTICE.md](NOTICE.md) | `9298b266f7d84c93f800a6d01ed86c915c5c0b92b03d1718bedd37426e06ae28` |
