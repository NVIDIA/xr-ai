<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Third-Party Notices

This file lists the third-party software distributed with or
required by `xr-ai`. Each entry gives the upstream project, version, SPDX
license identifier, and a link to the upstream source repository. Complete
license and attribution texts for the components noted below are bundled
under [`third_party_licenses/`](third_party_licenses/).

`xr-ai` itself is licensed under Apache-2.0 — see [`LICENSE`](LICENSE).

## Python (server-side and AI services)

Used by `agent-sdk/`, `utils/`, `services/`,
`agent-samples/` and `tests/`.
For the per-package dependency mapping, see [`DEPENDENCIES.md`](DEPENDENCIES.md).

The versions below follow the dependency declarations where a package is
listed with a minimum or range. Exact transitive versions are taken from the
qualified lock; the reciprocal-license table records exact resolved versions.

| Package        | Version  | License       | Upstream |
|---             |---       |---            |---|
| `msgpack`      | 1.0.0    | Apache-2.0    | https://github.com/msgpack/msgpack-python |
| `certifi`      | 2026.7.22 | MPL-2.0      | https://github.com/certifi/python-certifi |
| `pyzmq`        | 27.2.0   | BSD-3-Clause  | https://github.com/zeromq/pyzmq |
| `uvicorn`      | 0.29.0   | BSD-3-Clause  | https://github.com/encode/uvicorn |
| `fastapi`      | 0.111.0  | MIT           | https://github.com/fastapi/fastapi |
| `httpx`        | 0.27.0   | BSD-3-Clause  | https://github.com/encode/httpx |
| `livekit`      | 0.17.0   | Apache-2.0    | https://github.com/livekit/python-sdks |
| `livekit-api`  | 0.7.0    | Apache-2.0    | https://github.com/livekit/python-sdks |
| `numpy`        | 1.24.0   | BSD-3-Clause  | https://github.com/numpy/numpy |
| `nemo-relay`   | >=0.7.2,<0.8 | Apache-2.0  | https://github.com/NVIDIA/NeMo-Relay |
| `Pillow`       | 10.0.0   | HPND          | https://github.com/python-pillow/Pillow |
| `pipecat-ai`   | >=1.6    | BSD-2-Clause  | https://github.com/pipecat-ai/pipecat |
| `pocket-tts`   | 3.0.2    | MIT           | https://github.com/kyutai-labs/pocket-tts |
| `num2words`    | 0.5.14   | LGPL-2.1-or-later | https://github.com/savoirfairelinux/num2words |
| `pycountry`    | 26.2.16  | LGPL-2.1-only | https://github.com/pycountry/pycountry |
| `soxr`         | 1.0.0    | LGPL-2.1-or-later | https://github.com/dofuuz/python-soxr |
| `soundfile`    | 0.14.0   | BSD-3-Clause  | https://github.com/bastibe/python-soundfile |
| `text-unidecode` | 1.3    | Artistic-1.0-Perl (elected) | https://github.com/kmike/text-unidecode |
| `tqdm`         | 4.70.0   | MPL-2.0 AND MIT | https://github.com/tqdm/tqdm |
| `opencv-python-headless` | 5.0.0.93 | Apache-2.0 and MIT packaging terms (wheel includes other licenses) | https://github.com/opencv/opencv-python |
| `pydantic`     | >=2.10   | MIT           | https://github.com/pydantic/pydantic |
| `python-multipart` | >=0.0.9 | Apache-2.0 | https://github.com/Kludex/python-multipart |
| `websockets`   | 12.0     | BSD-3-Clause  | https://github.com/python-websockets/websockets |

### Qualified HTTP and JWT security releases

The release security refresh resolves `urllib3` 2.8.0 and `PyJWT` 2.15.1,
both under the MIT license. Unmodified upstream license texts are bundled in
[`urllib3-2.8.0`](third_party_licenses/urllib3-2.8.0/) and
[`pyjwt-2.15.1`](third_party_licenses/pyjwt-2.15.1/), including PyJWT's author
attributions. Each directory records the SHA-256-verified artifact URL and
copied-file hashes.

### Qualified pip vLLM security refresh

The release security refresh resolves the following versions. The linked
license texts were copied without modification from SHA-256-verified upstream
artifacts; each directory records the artifact URL and hashes.

| Component | Resolved version | License and notices |
|---|---|---|
| `vllm` | 0.30.0 | Apache-2.0; [upstream license](third_party_licenses/vllm-0.30.0/) |
| `huggingface-hub` | 1.32.0 | Apache-2.0; [upstream license](third_party_licenses/huggingface-hub-1.32.0/) |
| `flashinfer-python` | 0.6.18.post1 | Apache-2.0; [upstream license](third_party_licenses/flashinfer-python-0.6.18.post1/) |
| `instanttensor` | 0.2.0 | Apache-2.0; [upstream license](third_party_licenses/instanttensor-0.2.0/) |
| `quack-kernels` | 0.6.5 | Apache-2.0; [upstream license](third_party_licenses/quack-kernels-0.6.5/) |
| `protobuf` | 7.36.0 | BSD-3-Clause; [upstream license](third_party_licenses/protobuf-7.36.0/) |
| `nvidia-cutlass-dsl` | 4.7.1 | NVIDIA Software License Agreement; [upstream EULA](third_party_licenses/nvidia-cutlass-dsl-4.7.1/) |

`instanttensor` is a new transitive dependency of vLLM. CUTLASS DSL is
proprietary software downloaded by users, rather than part of XR AI's source
release. Its `libs-base`, `libs-core`, `libs-cu12`, and `libs-cu13` dependencies
also resolve to 4.7.1; their upstream package metadata identifies the NVIDIA
proprietary license. The BSD license of CUTLASS C++ does not replace the DSL
EULA. The lock also retains protobuf 6.33.5 for a separate dependency branch.

### Pocket TTS model and voice

Pocket TTS 3.0.2 pins the following downloaded artifacts for the default
English configuration. XR AI does not redistribute them.

- Gated weights from `kyutai/pocket-tts` at revision
  `39592ff23c9ef80098bb74895d104c26275fe2c9`.
- Ungated fallback weights and the tokenizer from
  `kyutai/pocket-tts-without-voice-cloning` at revision
  `d29db7978e464fb90cb3359ee0c69a273b9142cc`.
- The predefined `bill_boerst` voice embedding from
  `kyutai/pocket-tts-without-voice-cloning` at revision
  `e81d79e8194ad4c7ce879c87a4258ef20cbf2487`.

Both model repositories are licensed under CC-BY-4.0. The `bill_boerst`
embedding derives from the CC0-1.0 Voice-Zero recording identified in the
[`kyutai/tts-voices` model card](https://huggingface.co/kyutai/tts-voices), but
the default service downloads the precomputed embedding rather than that
recording.

The model records reviewed at repository revisions
`492522650173a0653b7575cdc25ae09810e5d741` (`kyutai/pocket-tts`) and
`e81d79e8194ad4c7ce879c87a4258ef20cbf2487`
(`kyutai/pocket-tts-without-voice-cloning`) carried the following
acceptable-use text on 2026-09-09:

> Prohibited use: Use of our model must comply with all applicable laws and
> regulations and must not result in, involve, or facilitate any illegal,
> harmful, deceptive, fraudulent, or unauthorized activity. Prohibited uses
> include, without limitation, voice impersonation or cloning without explicit
> and lawful consent; misinformation, disinformation, or deception (including
> fake news, fraudulent calls, or presenting generated content as genuine
> recordings of real people or events); and the generation of unlawful,
> harmful, libelous, abusive, harassing, discriminatory, hateful, or
> privacy-invasive content. We disclaim all liability for any non-compliant use.

The exact UTF-8 text, without a trailing newline, has SHA-256
`8febd058c61bdeebf64b22d5a7bd3a78081e9b136a845fd98a2daf1de9d6381e`.
Review the live [gated model record](https://huggingface.co/api/models/kyutai/pocket-tts)
before accepting the terms because upstream may revise them. This release
accepts only the reviewed `bill_boerst` voice; it does not expose Pocket TTS's
other predefined voices or arbitrary voice-cloning inputs.

### Reciprocal licenses and bundled native libraries

The resolved Python environment contains the following reciprocal-license
components. They are unmodified upstream dependencies; no source from them is
copied into XR AI. Their complete license and attribution texts are bundled in
this repository.

| Component | How it is used | License and notices | Corresponding source |
|---|---|---|---|
| `num2words` 0.5.14 | Python dependency of `pipecat-ai` | LGPL-2.1-or-later; [`COPYING`](third_party_licenses/num2words-0.5.14/COPYING) | [0.5.14 source distribution](https://pypi.org/project/num2words/0.5.14/#files) |
| `pycountry` 26.2.16 | Transitive dependency through `vllm` → `mistral-common` → `pydantic-extra-types[pycountry]` | LGPL-2.1-only; [`LICENSE.txt` and data attributions](third_party_licenses/pycountry-26.2.16/) | [26.2.16 source distribution](https://pypi.org/project/pycountry/26.2.16/#files) |
| `tqdm` 4.70.0 | Progress reporting used by AI dependencies | MPL-2.0 AND MIT; [`LICENCE`](third_party_licenses/tqdm-4.70.0/LICENCE) | [4.70.0 source distribution](https://pypi.org/project/tqdm/4.70.0/#files) |
| `soxr` 1.0.0 / libsoxr 0.1.3 / PFFFT | Python extension used through `pipecat-ai` and `librosa`; its wheel includes PFFFT and Python-SoXR's modified libsoxr fork | LGPL-2.1-or-later for libsoxr and permissive PFFFT terms; [`license files`](third_party_licenses/soxr-1.0.0/) | [`soxr` 1.0.0 source distribution](https://pypi.org/project/soxr/1.0.0/#files); its `libsoxr/` directory is the applicable fork source |
| `soundfile` 0.14.0 / libsndfile 1.2.2 | Platform wheels include libsndfile, which statically includes libmp3lame 3.100 and libmpg123 | LGPL-2.1-or-later for libsndfile, LGPL-2.0-or-later for libmp3lame, and LGPL-2.1 for libmpg123; [`COPYING`](third_party_licenses/libsndfile-1.2.2/COPYING), [wrapper license and native-library source notes](third_party_licenses/soundfile-0.14.0/) | [libsndfile 1.2.2 source](https://github.com/libsndfile/libsndfile/releases/tag/1.2.2) and [SoundFile 0.14.0 source distribution](https://pypi.org/project/soundfile/0.14.0/#files) |
| `opencv-python-headless` 5.0.0.93 / FFmpeg | Platform wheels include FFmpeg shared libraries | LGPL-2.1-or-later; [`OpenCV wheel and bundled-library notices`](third_party_licenses/opencv-python-headless-5.0.0.93/) | [5.0.0.93 source distribution](https://pypi.org/project/opencv-python-headless/5.0.0.93/#files) and [FFmpeg source](https://ffmpeg.org/download.html) |
| `certifi` 2026.7.22 | CA certificate bundle used by HTTP clients | MPL-2.0; [`LICENSE`](third_party_licenses/certifi-2026.7.22/LICENSE) | [2026.7.22 source distribution](https://pypi.org/project/certifi/2026.7.22/#files) |
| `pyzmq` 27.2.0 / libzmq 4.3.5 / libsodium 1.0.22 | Platform wheels include libzmq and libsodium shared libraries | MPL-2.0 for libzmq and ISC for libsodium; [`PyZMQ and bundled-library notices`](third_party_licenses/pyzmq-27.2.0/) | [libzmq 4.3.5 source](https://github.com/zeromq/libzmq/releases/tag/v4.3.5), [libsodium 1.0.22 source](https://github.com/jedisct1/libsodium/releases/tag/1.0.22-RELEASE), and [PyZMQ 27.2.0 source distribution](https://pypi.org/project/pyzmq/27.2.0/#files) |

These obligations apply to downstream redistributors of installed environments,
containers, or other binary bundles; they do not change XR AI's Apache-2.0
license. The standard Python installation keeps the LGPL native components in
replaceable shared-library or extension files rather than incorporating them
into XR AI source files. Anyone redistributing a built environment, container,
or other binary bundle must preserve these notices, permit replacement and
reverse engineering for debugging modifications as required by LGPL-2.1, and
accompany the LGPL components with their complete corresponding source. A
redistributor that modifies an LGPL or MPL component must also publish the
applicable modified source under that component's license. The source links
above identify the unmodified versions resolved for this release; artifact
URLs and SHA-256 hashes are pinned in
[`dependency-manifest/uv.lock`](dependency-manifest/uv.lock).

### Additional transitive package notices

The qualified lock also resolves the following packages. XR AI uses them as
unmodified dependencies installed by users; their implementation is not copied
into XR AI source. Each notice directory includes the exact inspected upstream
artifact and SHA-256 hashes for the copied texts.

| Package | Version | License and notices | Dependency path |
|---|---|---|---|
| `llvmlite` | 0.47.0 | BSD-2-Clause AND Apache-2.0 WITH LLVM-exception; [license texts](third_party_licenses/llvmlite-0.47.0/) | AI and audio dependencies → `numba` → `llvmlite` |
| `nvtx` | 0.2.15 | Apache-2.0 WITH LLVM-exception; [license text](third_party_licenses/nvtx-0.2.15/) | Model services → `vllm` → `nvtx` |
| `matplotlib` | 3.11.1 | Matplotlib License 1.3+ and additional font/library terms; [license texts](third_party_licenses/matplotlib-3.11.1/) | Magpie TTS → `nemo-toolkit[tts]` → `matplotlib` |
| `regex` | 2026.7.19 | Apache-2.0 AND CNRI-Python; [license text](third_party_licenses/regex-2026.7.19/) | Voice SDK → `nltk`; model dependencies → `transformers` / `nemo-text-processing` |
| `supervisor` | 4.3.0 | BSD-derived Repoze license, including its modification notice clause; [license text](third_party_licenses/supervisor-4.3.0/) | Model services → `vllm` → `model-hosting-container-standards` → `supervisor` |
| `humming-kernels` | 0.1.12 | Apache-2.0; [license text](third_party_licenses/humming-kernels-0.1.12/) | Model services → `vllm` → `humming-kernels` |
| `quack-kernels` | 0.6.5 | Apache-2.0; [license text](third_party_licenses/quack-kernels-0.6.5/) | Model services → `vllm` → `quack-kernels` |
| `torchcodec` | 0.16.0 | BSD-3-Clause and platform-specific bundled-library terms, including NVIDIA CUDA EULA for Linux nvJPEG; [license texts and platform scope](third_party_licenses/torchcodec-0.16.0/) | Model services → `vllm` → `torchcodec` |

`AND` license expressions retain all listed terms; they are not alternative
license elections. XR AI elects the permissive FreeType License (FTL) option
for Matplotlib's FreeType dependency. Portions of this software are copyright
© The FreeType Project (https://www.freetype.org/). All rights reserved.
The complete upstream FreeType license notice, including its alternatives,
is preserved in the Matplotlib notice directory.

### NVIDIA CUTLASS DSL packages

The following packages are covered by the NVIDIA CUTLASS DSL EULA. They are
separate from the BSD-licensed CUTLASS C++ project. Users download these
packages during installation; XR AI does not redistribute their binaries.

| Package | Version | Terms |
|---|---|---|
| `nvidia-cutlass-dsl` | 4.7.1 | NVIDIA proprietary CUTLASS DSL EULA |
| `nvidia-cutlass-dsl-libs-base` | 4.7.1 | NVIDIA proprietary CUTLASS DSL EULA |
| `nvidia-cutlass-dsl-libs-core` | 4.7.1 | NVIDIA proprietary CUTLASS DSL EULA |
| `nvidia-cutlass-dsl-libs-cu12` | 4.7.1 | NVIDIA proprietary CUTLASS DSL EULA |
| `nvidia-cutlass-dsl-libs-cu13` | 4.7.1 | NVIDIA proprietary CUTLASS DSL EULA |

The [upstream license page](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/license.html)
and [copied EULA with artifact provenance](third_party_licenses/nvidia-cutlass-dsl-4.7.1/)
record the applicable terms. The permissive license elections elsewhere in
this file do not apply to these proprietary packages.

### `text-unidecode` license election

`text-unidecode` 1.3 is available under the Artistic License or GPLv2+. XR AI
elects the **Artistic-1.0-Perl** option and does not rely on the GPL grant. The
package is an unmodified transitive dependency of NeMo Toolkit. Its complete
dual-license notice and Artistic License text are bundled as
[`third_party_licenses/text-unidecode-1.3/LICENSE`](third_party_licenses/text-unidecode-1.3/LICENSE),
and its corresponding source is the
[`text-unidecode` 1.3 source distribution](https://pypi.org/project/text-unidecode/1.3/#files).

## Swift (iOS / visionOS client)

Used by `client-samples/ios-visionos/`. Resolved via Swift Package Manager.

| Package | Version | License | Upstream |
|---|---|---|---|
| `LiveKitClient` (`livekit/client-sdk-swift`)            | 2.16.0       | Apache-2.0   | https://github.com/livekit/client-sdk-swift |
| `livekit/webrtc-xcframework`                            | 144.7559.3   | MIT          | https://github.com/livekit/webrtc-xcframework |
| `livekit/livekit-uniffi-xcframework`                    | 0.0.5        | Apache-2.0   | https://github.com/livekit/livekit-uniffi-xcframework |
| `swift-protobuf` (`apple/swift-protobuf`)               | 1.37.0       | Apache-2.0   | https://github.com/apple/swift-protobuf |

## Android build toolchain (`client-samples/android/`)

The bundled Gradle wrapper files are generated by
[Gradle](https://github.com/gradle/gradle):
[`gradlew`](client-samples/android/gradlew),
[`gradlew.bat`](client-samples/android/gradlew.bat),
[`gradle-wrapper.jar`](client-samples/android/gradle/wrapper/gradle-wrapper.jar),
and [`gradle-wrapper.properties`](client-samples/android/gradle/wrapper/gradle-wrapper.properties).
The properties file selects Gradle 8.9 through its `distributionUrl`.
The scripts retain their upstream copyright notices:
"Copyright © 2015-2021 the original authors" in `gradlew` and
"Copyright 2015 the original author or authors" in `gradlew.bat`.
The wrapper files are licensed under
[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0).
The JAR contains compiled Gradle Wrapper classes and its own license text in
`META-INF/LICENSE`; the Apache-2.0 text is also bundled in [`LICENSE`](LICENSE).

Building the Android sample with the Android Gradle Plugin (declared in
`client-samples/android/gradle/libs.versions.toml`) places dual-licensed
artifacts on the build classpath, including the following, for which `xr-ai`
records a license election:

| Package | Version | Dual license | Elected | Upstream |
|---|---|---|---|---|
| Java Native Access (`net.java.dev.jna:jna`, `jna-platform`) | 5.6.0 | LGPL-2.1-or-later OR Apache-2.0 | Apache-2.0 | https://github.com/java-native-access/jna |
| `javax.annotation:javax.annotation-api`                 | 1.3.2 | CDDL-1.1 OR GPL-2.0-with-classpath-exception | CDDL-1.1 | https://github.com/eclipse-ee4j/common-annotations-api |

Neither artifact is bundled into the APK produced by the current build
configuration (verified against the resolved `releaseRuntimeClasspath`), and
`xr-ai` does not redistribute them; the election records the terms under
which each artifact is used at build time.

## License texts

The full text of each SPDX license identifier that applies to `xr-ai`'s use
of the software above (for dual-licensed artifacts, the elected license) is
available at:

- **Apache-2.0**: https://www.apache.org/licenses/LICENSE-2.0 — also bundled
  with this repository as [`LICENSE`](LICENSE).
- **BSD-3-Clause**: https://opensource.org/license/bsd-3-clause
- **BSD-2-Clause**: https://opensource.org/license/bsd-2-clause
- **HPND**: https://opensource.org/license/historical-ntu-disclaimer
- **MIT**: https://opensource.org/license/mit
- **CC-BY-4.0**: https://creativecommons.org/licenses/by/4.0/
- **CC0-1.0**: https://creativecommons.org/publicdomain/zero/1.0/
- **CDDL-1.1**: https://spdx.org/licenses/CDDL-1.1.html
- **LGPL-2.1-or-later**: bundled with the applicable components under
  [`third_party_licenses/`](third_party_licenses/)
- **LGPL-2.1-only** and **LGPL-2.0-or-later**: bundled with the applicable
  components under [`third_party_licenses/`](third_party_licenses/)
- **MPL-2.0**: bundled with the applicable components under
  [`third_party_licenses/`](third_party_licenses/)
- **Artistic-1.0-Perl**: bundled as
  [`third_party_licenses/text-unidecode-1.3/LICENSE`](third_party_licenses/text-unidecode-1.3/LICENSE)

Each upstream project repository linked above includes its own canonical
license file (typically `LICENSE`, `LICENSE.txt`, or `COPYING`).

## Bundled license file index

The following index links all preserved license files under
`third_party_licenses/`, including licenses for bundled native libraries, fonts,
and other upstream components. The upstream texts are preserved unchanged;
refer to the component disclosures above for applicable license elections.

| Component | Preserved license file |
|---|---|
| `certifi-2026.7.22` | [`LICENSE`](third_party_licenses/certifi-2026.7.22/LICENSE) |
| `flashinfer-python-0.6.18.post1` | [`LICENSE`](third_party_licenses/flashinfer-python-0.6.18.post1/LICENSE) |
| `huggingface-hub-1.32.0` | [`LICENSE`](third_party_licenses/huggingface-hub-1.32.0/LICENSE) |
| `humming-kernels-0.1.12` | [`LICENSE`](third_party_licenses/humming-kernels-0.1.12/LICENSE) |
| `instanttensor-0.2.0` | [`LICENSE`](third_party_licenses/instanttensor-0.2.0/LICENSE) |
| `libsndfile-1.2.2` | [`COPYING`](third_party_licenses/libsndfile-1.2.2/COPYING) |
| `llvmlite-0.47.0` | [`LICENSE`](third_party_licenses/llvmlite-0.47.0/LICENSE) |
| `llvmlite-0.47.0` | [`LICENSE.thirdparty`](third_party_licenses/llvmlite-0.47.0/LICENSE.thirdparty) |
| `matplotlib-3.11.1` | [`FTL.TXT`](third_party_licenses/matplotlib-3.11.1/FTL.TXT) |
| `matplotlib-3.11.1` | [`LICENSE`](third_party_licenses/matplotlib-3.11.1/LICENSE) |
| `matplotlib-3.11.1` | [`LICENSE_AMSFONTS`](third_party_licenses/matplotlib-3.11.1/LICENSE_AMSFONTS) |
| `matplotlib-3.11.1` | [`LICENSE_BAKOMA`](third_party_licenses/matplotlib-3.11.1/LICENSE_BAKOMA) |
| `matplotlib-3.11.1` | [`LICENSE_COLORBREWER`](third_party_licenses/matplotlib-3.11.1/LICENSE_COLORBREWER) |
| `matplotlib-3.11.1` | [`LICENSE_COURIERTEN`](third_party_licenses/matplotlib-3.11.1/LICENSE_COURIERTEN) |
| `matplotlib-3.11.1` | [`LICENSE_DEJAVU`](third_party_licenses/matplotlib-3.11.1/LICENSE_DEJAVU) |
| `matplotlib-3.11.1` | [`LICENSE_FONT_AWESOME`](third_party_licenses/matplotlib-3.11.1/LICENSE_FONT_AWESOME) |
| `matplotlib-3.11.1` | [`LICENSE_FREETYPE`](third_party_licenses/matplotlib-3.11.1/LICENSE_FREETYPE) |
| `matplotlib-3.11.1` | [`LICENSE_HARFBUZZ`](third_party_licenses/matplotlib-3.11.1/LICENSE_HARFBUZZ) |
| `matplotlib-3.11.1` | [`LICENSE_JSXTOOLS_RESIZE_OBSERVER`](third_party_licenses/matplotlib-3.11.1/LICENSE_JSXTOOLS_RESIZE_OBSERVER) |
| `matplotlib-3.11.1` | [`LICENSE_LAST_RESORT_FONT`](third_party_licenses/matplotlib-3.11.1/LICENSE_LAST_RESORT_FONT) |
| `matplotlib-3.11.1` | [`LICENSE_LIBRAQM`](third_party_licenses/matplotlib-3.11.1/LICENSE_LIBRAQM) |
| `matplotlib-3.11.1` | [`LICENSE_QT4_EDITOR`](third_party_licenses/matplotlib-3.11.1/LICENSE_QT4_EDITOR) |
| `matplotlib-3.11.1` | [`LICENSE_SHEENBIDI`](third_party_licenses/matplotlib-3.11.1/LICENSE_SHEENBIDI) |
| `matplotlib-3.11.1` | [`LICENSE_SOLARIZED`](third_party_licenses/matplotlib-3.11.1/LICENSE_SOLARIZED) |
| `matplotlib-3.11.1` | [`LICENSE_STIX`](third_party_licenses/matplotlib-3.11.1/LICENSE_STIX) |
| `matplotlib-3.11.1` | [`LICENSE_YORICK`](third_party_licenses/matplotlib-3.11.1/LICENSE_YORICK) |
| `matplotlib-3.11.1` | [`copying`](third_party_licenses/matplotlib-3.11.1/copying) |
| `num2words-0.5.14` | [`COPYING`](third_party_licenses/num2words-0.5.14/COPYING) |
| `nvidia-cutlass-dsl-4.7.1` | [`LICENSE`](third_party_licenses/nvidia-cutlass-dsl-4.7.1/LICENSE) |
| `nvtx-0.2.15` | [`LICENSE.txt`](third_party_licenses/nvtx-0.2.15/LICENSE.txt) |
| `opencv-python-headless-5.0.0.93` | [`LICENSE-3RD-PARTY.txt`](third_party_licenses/opencv-python-headless-5.0.0.93/LICENSE-3RD-PARTY.txt) |
| `opencv-python-headless-5.0.0.93` | [`LICENSE.txt`](third_party_licenses/opencv-python-headless-5.0.0.93/LICENSE.txt) |
| `protobuf-7.36.0` | [`LICENSE`](third_party_licenses/protobuf-7.36.0/LICENSE) |
| `pycountry-26.2.16` | [`LICENSE.txt`](third_party_licenses/pycountry-26.2.16/LICENSE.txt) |
| `pyjwt-2.15.1` | [`LICENSE`](third_party_licenses/pyjwt-2.15.1/LICENSE) |
| `pyzmq-27.2.0` | [`LICENSE`](third_party_licenses/pyzmq-27.2.0/LICENSE) |
| `pyzmq-27.2.0` | [`LICENSE.libsodium`](third_party_licenses/pyzmq-27.2.0/LICENSE.libsodium) |
| `pyzmq-27.2.0` | [`LICENSE.tornado`](third_party_licenses/pyzmq-27.2.0/LICENSE.tornado) |
| `pyzmq-27.2.0` | [`LICENSE.zeromq`](third_party_licenses/pyzmq-27.2.0/LICENSE.zeromq) |
| `quack-kernels-0.6.5` | [`LICENSE`](third_party_licenses/quack-kernels-0.6.5/LICENSE) |
| `regex-2026.7.19` | [`LICENSE.txt`](third_party_licenses/regex-2026.7.19/LICENSE.txt) |
| `soundfile-0.14.0` | [`LICENSE`](third_party_licenses/soundfile-0.14.0/LICENSE) |
| `soxr-1.0.0` | [`COPYING.LGPL`](third_party_licenses/soxr-1.0.0/COPYING.LGPL) |
| `soxr-1.0.0` | [`LICENSE`](third_party_licenses/soxr-1.0.0/LICENSE) |
| `soxr-1.0.0` | [`LICENSE-PFFFT`](third_party_licenses/soxr-1.0.0/LICENSE-PFFFT) |
| `soxr-1.0.0` | [`LICENSE-libsoxr`](third_party_licenses/soxr-1.0.0/LICENSE-libsoxr) |
| `supervisor-4.3.0` | [`LICENSES.txt`](third_party_licenses/supervisor-4.3.0/LICENSES.txt) |
| `text-unidecode-1.3` | [`LICENSE`](third_party_licenses/text-unidecode-1.3/LICENSE) |
| `torchcodec-0.16.0` | [`COPYING.dav1d`](third_party_licenses/torchcodec-0.16.0/COPYING.dav1d) |
| `torchcodec-0.16.0` | [`LICENSE`](third_party_licenses/torchcodec-0.16.0/LICENSE) |
| `torchcodec-0.16.0` | [`LICENSE.libavif`](third_party_licenses/torchcodec-0.16.0/LICENSE.libavif) |
| `torchcodec-0.16.0` | [`LICENSE.libjpeg-turbo`](third_party_licenses/torchcodec-0.16.0/LICENSE.libjpeg-turbo) |
| `torchcodec-0.16.0` | [`LICENSE.libnvjpeg-NVIDIA-CUDA-EULA.txt`](third_party_licenses/torchcodec-0.16.0/LICENSE.libnvjpeg-NVIDIA-CUDA-EULA.txt) |
| `torchcodec-0.16.0` | [`LICENSE.libpng`](third_party_licenses/torchcodec-0.16.0/LICENSE.libpng) |
| `torchcodec-0.16.0` | [`LICENSE.libwebp`](third_party_licenses/torchcodec-0.16.0/LICENSE.libwebp) |
| `torchcodec-0.16.0` | [`LICENSE.libyuv`](third_party_licenses/torchcodec-0.16.0/LICENSE.libyuv) |
| `torchcodec-0.16.0` | [`LICENSE.zlib`](third_party_licenses/torchcodec-0.16.0/LICENSE.zlib) |
| `tqdm-4.70.0` | [`LICENCE`](third_party_licenses/tqdm-4.70.0/LICENCE) |
| `urllib3-2.8.0` | [`LICENSE.txt`](third_party_licenses/urllib3-2.8.0/LICENSE.txt) |
| `vllm-0.30.0` | [`LICENSE`](third_party_licenses/vllm-0.30.0/LICENSE) |

Additional attribution and native-library notices are preserved in
[`pycountry COPYRIGHT.txt`](third_party_licenses/pycountry-26.2.16/COPYRIGHT.txt),
[`PyJWT AUTHORS.rst`](third_party_licenses/pyjwt-2.15.1/AUTHORS.rst), and
[`SoundFile license notes`](third_party_licenses/soundfile-0.14.0/license_notes.md).

## Updating this file

When adding, removing, or upgrading a third-party dependency:

1. Update the relevant manifest — `pyproject.toml` (Python),
   `Package.swift` (Swift), `gradle/libs.versions.toml` (Android), or the
   web client's manifest.
2. Update [`DEPENDENCIES.md`](DEPENDENCIES.md) — the internal/external
   dependency map.
3. Inspect the resolved wheel and source metadata, including bundled native
   libraries and their native versions, for license and attribution terms.
4. Update this file and copy the exact upstream license, attribution, and
   native-library source notices into [`third_party_licenses/`](third_party_licenses/).
5. Run `uv run pytest tests/test_third_party_notices.py` to verify that the
   notice versions and bundled texts match the qualified lock.

All updates belong in the same commit.
