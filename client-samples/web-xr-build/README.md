<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# web-xr-build — web vendor bundles

Produces the vendor bundles for `../web-xr/vendor/`:

- `cloudxr-sdk.esm.mjs`   — webpack-bundled from the @nvidia/cloudxr NGC tarball.
- `livekit-client.esm.mjs` — copied from npm's prebuilt ESM.

The bundles are gitignored — fetched/built per host. The browser loads
them same-origin, so XR headsets and offline LANs work after the host
has run this script once.

`../web/` (the basic sample) loads LiveKit directly from CDN and requires
no build step. Only `../web-xr/` (the XR render demo) needs this build.

## Usage

```bash
cd client-samples/web-xr-build
./build.sh
```

`build.sh` is idempotent:

1. Reads the pinned CloudXR version from `.sdk-version` (currently `6.1.0`).
2. If `sdk.tgz` is already present, skips the fetch.
3. Else, if `~/hub/IsaacTeleop/deps/cloudxr/nvidia-cloudxr-${VERSION}.tgz`
   exists, copies it (saves a download during local development).
4. Else, fetches from public NGC:
   `https://api.ngc.nvidia.com/v2/resources/org/nvidia/cloudxr-js/${VERSION}/files?redirect=true&path=nvidia-cloudxr-${VERSION}.tgz`
5. Runs `npm install` (which also pulls `livekit-client` from npm).
6. Runs `npm run build` — webpack writes `../web/vendor/cloudxr-sdk.esm.mjs`.
7. Copies `node_modules/livekit-client/dist/livekit-client.esm.mjs` →
   `../web/vendor/livekit-client.esm.mjs`.

## Bumping the CloudXR SDK version

1. Edit `.sdk-version`.
2. `rm -rf sdk.tgz node_modules` for a clean build.
3. Re-run `./build.sh`.

## Bumping livekit-client

1. Edit the `livekit-client` version in `package.json`.
2. `rm -rf node_modules` (forces npm to resolve the new range).
3. Re-run `./build.sh`.

## Files

- `.sdk-version` — pinned CloudXR Web SDK version.
- `build.sh` — the one command.
- `package.json` — declares `@nvidia/cloudxr` (local tarball) and
  `livekit-client` (from npm).
- `webpack.config.js` — emits the CloudXR ESM bundle with
  `experiments.outputModule`.
- `src/index.js` — re-exports the CloudXR symbols the page uses.
- `sdk.tgz`, `node_modules/`, `package-lock.json` — gitignored.
- `../web-xr/vendor/*.esm.mjs` — gitignored (output of this script).
