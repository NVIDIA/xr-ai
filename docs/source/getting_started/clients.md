<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Connecting Clients

Every sample follows the same pattern: **start the server, then connect a
client.** This page covers the clients that ship under `client-samples/`, how
to set up, build, and run each one, and the shared token-on-startup connect
flow they all use.

For the server side — starting the XR-Media-Hub and the agent samples — refer to
{doc}`quickstart`.

## Which clients exist

| Client | Location | Transport | Build step |
|---|---|---|---|
| **Web** (basic sample) | `client-samples/web/` | LiveKit JS SDK from CDN | None — plain ES modules |
| **Web-XR** (XR render demo) | `client-samples/web-xr/` | LiveKit + CloudXR, same-origin vendor bundles | `client-samples/web-xr-build/build.sh` |
| **Android** | `client-samples/android/` | LiveKit Android SDK | Android Studio or Gradle |
| **iOS/visionOS** | `client-samples/ios-visionos/` | LiveKit Swift SDK | Xcode |
| **Native C++** | `client-samples/native/` | LiveKit C++ SDK | CMake |

All five share the same **StreamKit** shape: a single transport-agnostic
`StreamSession` entry-point delegating to a `StreamingBackend`, with
`LiveKitBackend` as the only component that imports LiveKit directly. The web,
Android, and iOS/visionOS clients are feature-equivalent (connection, audio,
camera, agent-status badge, data channel).

## The connect flow

When you start a server sample, the hub prints its connection details on
startup:

```
[hub]   LiveKit URL : wss://0.0.0.0:8080
[hub]   Room        : xr-room
[hub]   Token       : eyJ…
[hub]   Web client  : https://localhost:8080
```

Three values matter to every client:

| Value | Notes |
|---|---|
| **Host or IP** | The machine running the server. `localhost` for the same machine; the LAN IP for a separate device. |
| **Port** | `8080` — the hub's web-server port. Clients connect through the same-origin wss `/rtc` proxy on `8080`, **not** LiveKit's internal `7880`, which stays on loopback. |
| **Token** | A signed LiveKit JWT. Paste the printed token directly, or leave the token-URL field blank to fetch one from the hub's built-in `/token` endpoint. |

A client either pastes the printed JWT into its **Token** field, or points its
**Token URL** field at the hub's `/token` endpoint and lets the SDK fetch one.
The token endpoint accepts both response shapes — a plain JWT string or a
`{"token": "eyJ…"}` JSON envelope:

```
GET https://<host>:8080/token?identity=<identity>
```

Tokens are valid for 24 hours. To get a fresh one, restart the server or call
the `/token` endpoint above.

### Self-signed certificate trust

The samples ship with HTTPS on by default — a self-signed certificate is
generated on first run at `~/.local/share/xr-ai/web-server.crt`. Each platform
trusts it differently (browser click-through, Android KeyChain install, iOS
profile install); the per-platform sections below describe each. Firewall ports
and the option to run over plain HTTP instead are covered in {doc}`networking`.

## Web (basic sample)

`client-samples/web/` is the plain-ES-module browser client. It loads the
LiveKit JS SDK v2 directly from CDN via an import map, so there is **no build
step** — the hub serves the page at `https://localhost:8080`.

To connect:

1. Open `https://localhost:8080` in a browser.
2. On the first connection you'll see a "Your connection is not private"
   warning from the self-signed certificate. Click **Advanced → Proceed**
   (Chrome or Edge) or **Accept the Risk and Continue** (Firefox). To trust the
   certificate permanently or run over plain HTTP instead, refer to
   {doc}`networking`.
3. Leave **Token URL** blank — the web client fetches a token from the server's
   `/token` endpoint automatically. Alternatively, paste the printed token
   directly.
4. Click **Connect**. You are now live in the XR session.

## Web-XR (XR render demo)

The XR render demo client lives in `client-samples/web-xr/`. Unlike the basic
web sample, it loads `livekit-client` and `@nvidia/cloudxr` from same-origin
**vendor bundles** under `client-samples/web-xr/vendor/`, so XR headsets and
offline LANs work after the host has built the bundles once. The bundles are
generated build output, not shipped in the repository.

The `xr-render-demo` orchestrator builds the bundles automatically on first run
(requires `npm` on PATH). For a manual rebuild — for example after bumping an
SDK version — use the build script:

```bash
cd client-samples/web-xr-build
./build.sh
```

`build.sh` is idempotent. It reads the pinned CloudXR version from
`.sdk-version`, fetches the CloudXR Web SDK tarball (reusing a local copy if
present, otherwise downloading from public NGC), runs `npm install` (which also
pulls `livekit-client`), and webpacks the two ESM bundles into
`client-samples/web-xr/vendor/`.

To bump a dependency, edit `.sdk-version` (CloudXR) or the `livekit-client`
version in `package.json`, remove the cached artifacts (`rm -rf sdk.tgz
node_modules` for CloudXR, `rm -rf node_modules` for livekit-client), and
re-run `./build.sh`.

Once the bundles exist, connect through the demo the same way as the basic web
client — open the served page, click through the certificate warning, leave the token
URL blank, and connect. The basic `web/` sample does **not** need this build
step; only `web-xr/` does.

## Android

The Android client (`client-samples/android/`) provides feature parity with the
web and iOS/visionOS clients: connection, audio (Voice Processing, Software
AEC, or Raw modes), camera (Camera2 selector), agent-status badge, and data
channel. Remote audio plays automatically once a remote participant publishes a
track.

### Requirements

| Tool | Minimum version |
|---|---|
| Android Studio | Hedgehog (2023.1.1) or later |
| JDK | 17 |
| Android Gradle Plugin | 8.5 |
| Kotlin | 2.0 |
| Min Android SDK | API 24 (Android 7.0) |
| Target Android SDK | API 34 (Android 14) |

### Build and run

1. In Android Studio, choose **File → Open** and select
   `client-samples/android/`.
2. Let Gradle sync finish — it downloads the LiveKit Android SDK and other
   dependencies (~300 MB on first run) automatically.
3. Select a device or emulator running API 24+, then **Run**.

Android Studio generates the Gradle wrapper on first sync. For command-line builds, run
`gradle wrapper --gradle-version 8.9` then `./gradlew assembleDebug`.

The app requests `RECORD_AUDIO` on the first tap of **Start Microphone** and
`CAMERA` on the first tap of **Start Camera**; `INTERNET`,
`MODIFY_AUDIO_SETTINGS`, and `BLUETOOTH_CONNECT` are granted at install.

### Connect

In the app's Connection section:

| Field | Value |
|---|---|
| Host or IP | IP of the machine running the server |
| Port | `8080` (the hub web-server port, not LiveKit's internal 7880) |
| Token | Paste the printed JWT |
| Identity | Any string unique to this device |

Leave **Token URL** blank to use the server's default `/token` endpoint, or
paste the printed JWT directly into **Token**.

Before connecting for the first time, tap **Install hub certificate** in the
Connection section. The app fetches the certificate from the hub and opens the
system certificate-install dialog — confirm to install. After install, Android
validates
against the system + user CA store automatically.

### Android XR

The Android client is a standard Android app (`targetSdk` 34, `minSdk` 24) with
no XR-specific code. Android XR runs unmodified Android apps, so the sample
should install and launch on an Android XR device or emulator as a flat 2D
windowed panel, and the LiveKit audio and data paths, agent-status badge, and
token flow work the same as on a phone.

```{warning}
Android XR support is **not yet validated**. The sample has not been tested on
Android XR hardware or the emulator, and two areas are expected to need work:

- **Camera.** The `Camera2` selector targets phone front and back cameras.
  World-facing and passthrough camera access on Android XR is governed by
  different APIs and permissions, so live-camera perception may select the
  wrong camera or none.
- **Immersive rendering.** The app draws a 2D panel. It does not use Android XR's
  immersive APIs (Jetpack XR, OpenXR, spatial panels, head tracking, hand input,
  or passthrough), so it will not render head-tracked or spatialized content.
```

A fully immersive Android XR client — spatialized UI, head and hand input, and
optional CloudXR remote rendering — is future work. For an immersive XR path
today, refer to the **Web-XR (XR render demo)** client described above.

## iOS/visionOS

The iOS/visionOS client (`client-samples/ios-visionos/`) is a SwiftUI sample
for both iOS and visionOS, built on StreamKit over LiveKit WebRTC. The sample
ships as source files plus a local Swift Package; you assemble the Xcode
project from them.

The StreamKit library depends on an unmodified upstream
[livekit-client-sdk-swift](https://github.com/livekit/client-sdk-swift) checked
out at `../livekit-client-sdk-swift` (one level above the sample folder).

### Create the Xcode project

Following the sample's README, create a Multiplatform SwiftUI app, then:

1. **New project** — Xcode → **File → New → Project → Multiplatform → App**;
   Product Name `StreamKitSample`, Interface SwiftUI, Language Swift.
2. **Add destinations** — select the project root, then **Supported
   Destinations → +**; add **visionOS**, remove **macOS** if auto-added. You
   should be left with iOS and visionOS.
3. **Add the StreamKit package** — **File → Add Package Dependencies… → Add
   Local…**, navigate to `client-samples/ios-visionos/StreamKit/`, and add it,
   ticking **StreamKit** and your app target.
4. **Replace the generated sources** — delete the auto-generated
   `ContentView.swift` and app entry point, then drag in the four files from
   the sample's `App/` directory (`StreamKitSampleApp.swift`, `AppModel.swift`,
   `ContentView.swift`, `ImmersiveView.swift`) with both targets checked.
5. **Add `Info.plist` keys** — `NSMicrophoneUsageDescription`,
   `NSCameraUsageDescription`, and (for visionOS passthrough)
   `NSMainCameraUsageDescription`.

```{note}
On visionOS, main-passthrough-camera access is an Apple **enterprise** API: it
requires both the `com.apple.developer.arkit.main-camera-access.allow`
entitlement and your team's `Enterprise.license` bundled into the `.app`. The
license is not bundled with the sample; place it at
`client-samples/ios-visionos/App/Enterprise.license`. Without it the build
still succeeds and every feature except main-camera passthrough works. The
visionOS and iOS **simulators** require none of this: they stream a bundled
`SimulatorFeed.gif` in place of a physical camera.
```

Refer to the sample's README for the authoritative build steps.

### Connect

In the app's Connection section:

| Field | Value |
|---|---|
| Host | IP of the machine running the server |
| Port | `8080` (the hub web-server port; not LiveKit's internal 7880) |
| Token | Paste the token printed on server startup |

The token is valid for 24 hours; restart the server or call
`GET https://<host>:8080/token?identity=<name>` for a fresh one.

**Trusting the self-signed certificate (one-time per device):** the LiveKit
Swift SDK's `URLSession` does not expose a server-trust hook, so iOS rejects the
hub's self-signed certificate until you install it as a trusted profile. In the app's
Connection section, enter the host and port and tap **Install hub
certificate** — this opens Safari at `https://<host>:<port>/cert`. Bypass the
warning (**Show Details → visit this website**), allow the configuration
profile download, then install it under **Settings → General → VPN & Device
Management** and finally enable **Settings → General → About → Certificate
Trust Settings → Enable Full Trust** for the new certificate. The connection then
completes without warnings. The sample's README documents recovery for the
common failure modes (the Full-Trust toggle not appearing, `errSSLBadCert` or
`-1202`, and a 401 on the room after the certificate is trusted).

## Native C++

The native client (`client-samples/native/`) is a C++ StreamKit
implementation backed by the LiveKit C++ SDK (`livekit::Room`). It is aimed at
developers embedding StreamKit in a native host — an embedded device, a game
engine plugin, or a CloudXR client.

### Build and run

Point CMake at a LiveKit SDK install, build, and run with `--host` and
`--token`:

```bash
cmake -S . -B build -DLIVEKIT_SDK_ROOT=/path/to/livekit-cpp-sdk
cmake --build build
./build/bin/streamkit_sample --host 192.168.1.100 --token <jwt>
```

If `LIVEKIT_SDK_ROOT` is not set, the backend compiles in **stub mode**:
`Connect()` reports connected immediately without opening a real session. This
lets you build the rest of StreamKit without the LiveKit SDK present.

The native backend takes the JWT inline via `--token` or `LiveKitConfig::token`;
the token-URL HTTP fetch is not implemented in this backend. A small unit-test
suite is available with `-DSTREAMKIT_BUILD_TESTS=ON`, run via CTest. Refer to the
sample's README for the test matrix and the current backend constraints.

## Adding a client for a new platform

The XR-Media-Hub speaks standard LiveKit, so you are not limited to the bundled clients.
If [LiveKit publishes a client SDK](https://docs.livekit.io/reference/) for your
platform — Unity, Flutter, React Native, Rust, Go, and others — you can build a
client for it against the same contract the existing samples use:

1. **Get a token.** Fetch a JWT from the hub at
   `GET https://<host>:8080/token?identity=<name>`, or paste the token printed on
   server startup.
2. **Join the room.** Point the SDK at the hub's web-server port (`8080`, not
   LiveKit's internal `7880`) and connect with the token.
3. **Publish input.** Publish the microphone track, and the camera track if the
   agent needs vision.
4. **Handle agent output.** Play the remote audio track the agent publishes, and
   read its data-channel messages (for example, the `agent.response` text topic).

That is the whole integration surface — no XR-AI-specific protocol. The brittle
part is usually the hub's self-signed certificate: each platform trusts it
differently, so reuse the per-platform guidance above, or run the hub over plain
HTTP on a trusted network. Refer to {doc}`networking` for the certificate and
plain-HTTP options.
