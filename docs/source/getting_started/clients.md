<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Connecting clients

Every sample starts DeviceIOHub and then accepts one or more platform clients.
This client reference is the canonical build and connection guide for the
clients under `client-samples/`. Refer to {doc}`quickstart` for starting an
agent sample and {doc}`networking` for firewall and TLS configuration.

(which-clients-exist)=
## Client matrix

| Client | Directory | Transport | Build |
|---|---|---|---|
| Web | `client-samples/web/` | LiveKit from CDN | None |
| Web-XR | `client-samples/web-xr/` | Local LiveKit and CloudXR bundles | `web-xr-build/build.sh` |
| Android | `client-samples/android/` | LiveKit Android | Android Studio or Gradle |
| iOS/visionOS | `client-samples/ios-visionos/` | LiveKit Swift and CloudXRKit | Xcode |
| Native C++ | `client-samples/native/` | LiveKit C++ | CMake |

The clients share a StreamKit shape: one transport-neutral `StreamSession`
delegates to a `StreamingBackend`, and `LiveKitBackend` is the only layer that
imports a LiveKit SDK. Connection, microphone, camera, participant status, data,
and network metrics remain separate operations.

(network-telemetry)=
Graphical clients display LiveKit connection quality, round-trip time, and
receive jitter; the C++ sample reports them through its callback. Backends sample
existing WebRTC statistics about once per second. No hub message or separate
telemetry service is involved.

(the-connect-flow)=
## Shared connection flow

The hub prints a URL, room, development token, and web-client URL at startup.
Browser, Android, and Apple clients connect to the hub's web-server port `8080`,
which proxies LiveKit on the same origin. The native C++ client also supports
this TLS path with `--secure --port 8080`. Its port 7880 default is a plaintext
direct-debugging exception; restrict it to the XR AI host or a trusted source
subnet as described in {doc}`networking`.

A client may paste the printed 24-hour development JWT or fetch a shorter-lived
token from:

```text
GET https://<host>:8080/token?identity=<identity>
```

The response is a JSON object containing `token`, `room`, and `url`. StreamKit
token fetchers also accept a plain JWT response. The room is encoded in the
token; clients do not send a separate room query parameter.

(self-signed-certificate-trust)=
The default HTTPS certificate is self-signed and stored under
`~/.local/share/xr-ai/`. Browsers allow a development click-through; Android
and Apple devices require the platform trust procedures below. Use a public or
managed certificate for production.

(web-basic-sample)=
## Web

The basic page in `client-samples/web/` uses plain ES modules and loads LiveKit
from jsDelivr. It has no build step. Open `https://<host>:8080`, accept the
development certificate warning, leave Token URL blank to use `/token`, and
connect. The camera preview follows the published track's aspect ratio.

## Web-XR (xr-render-demo)

The Web-XR page uses same-origin bundles under
`client-samples/web-xr/vendor/`. The xr-render orchestrator builds missing
bundles automatically when `npm` is available. To rebuild manually:

```bash
cd client-samples/web-xr-build
./build.sh
```

The idempotent script reads the CloudXR version from `.sdk-version`, reuses a
cached `sdk.tgz` when present, otherwise reuses the matching IsaacTeleop tarball
at `~/hub/IsaacTeleop/deps/cloudxr/` when available, and finally downloads the
public NGC tarball. It installs npm dependencies and writes both ESM bundles to
`../web-xr/vendor/`.
To bump CloudXR, edit `.sdk-version` and remove `sdk.tgz` plus `node_modules`.
To bump LiveKit, edit `package.json` and remove `node_modules`. Re-run the script
after either change.

## Android

The Android app is a Jetpack Compose client with selectable Camera2 devices,
microphone capture, participant status, arbitrary data, and network metrics.

(requirements)=
(build-and-run)=
### Requirements and build

| Requirement | Version |
|---|---|
| Android Studio | Koala 2024.1.1 or newer |
| JDK | 17 |
| Minimum Android | API 24 |
| Target Android | API 34 |

Open `client-samples/android/` in Android Studio, allow Gradle sync to finish,
select a device, and run. The checked-in wrapper also supports:

```bash
cd client-samples/android
./gradlew assembleDebug
```

(connect)=
### Connect and permissions

Enter the server host, port `8080`, a unique identity, and either the printed
token or the default token URL. Tap **Install hub certificate** before the
first connection; Android opens the system certificate-install flow.

The app requests microphone permission when starting the microphone and camera
permission when opening a physical camera. `INTERNET` and
`MODIFY_AUDIO_SETTINGS` are install-time permissions. `BLUETOOTH_CONNECT` is a
runtime permission on Android 12 and newer; the current sample declares it but
does not prompt for it, so grant Nearby devices in system settings before
depending on Bluetooth audio routing.

The current Android backend maps every enabled `AudioConfig` mode to LiveKit's
default microphone capture. It does not yet select distinct voice-processing,
software-processing, or raw DSP settings, so the sample UI does not expose a
DSP-mode picker. LiveKit plays remote agent audio automatically. Camera choices
come from Camera2 and include front, back, extra built-in lenses, and attached
USB cameras when the device exposes them.

The root Android build pins the Netty dependency pulled in by AGP's test tooling
above known vulnerable 4.1 releases. After changing AGP or the version catalog,
run `./gradlew verifyNettyPin` from `client-samples/android/`.

(android-xr)=
The app can run as a flat Android panel on Android XR, but that path is not
validated. It does not use Jetpack XR or immersive Android XR APIs, and its
Camera2 selector is not a passthrough-camera integration.

## iOS/visionOS

The Apple client is a checked-in SwiftUI project and local Swift package. It
targets iOS 18 and visionOS 26, requires Xcode 26 with Swift 6.2, and resolves
LiveKit Swift and CloudXRKit through Swift Package Manager.

(create-the-xcode-project)=
### Build

Open `client-samples/ios-visionos/StreamKitSample.xcworkspace`, select the
`StreamKitSample` scheme and destination, choose a signing team and a bundle ID
owned by that team, then build and run. The workspace already includes the app
project and local StreamKit package; do not create a replacement Xcode project.

The checked-in bundle ID is `com.nvidia.xr-ai-example`. Changing it does not
change the `StreamKitSample` display name, but it does select a new
`UserDefaults` domain, so saved settings reset on the first launch after a
rename.

The simulators stream
`StreamKit/Sources/StreamKit/Resources/SimulatorFeed.gif` instead of a physical
camera. Replace that resource to customize the simulated feed.

On an iOS device the preview follows the live camera aspect ratio. The Vision
Pro main-camera track goes directly from ARKit to LiveKit and is not copied into
the 2D preview; the `LIVE` badge, rather than the placeholder preview, indicates
capture. Start the immersive space before camera capture on a Vision Pro device.

### Vision Pro permissions

Main passthrough camera access on a Vision Pro device requires both the
`com.apple.developer.arkit.main-camera-access.allow` entitlement and the team's
Apple-issued `Enterprise.license`. Put the non-redistributable license at
`client-samples/ios-visionos/App/Enterprise.license`; the build phase copies it
into the application. A missing license leaves audio, data, simulator camera,
and the rest of the app usable, but disables device passthrough camera access.

Xcode automatic signing supports development builds. App Store and TestFlight
distribution require a manually issued provisioning profile that grants the
main-camera entitlement.

The native CloudXR path also declares
`com.apple.developer.low-latency-streaming`, which requires an Apple Developer
Program team. A team that cannot provision this entitlement can remove it from
`client-samples/ios-visionos/App/StreamKitSample.entitlements`; native CloudXR
remains available with higher latency. Simulators do not require the enterprise
camera license.

### Connect and trust the certificate

Enter the host and port `8080`, then use a pasted token or the default `/token`
endpoint. `SessionConfig` contains only the participant identity; the token
encodes the room. Incoming data callbacks receive `(topic, data)`, and
`_agent.status` is delivered separately through the status callback.

The LiveKit WebSocket requires system trust for the hub certificate. In the
app, tap **Install hub certificate**, allow Safari to download the profile,
install it under **Settings → General → VPN & Device Management**, then enable
full trust under **Settings → General → About → Certificate Trust Settings**.
Refer to {doc}`/guides/troubleshooting` for certificate regeneration, SAN mismatch,
401, and media-interruption diagnostics.

When microphone or camera state appears stuck, filter Console.app by
`category:MediaSession` to inspect interruption and recovery events. CoreAudio
`-50` and `FigAudioSession` `-19224` messages also appear during successful
microphone starts and are not failure indicators by themselves.

If microphone startup fails with
`io.livekit.swift-sdk Code=101 "Timed out"` after about five seconds, the
LiveKit recording engine did not produce the first buffer. Current builds
pre-warm that engine before publishing. After leaving XR, an iOS microphone
activation sound can play while capture restarts; this is expected.

### Native CloudXR

Native Apple clients require the CloudXR native device profile. Start the demo
from the repository root with:

```bash
NV_DEVICE_PROFILE=auto-native \
  uv run --project agent-samples/xr-render-demo xr_render_demo
```

The environment variable overrides the `auto-webrtc` YAML default without a
checkout edit. LiveKit remains on port 8080 while CloudXRKit uses its native
transport. Connecting or stopping XR does not disconnect the LiveKit agent
session. Closing the immersive space or disconnecting from the hub stops the XR
session so no render component is orphaned.

In `auto-native` mode, CloudXR does not use the web client's port 48322 WSS
proxy and needs no additional certificate installation. CloudXRKit's native
transport provides its own encryption without a user-facing trust prompt. The
hub certificate on port 8080 remains required for the LiveKit channel.

## Native C++

The C++20 sample ships a working backend for the LiveKit C++ SDK v0.4.1.

```bash
cd client-samples/native
cmake -S . -B build -DLIVEKIT_SDK_ROOT=/path/to/livekit-cpp-sdk
cmake --build build
./build/bin/streamkit_sample --host 192.168.1.100 --port 8080 \
  --secure --token <jwt>
```

Install the hub certificate in the native host's trust store before connecting.
Refer to {ref}`linux-native-certificate-trust` for the Ubuntu and Debian
procedure. The command uses the supported TLS proxy on port 8080. Insecure port
7880 is a direct LiveKit debugging path; do not expose it outside the XR AI host
or a trusted development network.

Without `LIVEKIT_SDK_ROOT`, it builds in stub mode and reports a connected state
without opening a network session. Build the standalone assertion-based tests
with `-DSTREAMKIT_BUILD_TESTS=ON` and run:

```bash
ctest --test-dir build --output-on-failure
```

| Test | Coverage |
|---|---|
| `streamkit_mapping_tests` | `ConnectionState` operations |
| `streamkit_agent_status_tests` | Canonical, missing, truncated, and empty `_agent.status` values |
| `streamkit_frame_sink_tests` | Move and span frame-injection overloads |
| `streamkit_audio_sink_tests` | Audio parameter forwarding and virtual dispatch |
| `streamkit_session_tests` | Mock-backed connection, media, data, status, and disconnection lifecycle |
| `streamkit_livekit_backend_tests` | State deduplication, disconnect races, stale metrics, and callback exceptions |

Run one test with CTest, invoke its binary directly, or rerun failures:

```bash
ctest --test-dir build -R streamkit_frame_sink_tests --output-on-failure
./build/StreamKit/Tests/StreamKitTests/streamkit_frame_sink_tests
ctest --test-dir build --rerun-failed --output-on-failure
```

The backend implements connection state, data, `_agent.status`, network
metrics, host-injected audio, and host-injected video. `StartAudio()` and
`StartCamera()` arm publication; the host opens its devices and pushes PCM or
frames through `AudioSink` and `FrameSink`. Camera facing and device ID are
therefore inert in this backend, while optional encoding settings apply before
the first frame creates the video track. Real-time producers must prefer the
`std::vector<uint8_t>&&` frame-injection overload; the span overload copies each
frame and can add about 1.4 MB of copying for a typical frame.

Token-URL HTTP fetching is not implemented; pass an inline token or subclass
`LiveKitBackend::FetchToken`. `GetRoom()` is the LiveKit-specific escape hatch
for remote audio rendering or AEC reference capture. Microphone processing
presets are not mapped because the C++ SDK exposes no corresponding source
controls.

(adding-a-client-for-a-new-platform)=
## Adding another platform

DeviceIOHub uses standard LiveKit. A new client fetches a token, joins through
the hub's port 8080, publishes microphone and optional camera tracks, plays the
remote audio track, and reads data topics such as `agent.response`. Keep the
transport-specific SDK behind a StreamKit backend and reuse the same status,
interruption, and participant-routing contracts.
