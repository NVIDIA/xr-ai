<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# ai-sdk-sample

A SwiftUI sample app for iOS and visionOS that demonstrates **StreamKit** — a thin,
backend-agnostic streaming SDK built on top of LiveKit WebRTC.

## Repository layout

```
ai-sdk-sample/
├── StreamKit/          # The SDK — add this as a local Swift Package in Xcode
│   ├── Package.swift
│   └── Sources/StreamKit/
│       ├── StreamSession.swift          # Public façade (@MainActor, ObservableObject)
│       ├── ConnectionState.swift
│       ├── StreamError.swift
│       ├── Config/
│       │   ├── SessionConfig.swift      # Room name, identity, audio + camera settings
│       │   ├── AudioConfig.swift        # Voice processing / software DSP / raw / disabled
│       │   └── CameraConfig.swift       # Resolution, fps, camera position (iOS only)
│       ├── MediaSessionDiagnostics.swift # Media route / capture diagnostics
│       └── Backends/
│           ├── StreamingBackend.swift   # Protocol (send(_:reliable:topic:) is topic-aware)
│           ├── BackendConfiguration.swift  # enum { .liveKit(LiveKitConfig) }
│           └── LiveKit/
│               └── LiveKitBackend.swift # LiveKit WebRTC backend; prepareAudio() launch hook
└── App/                # Sample app source files — add to your Xcode project
    ├── StreamKitSampleApp.swift
    ├── AppModel.swift
    ├── ContentView.swift
    └── ImmersiveView.swift   # visionOS only: also hosts the CloudXR render surface
```

The `StreamKit` library depends on an unmodified upstream
[livekit-client-sdk-swift](https://github.com/livekit/client-sdk-swift), pinned to
`2.13.0` and resolved directly from GitHub by Swift Package Manager (see
`StreamKit/Package.swift`). No local checkout is required.

---

## Creating the Xcode project

### 1. New project

Open Xcode → **File → New → Project → Multiplatform → App**

| Field | Value |
|---|---|
| Product Name | `StreamKitSample` |
| Interface | SwiftUI |
| Language | Swift |

### 2. Add destinations

Select the project root in the navigator → **Supported Destinations → +**

- Add **visionOS**
- Remove **macOS** if it was added automatically

You should be left with **iOS** and **visionOS**.

### 3. Add the StreamKit package

**File → Add Package Dependencies… → Add Local…**

Navigate to `ai-sdk-sample/StreamKit/` and click **Add Package**. In the dialog that
follows, tick **StreamKit** and confirm your app target is selected.

### 4. Replace the generated source files

Xcode auto-generates a `ContentView.swift` and an app entry point. Delete both, then
drag the four files from `ai-sdk-sample/App/` into the project navigator:

- `StreamKitSampleApp.swift`
- `AppModel.swift`
- `ContentView.swift`
- `ImmersiveView.swift`

When prompted: **Copy items if needed → unchecked**, both iOS and visionOS targets
checked.

### 5. Info.plist entries

Add the following keys to your app's `Info.plist` (or the equivalent entries in the
target's **Info** tab):

```xml
<!-- Microphone — required for LiveKit audio -->
<key>NSMicrophoneUsageDescription</key>
<string>Used to stream microphone audio.</string>

<!-- Camera — required for iOS AVCaptureSession -->
<key>NSCameraUsageDescription</key>
<string>Used to stream the camera feed.</string>

<!-- visionOS passthrough camera — requires Apple enterprise entitlement (see §6) -->
<key>NSMainCameraUsageDescription</key>
<string>Used to stream the main passthrough camera via ARKit.</string>
```

### 6. visionOS passthrough camera — device only

Access to the Apple Vision Pro main passthrough camera is an Apple **enterprise** API.
Two things are required; without both the camera APIs are silent no-ops at
runtime (`CameraVideoFormat.supportedVideoFormats(...)` returns `[]`). All
other features — audio, data channel, and the visionOS simulator — work
without any of this.

| # | What | Where |
|---|---|---|
| 1 | Entitlement key in the signed binary | `App/StreamKitSample.entitlements` declares `com.apple.developer.arkit.main-camera-access.allow`; wired in via the project's `CODE_SIGN_ENTITLEMENTS` build setting |
| 2 | The team's `Enterprise.license` bundled into the `.app` | See below |

Xcode's automatic signing works for development builds. App Store / TestFlight
distribution requires a manually-issued provisioning profile that grants the
entitlement.

> **Bundle ID note**: this sample's Bundle ID is `com.nvidia.xr-ai-example`.
> If you fork it, change the ID under Signing & Capabilities to one your team
> owns. The display name (`StreamKitSample`) is independent of the Bundle ID
> and is unchanged. `UserDefaults` are keyed by Bundle ID, so saved settings
> reset on first launch after a rename.

#### Bundling `Enterprise.license`

The Enterprise license is issued by Apple, per team. Apple's terms restrict
redistribution, so it is **gitignored** (`**/Enterprise.license`) and never
committed. A placeholder at `App/Enterprise.license.sample` documents the
location.

Place your team's license at:

```
client-samples/ios-visionos/App/Enterprise.license
```

A "Copy Enterprise.license" build phase copies it into the `.app` at build
time; visionOS auto-loads it from the bundle. If the file is missing, the
build still succeeds with a warning and every feature except main-camera
passthrough works.

If you prefer to keep the file outside the repo, symlink it (the gitignore
rule still matches):

```bash
ln -s ~/wherever/Enterprise.license client-samples/ios-visionos/App/Enterprise.license
```

The visionOS simulator does **not** require any of this and will always use the
GIF-based camera feed regardless.

### 7. Build and run

| Destination | Camera | Immersive Space | Microphone | In-app camera preview |
|---|---|---|---|---|
| **visionOS device** | ARKit passthrough — requires enterprise entitlement (see §6) | Supported — must be opened before starting the camera | Works | **Placeholder only** — ARKit frames bypass the SwiftUI sink, only the `LIVE` badge indicates capture |
| **visionOS Simulator** | Streams `SimulatorFeed.gif` (see below) | Not supported by the simulator — the UI row is hidden automatically | Works if the host platform has a mic | Live GIF preview |
| **iOS / iPadOS device** | `AVCaptureSession` front/back camera | N/A | Works | Live preview |
| **iOS Simulator** | Streams `SimulatorFeed.gif` (see below) | N/A | Limited — WebRTC ADM may error; other features unaffected | Live GIF preview |

The camera preview card at the top of `ContentView` mirrors the web client's
`<video>` element. It is wired through StreamKit's `CameraPreviewView`, which
wraps the LiveKit Swift `SwiftUIVideoView`. The card's aspect ratio follows
the live capture dimensions (so a portrait phone camera renders as 9:16 and
a landscape sensor as 16:9), with a 16:9 fallback before the first frame
arrives; its width is capped so the Agent panel below stays visible without
scrolling. On visionOS device builds the ARKit main-camera passthrough track
is forwarded straight to LiveKit's WebRTC pipeline and is not surfaced
through a 2D video sink; the preview card stays on its "Camera off"
placeholder while the `LIVE` badge signals active capture.

#### Simulator camera feed

On both the iOS and visionOS simulators there is no physical camera. Instead, the SDK
streams an animated GIF bundled inside the package:

```
StreamKit/Sources/StreamKit/Resources/SimulatorFeed.gif
```

To use a custom feed, replace that file with any animated GIF of the same name before
building. No code changes are required — the file is declared as a Swift Package
resource and loaded automatically at runtime.

---

## Trusting the hub's self-signed cert (one-time per device)

The hub ships TLS-on-by-default with a self-signed cert, and the LiveKit
Swift SDK's `URLSession` does not expose a server-trust hook — so the wss
handshake fails until iOS trusts the cert. The `TrustingSessionDelegate`
inside `LiveKitBackend.swift` only covers the `/token` HTTP fetch, not the
LiveKit WebSocket. Install the cert once:

1. In the app's Connection section, enter the hub host and port, then tap
   **Install hub certificate**. This opens Safari at
   `https://<host>:<port>/cert`.
2. Safari shows "Not Private" — tap **Show Details → visit this website**.
3. iOS prompts: **Download Configuration Profile**. Tap **Allow**.
4. Open **Settings → General → VPN & Device Management**, tap the
   downloaded profile under "Downloaded Profile", then **Install** (top
   right) and enter your passcode.
5. Open **Settings → General → About → Certificate Trust Settings** and
   toggle **Enable Full Trust** for the new cert.

The connection now completes without warnings. To switch hubs, repeat for
each new host or replace the auto-generated cert with one from a public
CA (`cert_file` / `key_file` in `xr_media_hub.yaml`).

### "Enable Full Trust" toggle does not appear

iOS only exposes the Certificate Trust Settings toggle for certs marked
`BasicConstraints CA:TRUE`. Older xr-ai builds generated a non-CA cert
and the toggle never appeared no matter how cleanly the profile was
installed.

**Fix:** the hub now auto-regenerates the cert as a self-signed CA on
next start if the cached one isn't already CA-marked. The recovery flow
is:

1. On each device that has the old profile, **Settings → General → VPN
   & Device Management** → tap the installed profile → **Remove
   Profile**.
2. On the server, restart the hub. It logs `TLS: cached cert is not a
   CA cert — regenerating…` and writes a new
   `~/.local/share/xr-ai/web-server.crt`. (If you want to force the
   regen explicitly, delete `~/.local/share/xr-ai/web-server.crt` and
   `~/.local/share/xr-ai/web-server.key` first.)
3. Re-open `https://<host>:8080/cert` on the device and follow the
   install steps above. The new cert appears under **Certificate Trust
   Settings** with the Full Trust toggle exposed.

### Connection fails with `errSSLBadCert` / `-1202` after the cert is trusted

If you installed the profile and toggled Full Trust on but the wss
handshake still errors out with NSURLErrorDomain `-1202` and a message
like *"pretending to be 10.29.90.196"*, the cert's SubjectAlternativeName
does not cover the IP you're typing into the app. This happens when the
hub generated the cert before that interface was up, or via an
`/etc/hosts` loopback alias (the Ubuntu default of `127.0.1.1` instead of
the LAN IP).

**Fix:** the hub now probes the kernel's outbound IPv4 addresses at cert
load and regenerates the cert whenever its SAN is missing a current local
IP. Restart the hub — it logs `TLS: cached cert SAN is missing local
IP(s) [10.29.90.196] — regenerating…` — then on the device remove the
old profile under **VPN & Device Management** and reinstall from
`https://<host>:8080/cert` exactly as above.

### TLS succeeds but the room rejects the token with 401

If the cert is trusted (no `-1202` error) but the room connection still
fails immediately with HTTP 401 / "no permissions to access the room",
the hub's wss /rtc proxy is dropping the `Authorization: Bearer <token>`
header the Swift SDK sends. The JS SDK puts the JWT in the query string,
so the web client never hit this code path; older proxy builds didn't
forward request headers.

**Fix:** purely server-side — pull the latest hub and restart. The proxy
now forwards `Authorization` (and every other end-to-end header) on both
the `/rtc/validate` HTTP shim and the `/rtc[/<version>]` WebSocket. No
client-side action needed.

### Microphone fails to start with a "Timed out" error

If tapping **Start** under Microphone occasionally fails — the SDK throws
`io.livekit.swift-sdk Code=101 "Timed out"` after ~5s — the LiveKit recording
engine didn't start, so the publish's frame watcher never saw a buffer. The
CoreAudio `-50` and `FigAudioSession err=-19224` lines that show up around the
same time are benign; they appear on successful starts too.

**Fix:** `LiveKitBackend.startAudio` pre-warms the recording engine
(`AudioManager.shared.setRecordingAlwaysPreparedMode(true)`) before publishing,
so a frame is available immediately and the publish completes. No client action
needed — it's built in.

### Orange mic indicator stays lit after stopping audio

Prepared mode keeps the engine input hot for fast re-enable, which the OS reads
as the mic still being in use. `LiveKitBackend.stopAudio` now drops prepared
mode and disables engine *input* while leaving *output* up (so the agent can
still be heard); the dot clears while you stay connected, and is fully released
on disconnect.

### Mic / camera go dead while the UI still says "on"

The OS can interrupt the microphone or camera for many reasons (phone call,
Siri, another app, route change, iPad multitasking), and LiveKit doesn't recover
on its own. The sample re-arms capture automatically whenever you'd left the mic
on, so it usually comes back with no action from you.

If it doesn't, filter Console.app with `category:MediaSession` (works untethered)
to see the interruption and recovery events. On XR exit you'll hear iOS's
mic-activation sound as capture restarts — that's expected, not a malfunction.

---

## Launching XR (CloudXR)

The sample ships a CloudXR render stream alongside the LiveKit agent
channel, the same pattern as the web-xr client. The **XR Stream** row in
the Media section (under the mic and camera buttons) connects directly
to the CloudXR runtime on the same host as the LiveKit hub (the
Connection section's **Host / IP**). `cloudxr_runtime` always runs
alongside the hub in this repo's stack.

XR support is currently limited to Apple Vision Pro, so the **XR Stream** row
(the Start XR control) ships on **visionOS**. iOS / iPadOS builds run every
other feature (agent, mic, camera, data) without an XR path.

### Two parallel transports

| Channel | Goes through | Port |
|---|---|---|
| Agent (mic, data, camera) | LiveKit via the hub's wss proxy | `8080` |
| CloudXR (XR render frames) | CloudXRKit native transport | CloudXR-managed |

The two are independent: connecting/disconnecting CloudXR never drops
the LiveKit room. When the CloudXR session reaches `.connected` the app
publishes an empty data message on the `xr.session.started` LiveKit
topic. `render-mcp` gates the LOVR launch on that signal exactly the
same way it does for the web client.

### Server prerequisite: change `NV_DEVICE_PROFILE` to `auto-native`

The committed `agent-samples/xr-render-demo/yaml/cloudxr_runtime.yaml`
defaults to `auto-webrtc` so the web-xr client works out of the box.
Native Apple clients need the proprietary CloudXR transport, which is
**mutually exclusive** with the WebRTC path per run.

To run this app against `xr-render-demo`:

1. Edit `agent-samples/xr-render-demo/yaml/cloudxr_runtime.yaml` on the
   server checkout, change one line:
   ```yaml
   cloudxr_env:
     NV_DEVICE_PROFILE: auto-native
   ```
2. Restart the stack: `uv run xr_render_demo`.
3. Revert that line when you go back to web-xr.

No code changes anywhere else on the server.

### CloudXRKit SPM dependency

The Xcode project already pins
[`NVIDIA/cloudxr-framework`](https://github.com/NVIDIA/cloudxr-framework)
(`CloudXRKit` product) in `StreamKitSample.xcodeproj/project.pbxproj`
and `Package.resolved`. Opening the workspace resolves the binary
xcframeworks on first launch. No manual **File → Add Package
Dependencies…** step.

### Apple Developer Program

The Vision Pro low-latency CloudXR path requires the
`com.apple.developer.low-latency-streaming` entitlement, which only
provisions when the build is signed by a team enrolled in the Apple
Developer Program. The entitlement is committed in
`App/StreamKitSample.entitlements`; non-ADP teams can clear it locally
(higher latency, otherwise works).

### On-device flow

1. Connect to the hub (Connection section → **Connect**). Mic / data /
   camera work as before.
2. Tap **Launch XR**. The CloudXR session connects directly to the
   Connection host; once it reaches `Streaming` the agent worker
   receives the `xr.session.started` signal and `render-mcp` launches
   LOVR.
3. Tap **Stop** to disconnect CloudXR while leaving the LiveKit room
   running.

On visionOS, **Launch XR** auto-opens the immersive space if it isn't
already open. No manual **Open Space** step required.

The session is also torn down automatically when its render target
disappears:

Closing the immersive space (**Close Space**, or the system home
gesture) calls `stopXR()`. The `CloudXRSessionComponent` would
otherwise be orphaned.

`disconnect()` from the LiveKit section also stops XR first, so leaving
the hub never leaves a dangling CloudXR session behind.

### Cert / trust notes

- The hub cert install on `:8080` (see the section above) is still
  required for the LiveKit channel.
- The `:48322` WSS proxy that the web-xr client uses is **unused** in
  `auto-native` mode. No extra cert install is needed for CloudXR.
- CloudXR has its own transport encryption; the SDK handles it without
  user-facing trust prompts on the LAN.

### Render target

The open `ImmersiveSpace` hosts the `CloudXRSessionComponent`. The
placeholder demo sphere hides itself while streaming.

---

## Quick-start usage

```swift
import StreamKit

// 1. Create a session backed by LiveKit
let session = StreamSession(.liveKit(LiveKitConfig(
    host: "192.168.1.100",
    token: myJWT          // or: tokenURL: URL(string: "https://…/token")!
)))

// 2. Connect (room name + identity are in SessionConfig)
try await session.connect(config: SessionConfig(roomName: "demo", identity: "ipad-1"))

// 3. Receive data
session.onDataReceived = { data in
    print("received \(data.count) bytes")
}

// 4. Send data
try await session.send(Data("hello".utf8))

// 5. Start camera
// visionOS: open your ImmersiveSpace first, then:
try await session.startCamera()
```

---

## Adding a custom backend

Conform to `StreamingBackend` and pass your instance to `StreamSession(backend:)`:

```swift
final class MyBackend: StreamingBackend {

    var onConnectionStateChanged: (@Sendable (ConnectionState) -> Void)?
    var onDataReceived: (@Sendable (Data) -> Void)?

    func connect(config: SessionConfig) async throws {
        // establish your connection …
        onConnectionStateChanged?(.connected)
    }

    func disconnect() async { … }
    func startCamera() async throws { … }
    func stopCamera() async throws { … }
    func send(_ data: Data, reliable: Bool) async throws { … }
}

let session = StreamSession(backend: MyBackend())
```

The `StreamSession` API and all app-level code above it remain unchanged regardless
of which backend is in use.

---

## Token server (LiveKit)

LiveKit requires a signed JWT. A minimal Python token server:

```python
# pip install livekit
from livekit import api
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.get("/token")
def token():
    t = (
        api.AccessToken("devkey", "secret")
           .with_grants(api.VideoGrants(room_join=True, room=request.args["room"]))
           .with_identity(request.args["identity"])
    )
    return jsonify({"token": t.to_jwt()})
```

Pass the endpoint URL to `LiveKitConfig(host:tokenURL:)` and the SDK appends
`?room=…&identity=…` automatically.
