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
│       └── Backends/
│           ├── StreamingBackend.swift   # Protocol — implement to plug in any transport
│           ├── BackendConfiguration.swift  # enum { .liveKit(LiveKitConfig) }
│           └── LiveKit/
│               └── LiveKitBackend.swift # Built-in LiveKit WebRTC implementation
└── App/                # Sample app source files — add to your Xcode project
    ├── StreamKitSampleApp.swift
    ├── AppModel.swift
    ├── ContentView.swift
    └── ImmersiveView.swift   # visionOS only
```

The `StreamKit` library depends on an unmodified upstream
[livekit-client-sdk-swift](https://github.com/livekit/client-sdk-swift) checked out at
`../livekit-client-sdk-swift` (one level above this folder).

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

<!-- visionOS passthrough camera — requires Apple enterprise entitlement -->
<key>NSEnterpriseMCAMUsageDescription</key>
<string>Used to stream the main passthrough camera via ARKit.</string>
```

### 6. Entitlements file (visionOS passthrough camera — device only)

> **Enterprise license required.**  
> Access to the Apple Vision Pro main passthrough camera is an Apple enterprise API.
> You must request the entitlement from Apple before it will work on a physical device.
> All other features — audio, data channel, and the visionOS simulator — work without it.

#### Requesting the entitlement

1. Visit <https://developer.apple.com/contact/request/visionos-enterprise-api/> and
   submit a request for the **Main Camera Access** enterprise entitlement.
2. Apple will review your use case and, if approved, add the entitlement to your
   App ID in the developer portal.
3. Regenerate your provisioning profile after it is granted.

#### Adding it to the project

An entitlements file is already provided at `App/StreamKitSample.entitlements`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.developer.arkit.main-camera-access.allow</key>
    <true/>
</dict>
</plist>
```

Verify that **Build Settings → Code Signing Entitlements** for the visionOS target
points at this file. Without a matching provisioning profile from Apple the build will
succeed but `startCamera()` will throw an access-denied error at runtime on device.

The visionOS simulator does **not** require this entitlement and will always use the
GIF-based camera feed regardless.

### 7. Build and run

| Destination | Camera | Immersive Space | Microphone |
|---|---|---|---|
| **visionOS device** | ARKit passthrough — requires enterprise entitlement (see §6) | Supported — must be opened before starting the camera | Works |
| **visionOS Simulator** | Streams `SimulatorFeed.gif` (see below) | Not supported by the simulator — the UI row is hidden automatically | Works if the host platform has a mic |
| **iOS / iPadOS device** | `AVCaptureSession` front/back camera | N/A | Works |
| **iOS Simulator** | Streams `SimulatorFeed.gif` (see below) | N/A | Limited — WebRTC ADM may error; other features unaffected |

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

## Quick-start usage

```swift
import StreamKit

// 1. Create a session backed by LiveKit
let session = StreamSession(.liveKit(LiveKitConfig(
    host: "192.168.1.100",
    token: myJWT          // or: tokenURL: URL(string: "http://…/token")!
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
