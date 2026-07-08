<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Android StreamKit Sample

Android client for xr-ai, providing feature parity with the web and iOS/visionOS clients.

## Feature set

Identical to the web client:

| Feature | Details |
|---|---|
| Connection | Host / IP, port, JWT token or token-server URL, participant identity |
| Audio | Start / stop microphone; three modes (Voice Processing, Software AEC, Raw) |
| Camera | Start / stop camera; selector auto-populated from Camera2 (front, back, any extra lens, USB cameras) |
| Agent status | Live badge (`idle` / `processing`) driven by the `_agent.status` channel |
| Data channel | Send Ping, send arbitrary UTF-8 messages |
| Received messages | Scrollable list with per-message timestamps |
| Error display | Snackbar toast, auto-dismissed after 4 s |

## Architecture

```
app/src/main/java/com/nvidia/xrai/streamkitsample/
├── streamkit/                      ← StreamKit library (mirrors web StreamKit/ + Swift StreamKit/)
│   ├── ConnectionState.kt
│   ├── StreamError.kt
│   ├── StreamSession.kt            ← public API — transport-agnostic
│   ├── config/
│   │   ├── AudioConfig.kt
│   │   ├── BackendConfiguration.kt ← LiveKitConfig lives here
│   │   ├── CameraConfig.kt
│   │   └── SessionConfig.kt
│   └── backends/
│       ├── StreamingBackend.kt     ← interface (custom backends plug in here)
│       └── livekit/
│           └── LiveKitBackend.kt   ← LiveKit Android SDK implementation
├── AppViewModel.kt                 ← observable state + actions (mirrors AppModel.swift / app.js model)
├── CertInstall.kt                  ← one-tap cert fetch + KeyChain install intent
└── MainActivity.kt                 ← Jetpack Compose UI
```

`StreamKit` is structured identically to the web (`client-samples/web/StreamKit/`) and
iOS/visionOS (`client-samples/ios-visionos/StreamKit/`) libraries:
- **`StreamSession`** — single public entry-point; wraps any `StreamingBackend`.
- **`LiveKitBackend`** — the only file that imports `io.livekit:livekit-android`.
- **`BackendConfiguration`** — selects the backend; custom backends bypass it entirely.

Remote audio is played automatically by the LiveKit Android SDK once a remote
participant publishes an audio track — no explicit attachment step is needed.

## Setup

### Requirements

| Tool | Minimum version |
|---|---|
| Android Studio | Hedgehog (2023.1.1) or later |
| JDK | 17 |
| Android Gradle Plugin | 8.5 |
| Kotlin | 2.4 |
| Min Android SDK | API 24 (Android 7.0) |
| Target Android SDK | API 34 (Android 14) |

### Open in Android Studio

1. Open Android Studio.
2. **File → Open** → select `client-samples/android/`.
3. Let Gradle sync finish (downloads ~300 MB of dependencies on first run).
4. Select a device or emulator running API 24+, then **Run ▶**.

For command-line builds, run:

```bash
./gradlew assembleDebug
```

### Connecting to the server

Start any agent sample in one command:

```bash
cd xr-ai/agent-samples/echo-agent
uv sync && uv run echo_agent
```

The hub prints:

```
[hub]   LiveKit URL : wss://0.0.0.0:8080
[hub]   Token       : eyJ…   ← paste into the app
```

In the app:

| Field | Value |
|---|---|
| Host / IP | IP of the machine running the server |
| Port | `8080` (the hub web-server port, not LiveKit's internal 7880) |
| Token | Paste the printed JWT |
| Identity | Any string unique to this device |

Leave **Token URL** blank to use the server's default `/token` endpoint, or paste
the printed JWT directly into **Token**.

The hub serves a self-signed cert by default. Before connecting for the first
time, tap **Install hub certificate** in the Connection section. The app fetches
the cert from the hub and opens the system cert-install dialog — confirm to
install. After install, connect normally; Android validates against the system +
user CA store automatically.

## Permissions

The app requests the following Android permissions:

| Permission | When requested |
|---|---|
| `INTERNET` | Always (network) |
| `RECORD_AUDIO` | On first tap of **Start Microphone** |
| `CAMERA` | On first tap of **Start Camera** |
| `MODIFY_AUDIO_SETTINGS` | Declared; granted at install |
| `BLUETOOTH_CONNECT` | Declared; granted at install (audio routing) |

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| `io.livekit:livekit-android` | 2.7.0 | WebRTC transport |
| Jetpack Compose BOM | 2024.11.00 | UI framework |
| `androidx.lifecycle:lifecycle-viewmodel-compose` | 2.8.7 | ViewModel + Compose integration |
| `androidx.activity:activity-compose` | 1.9.3 | `ComponentActivity` Compose entry-point |
