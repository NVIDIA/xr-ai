<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Networking and firewall

The XR-Media-Hub and CloudXR runtime use the following ports. Open them permanently
if a firewall is active.

| Port | Protocol | Purpose |
|------|----------|---------|
| 7880 | TCP | LiveKit signaling (internal — bound to 127.0.0.1 via the hub's /rtc proxy; browsers and mobile clients do not connect here) |
| 7881 | TCP | LiveKit WebRTC TCP fallback (DTLS/SRTP — already encrypted) |
| 7882 | UDP | LiveKit WebRTC UDP media (DTLS/SRTP — already encrypted) |
| 8080 | TCP | Web client + token server + wss:// /rtc proxy (HTTPS — the single entry point for browser, Android, iOS, and visionOS clients) |
| 48322 | TCP | CloudXR WSS proxy (XR headset or client connection) |

## Ubuntu or Debian (`ufw`)

```bash
sudo ufw allow 7881/tcp     # WebRTC TCP fallback
sudo ufw allow 7882/udp     # WebRTC UDP media
sudo ufw allow 8080/tcp     # https + wss entry point
sudo ufw allow 48322/tcp    # CloudXR (xr-render-demo)
sudo ufw reload
```

7880 stays on `127.0.0.1`; do not expose it externally — browsers and
mobile clients reach LiveKit through the same-origin `wss://<host>:8080/rtc`
proxy, not directly.

## RHEL, Fedora, or CentOS (`firewall-cmd`)

```bash
sudo firewall-cmd --permanent --add-port=7881/tcp
sudo firewall-cmd --permanent --add-port=7882/udp
sudo firewall-cmd --permanent --add-port=8080/tcp
sudo firewall-cmd --permanent --add-port=48322/tcp
sudo firewall-cmd --reload
```

## TLS for the web client

TLS is **on by default** — `web_server_tls: true` is the built-in default.
The web server terminates HTTPS on `web_server_port` (8080 by default) and
also exposes a same-origin `wss://<host>:8080/rtc` proxy that forwards
LiveKit signaling to the internal plaintext port. This is the only path
browser, Android, iOS, and visionOS clients use; LiveKit's native 7880 is
never reached directly by client traffic.

On first run a self-signed certificate is generated at
`~/.local/share/xr-ai/web-server.crt`. To use your own, set `cert_file`
and `key_file` in `xr_media_hub.yaml`.

To **disable** TLS for `localhost`-only dev where the certificate warning is
noise, set `web_server_tls: false`. With TLS off, the same-origin proxy
serves plain `ws://` instead of `wss://`, and `localhost` is the only
context where camera and mic permissions are granted without HTTPS.

To **trust the self-signed certificate** so you stop seeing the warning:

- **Chrome or Edge**: navigate to `https://<host>:8080`, click **Advanced →
  Proceed to … (unsafe)**.
- **Firefox**: click **Advanced → Accept the Risk and Continue**.
- **Android**: tap **Install hub certificate** in the app's Connection
  section (visible before the first connection). The app fetches the
  certificate from `https://<host>:<port>/cert` and opens the system install
  dialog. After confirming, connect normally — the LiveKit SDK validates
  against the system + user CA store automatically.

- **iOS, iPadOS, and visionOS**: tap **Install hub certificate** in the
  app's Connection section. This opens Safari at
  `https://<host>:<port>/cert`. In Safari: tap **Show Details → visit
  this website** past the certificate warning → **Download Configuration
  Profile** → **Allow** → install via **Settings → General → VPN &
  Device Management** → enable **Settings → General → About →
  Certificate Trust Settings → Enable Full Trust** for the new certificate.

```{warning}
On iOS, this step is **mandatory**: the LiveKit Swift SDK's `URLSession`
does not expose a server-trust auth-challenge hook, and ATS does not
bypass certificate-chain validation regardless of `NSAllowsArbitraryLoads`.
Until the certificate is trusted at the OS level, the wss handshake fails.
```

Production deployments on any platform should replace the auto-generated
certificate with one from a public CA via `cert_file` and `key_file` in
`xr_media_hub.yaml`.
