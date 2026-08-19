<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Networking and firewall

XR-Media-Hub, CloudXR, and optional application viewers use the following
ports. Open only the externally reachable ports required by the deployment.

| Port | Protocol | Purpose |
|------|----------|---------|
| 7880 | TCP | LiveKit signaling (internal — bound to 127.0.0.1 via the hub's /rtc proxy; browsers and mobile clients do not connect here) |
| 7881 | TCP | LiveKit WebRTC TCP fallback (DTLS/SRTP — already encrypted) |
| 7882 | UDP | LiveKit WebRTC UDP media (DTLS/SRTP — already encrypted) |
| 8080 | TCP | Web client + token server + wss:// /rtc proxy (HTTPS — the single entry point for browser, Android, iOS, and visionOS clients) |
| 8092 | TCP | Optional live agent-event viewer (plain HTTP — bound to 127.0.0.1 by default; do not expose to an untrusted network) |
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

## Cloud VMs behind NAT

The signaling proxy on port 8080 does not proxy WebRTC media. LiveKit still
needs to advertise an ICE address that clients can reach on ports 7881 and
7882. On a cloud VM whose network interface has only a private address, enable
STUN-based public-IP discovery in the sample's `xr_media_hub.yaml`:

```yaml
lk_use_external_ip: true
```

LiveKit validates the discovered address with a self-ping before advertising
it. If the provider's NAT does not support that hairpin path, also skip the
validation:

```yaml
lk_use_external_ip: true
lk_skip_external_ip_validation: true
```

Skipping validation does not make a closed port reachable. Ensure the VM's
cloud firewall and host firewall allow 7881/TCP and 7882/UDP. Keep both options
disabled for local and private-network deployments.

## RHEL, Fedora, or CentOS (`firewall-cmd`)

```bash
sudo firewall-cmd --permanent --add-port=7881/tcp
sudo firewall-cmd --permanent --add-port=7882/udp
sudo firewall-cmd --permanent --add-port=8080/tcp
sudo firewall-cmd --permanent --add-port=48322/tcp
sudo firewall-cmd --reload
```

## Live agent-event viewer

The `WebEventsAgent` SDK defaults to `http://127.0.0.1:8092`. The lab-instrument
and tea-making samples override that listener to `0.0.0.0:8092`, making their
event pages directly reachable at `http://<xr-host>:8092`. Restrict TCP port
8092 to a trusted development network. For a loopback-bound deployment, use an
SSH tunnel:

```bash
ssh -L 8092:127.0.0.1:8092 user@xr-host
```

Then open `http://127.0.0.1:8092` locally.

For direct access on a trusted private network, configure the application-owned
viewer explicitly when registering it with the runtime:

```python
viewer = runtime.register(
    "web-events",
    WebEventsAgent(host="0.0.0.0", port=8092),
)
```

Restrict the firewall rule to the client subnet instead of opening the port to
every source. For example, with `ufw` and a `192.168.1.0/24` development
network:

```bash
sudo ufw allow from 192.168.1.0/24 to any port 8092 proto tcp
sudo ufw reload
```

Apply the equivalent source-restricted rule to the cloud security group when
the host is behind a provider firewall. The viewer rejects unrecognized HTTP
`Host` names to prevent DNS rebinding. Connect using a literal server address;
an authenticated reverse proxy can instead rewrite `Host` to `127.0.0.1`.
The listener currently accepts IPv4 addresses only.

```{warning}
The live event viewer does not provide authentication or TLS, and its payloads
can include transcripts and camera-derived text. Do not expose port 8092 to
the public Internet. Use the loopback binding with an SSH tunnel, or put the
viewer behind an authenticated TLS reverse proxy.
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
