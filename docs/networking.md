<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Networking and firewall

The hub and CloudXR runtime use the following ports. Open them permanently
if a firewall is active.

| Port | Protocol | Purpose |
|------|----------|---------|
| 7880 | TCP | LiveKit WebSocket signaling |
| 7881 | TCP | LiveKit WebRTC TCP fallback |
| 7882 | UDP | LiveKit WebRTC UDP media |
| 8080 | TCP | Web client / token server (HTTP) |
| 8443 | TCP | Web client / token server (HTTPS, if enabled) |
| 48322 | TCP | CloudXR WSS proxy (XR headset / client connection) |

## Ubuntu / Debian (`ufw`)

```bash
sudo ufw allow 7880/tcp
sudo ufw allow 7881/tcp
sudo ufw allow 7882/udp
sudo ufw allow 8080/tcp
sudo ufw allow 8443/tcp   # HTTPS only
sudo ufw allow 48322/tcp  # CloudXR (xr-render-demo)
sudo ufw reload
```

## RHEL / Fedora / CentOS (`firewall-cmd`)

```bash
sudo firewall-cmd --permanent --add-port=7880/tcp
sudo firewall-cmd --permanent --add-port=7881/tcp
sudo firewall-cmd --permanent --add-port=7882/udp
sudo firewall-cmd --permanent --add-port=8080/tcp
sudo firewall-cmd --permanent --add-port=8443/tcp   # HTTPS only
sudo firewall-cmd --permanent --add-port=48322/tcp  # CloudXR (xr-render-demo)
sudo firewall-cmd --reload
```

## TLS for the web client

Camera access in browsers is only permitted over `localhost` or HTTPS. When
connecting from another device on the network, enable TLS in
`xr_media_hub.yaml`:

```yaml
web_server_tls: true
web_server_port: 8443   # conventional HTTPS alt-port (optional)
```

On first run a self-signed certificate is generated at
`~/.local/share/xr-ai/web-server.crt`. To trust it:

- **Chrome / Edge**: navigate to `https://<host>:8443`, click **Advanced →
  Proceed to … (unsafe)**.
- **Firefox**: click **Advanced → Accept the Risk and Continue**.
- **iOS / Safari**: open the cert URL, follow the prompt to install the
  profile, then enable it under **Settings → General → VPN & Device
  Management**.

To use your own certificate, set `cert_file` and `key_file` in
`xr_media_hub.yaml`.

See also `docs/architecture.md` for the LiveKit `ws://` mixed-content
limitation.
