<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Networking and firewall

DeviceIOHub, CloudXR, and optional application viewers use the following
ports. Open only the externally reachable ports required by the deployment.

| Port | Protocol | Purpose |
|------|----------|---------|
| 7880 | TCP | Plaintext LiveKit signaling (used behind the hub's `/rtc` proxy; optional direct native-client debugging path) |
| 7881 | TCP | LiveKit WebRTC TCP fallback (DTLS/SRTP — already encrypted) |
| 7882 | UDP | LiveKit WebRTC UDP media (DTLS/SRTP — already encrypted) |
| 8080 | TCP | Web client, token server, and `wss://` `/rtc` proxy (HTTPS; the normal LiveKit signaling entry point for shipped clients) |
| 8092 | TCP | Optional live agent-event viewer (plain HTTP — bound to 127.0.0.1 by default; do not expose to an untrusted network) |
| 48322 | TCP | CloudXR WSS proxy (WebRTC profiles; unused by native Apple clients) |

## Ubuntu or Debian (`ufw`)

```bash
sudo ufw allow 7881/tcp     # WebRTC TCP fallback
sudo ufw allow 7882/udp     # WebRTC UDP media
sudo ufw allow 8080/tcp     # https + wss entry point
sudo ufw allow 48322/tcp    # CloudXR (xr-render-demo)
sudo ufw reload
```

These rules intentionally omit port 7880. The generated LiveKit configuration
does not set `bind_addresses`, and the container uses host networking, so the
plaintext listener can accept connections on the host's network interfaces.
Merely omitting an allow rule protects it only when the host firewall and any
cloud security group reject other inbound traffic. Browsers and mobile clients
must use the same-origin `wss://<host>:8080/rtc` proxy.

The native C++ executable defaults to port 7880 for direct, insecure debugging.
Keep that traffic on the XR AI host whenever possible. If a native client must
connect directly from a trusted development network, restrict access to its
source subnet. For example, with a `192.168.1.0/24` development network:

```bash
sudo ufw allow from 192.168.1.0/24 to any port 7880 proto tcp
sudo ufw reload
```

Apply the equivalent source restriction to the cloud security group. Do not
expose port 7880 to the public Internet: it carries plaintext signaling, and a
LiveKit bearer token authenticates a client but does not encrypt the transport.

## Cloud VMs behind NAT

The signaling proxy on port 8080 does not proxy WebRTC media. LiveKit still
needs to advertise an ICE address that clients can reach on ports 7881 and
7882. On a cloud VM whose network interface has only a private address, enable
STUN-based public-IP discovery in the sample's `device_io_hub.yaml`:

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

NAT also affects the TLS certificate: clients dial the VM's public IP, which
must be listed in the certificate's SAN via `web_server_extra_sans`. Refer to
[TLS for the web client](#tls-for-the-web-client) below.

## RHEL, Fedora, or CentOS (`firewall-cmd`)

```bash
sudo firewall-cmd --permanent --add-port=7881/tcp
sudo firewall-cmd --permanent --add-port=7882/udp
sudo firewall-cmd --permanent --add-port=8080/tcp
sudo firewall-cmd --permanent --add-port=48322/tcp
sudo firewall-cmd --reload
```

## Live agent-event viewer

The `WebEventsAgent` SDK and shipped samples default to
`http://127.0.0.1:8092`. For access from another machine, keep the loopback
binding and use an SSH tunnel:

```bash
ssh -L 8092:127.0.0.1:8092 user@xr-host
```

Then open `http://127.0.0.1:8092` locally.

For direct access on a trusted private network, pass `--expose-web-events` to
the lab-instrument or tea-making sample. This binds the viewer to
`0.0.0.0:8092`. Applications can configure the same behavior when registering
their viewer with the runtime:

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

(tls-for-the-web-client)=
## TLS for clients

TLS is **on by default** — `web_server_tls: true` is the built-in default.
The web server terminates HTTPS on `web_server_port` (8080 by default) and
also exposes a same-origin `wss://<host>:8080/rtc` proxy that forwards
LiveKit signaling to the internal plaintext port. Browsers and mobile clients
use only this path. It is also the supported native C++ path when the executable
runs with `--secure --port 8080`; the native executable's port 7880 default is
an insecure debugging exception.

On first run DeviceIOHub generates a stable development root CA and a server
leaf signed by that root under `~/.local/share/xr-ai/`:

- `root-ca.crt` and `root-ca.key` are the stable trust anchor. Only the public
  certificate is available from `/cert`; the private key never leaves the hub.
- `web-server.crt` and `web-server.key` are the CA-signed server leaf and its
  private key. The web server presents this leaf for HTTPS and WSS.

To use your own certificate, set `cert_file` and `key_file` in
`device_io_hub.yaml`. Externally managed TLS continues to work unchanged;
`/cert` returns 404 because DeviceIOHub does not own that certificate's root.

The generated leaf covers `localhost`, the hostname, and automatically
discovered local IPv4 addresses. When clients dial an address that discovery
misses, or one that is not local (the public IP of a NAT'd cloud VM such as Brev,
a forwarding proxy's address, or a DNS name), list it in `device_io_hub.yaml` and
the leaf is regenerated to include it on the next hub start:

```yaml
web_server_extra_sans:
  - 203.0.113.7
  - hub.example.com
```

Changing these addresses never rotates `root-ca.crt`, so clients do not need to
reinstall the root after a Brev public-IP change.

To **disable** TLS for `localhost`-only dev where the certificate warning is
noise, set `web_server_tls: false`. With TLS off, the same-origin proxy
serves plain `ws://` instead of `wss://`, and `localhost` is the only
context where camera and mic permissions are granted without HTTPS.

To **trust the development root CA** so you stop seeing the warning:

- **Chrome or Edge**: navigate to `https://<host>:8080`, click **Advanced →
  Proceed to … (unsafe)**.
- **Firefox**: click **Advanced → Accept the Risk and Continue**.
- **Android**: tap **Install hub certificate** in the app's Connection
  section (visible before the first connection). The app fetches the
  public root CA from `https://<host>:<port>/cert` and opens the system install
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

(linux-native-certificate-trust)=
### Linux native certificate trust

The native C++ client validates the hub through the Linux system CA bundle.
Before connecting, ensure the address passed to `--host` appears in the server
leaf's subject alternative names. If the native client uses another IP address
or DNS name, configure `web_server_extra_sans` and restart the hub to regenerate
only the leaf and leaf key.

Copy the resulting `~/.local/share/xr-ai/root-ca.crt` from the hub
host to the native client host. On Ubuntu or Debian, install that copy and
refresh the bundle:

```bash
sudo install -m 0644 /path/to/root-ca.crt \
  /usr/local/share/ca-certificates/xr-ai-hub.crt
sudo update-ca-certificates
```

Restart the native client after updating the bundle. Later SAN changes rotate
only the server leaf and do not require another trust installation.

For production deployments on any platform, replace the auto-generated
certificate with one from a public CA by setting `cert_file` and `key_file` in
`device_io_hub.yaml`.
