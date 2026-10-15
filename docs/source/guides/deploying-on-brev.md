<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Deploying on Brev or your own host

XR AI Experiences demonstrates deploying samples on one Linux GPU host and
operating them through a web UI. For cloud use, the recommended entry point is
the [XR AI Experiences Brev Launchable](https://brev.nvidia.com/launchable/deploy/now?launchableID=env-3I1iZdrqmkxKSgsIsMXvqu8xrp7).
Developers can use the same launcher on a prepared local host to exercise the
deployment workflow before testing on a cloud VM.

## What this adds to a sample command

Use a sample's documented `uv run …` command when developing that sample directly.
Use `deployments/brev/launch.sh setup` when you want the deployment UI and lifecycle:

| Direct sample workflow | Experiences launcher |
|---|---|
| Start the required model stack and sample yourself | Start the selected model stack, then its sample, in the background |
| Work from terminal commands and output | Choose samples, inspect status, and read logs in a web UI |
| Stop the previous stack before switching | Stop the recorded active stack before deploying another sample |
| Supply credentials and client network configuration | Persist the HF token privately, configure reachable IP and ports, and provide client links |
| Use the sample's existing certificate behavior | Supply public or LAN IP SANs to XR AI's certificate generation; display the root fingerprint |

The launcher delegates models, GPU profiles, and processes to XR AI; it is not a
second agent framework. Local use runs real samples and models, not a mock, and
exercises the same single-host startup and switching flow. It does not reproduce
cloud NAT, provider firewalls, Brev Secure Links, or GPU provisioning. Those still
need validation on a cloud VM. This is a trusted-host demo administration UI, not
a production multi-tenant deployment platform.

## Run on Brev

Launch the VM through the Launchable, then use its configured Secure Link for the
chooser. Brev authenticates access through the Secure Link; no second chooser
login is needed when proxy authentication is configured below. Sample-client
connections also require the provider network configuration described under
Networking and clients; a Secure Link for the chooser does not tunnel media ports.

For Launchable maintainers: fetch a trusted XR AI revision and invoke
the following from that checkout after configuring an authenticated Secure Link
to port 8091:

```bash
XR_AI_UI_AUTH=proxy XR_AI_TRUST_PROXY=1 ./deployments/brev/launch.sh setup
```

Keep the implementation and UI here rather than copying them into the Launchable.
The Launchable owns
Brev-specific image selection, prerequisites, firewall policy, and Secure Link
configuration. Any driver installation, reboot, or boot-service provisioning
belongs to that explicit host-preparation step, not normal launcher setup.

## Run on a local GPU host

Use a dedicated Linux host with systemd/logind, a supported NVIDIA GPU, driver
580 or newer, Docker 24 or newer, the NVIDIA Container Toolkit configured for
Docker, and NVENC and NVDEC libraries. XR Render also needs the Vulkan loader
(`libvulkan1` on Ubuntu) and Node.js 24 with npm to build its web client.
Install Git, curl, Python 3, uv, and util-linux (`setsid`, `flock`).
Refer to {doc}`../getting_started/requirements`.

Run from the root of a trusted XR AI checkout as a regular user with Docker
access, not root. Use a dedicated runtime user: do not run unrelated model stacks
on the same ports, because upstream stop commands recognize servers by their ports.

Check that lingering is enabled for the runtime user:

```bash
loginctl show-user "$(id -un)" -p Linger
```

If it is not `Linger=yes`, setup stops with instructions. An administrator can
explicitly enable it with `sudo loginctl enable-linger "$(id -un)"`. This prevents
systemd from removing shared-memory rings when the last login session closes.
It persists across logouts and reboots and is never disabled by cleanup.

Replace the example below with the host's LAN IPv4 address reachable by your
headset or other sample clients. Then start the chooser:

```bash
XR_AI_UI_AUTH=password XR_AI_PUBLIC_HOST=192.168.1.50 ./deployments/brev/launch.sh setup
cat "$HOME/.cache/xr-ai-experiences/ui.password"
```

Open `http://127.0.0.1:8091` in a browser on that host. Sign in as `xrai` using
the password printed by the second command. Select an experience and deploy it;
inspect **Server logs** if startup fails.

Setup attempts Simple VLM automatically when hardware is compatible. Current
per-sample model stacks require an HF token; use **Use HF token** to supply it
and restart deployment. Older shared stacks can start anonymously. The chooser
remains available after a sample fails. Setup finishing means background startup
was initiated, not that models are loaded: wait for **Ready** and the client links.
**Stop active** stops the sample and its model stack, not the chooser.

To stop the experience, its models, and the chooser on this host, run:

```bash
./deployments/brev/launch.sh stop --all
```

This command is safe to repeat. It retains downloaded models, runtime checkouts,
recordings, logs, and credentials so a later `setup` can reuse them. It does not
change host settings or disable lingering. Close any SSH forwarding session
separately. To remove the saved HF token, use `forget-token` before `stop --all`.

Normal setup installs user-local Python 3.12 through uv, fetches a runtime checkout,
writes private user state, and starts processes. Model startup can download large
checkpoints and images. It does not install host packages, upgrade drivers or the
container toolkit, restart Docker, open firewall ports, install a boot service,
enable lingering, or reboot. Internet access and sufficient disk space are required.

## Remote access and saved settings

The chooser binds to `127.0.0.1:8091` by default. `XR_AI_PUBLIC_HOST` selects
the sample-client address; it does not expose the chooser at that address.
Setup prints an SSH tunnel command using your user, host address, and selected
UI port. Run that command on your workstation and keep its terminal open:

```bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8091:127.0.0.1:8091 user@host
```

Open `http://127.0.0.1:8091` with HTTP on the workstation. The tunnel carries only the chooser;
sample clients still need direct access to the configured host address and ports.

The default `XR_AI_UI_AUTH=password` requires the generated chooser password.
For Brev Secure Link or another authenticated HTTPS proxy, explicitly select
`XR_AI_UI_AUTH=proxy XR_AI_TRUST_PROXY=1`. This delegates authentication to that
proxy, skips chooser password generation, and removes the extra login prompt.
It does not authenticate requests itself or inspect forwarding headers for identity.
Existing chooser password files are retained so returning to password mode can
reuse them. HF tokens remain separate and are still required by model services.

Prefer proxying to the loopback listener. If the proxy
needs a network listener, explicitly set `XR_AI_UI_BIND=0.0.0.0 XR_AI_TRUST_PROXY=1`
when running setup. First firewall port 8091 so only the proxy can reach it and
require HTTPS and authentication on the proxy. Proxy mode must never expose the
backend directly: anyone reaching it can operate the chooser without another login.
The script does not configure the Secure Link or firewall; the Launchable or host
administrator must enforce this boundary. Never expose the plain HTTP listener
directly to the Internet: passwords and tokens need secure transport.

The default state directory is `~/.cache/xr-ai-experiences`; `XR_AI_STATE` overrides
it. In password mode, the generated login password is in `ui.password`. Access
mode and network overrides are saved in `host.env`. Later invocations reuse them;
explicit environment values
override saved values. Set `XR_AI_PUBLIC_HOST=` during setup to return to automatic
public-IP discovery. Tokens are stored separately and never placed in the boot unit.
Logs and events are protected by the chooser password or the authenticated proxy.
Known saved credentials are redacted, but
logs can still contain user content or other secrets; review them before sharing.
The CLI command `./deployments/brev/launch.sh forget-token` stops the active stack
and removes the saved HF token. Plain `stop` never removes credentials.

For optional startup after reboot, run `./deployments/brev/launch.sh provision-host`
after setup. This opt-in helper enables lingering and installs `xrai.service` using
the saved access mode and network settings, including port overrides. It does not start the service
immediately or reboot. Keep the launcher checkout at its installed path. After a
reboot, setup attempts its default sample; it does not promise to restore the last
active sample. Normal stop does not disable the boot service.
Neither does `stop --all`. If you previously enabled boot startup and no longer
want it, run `sudo systemctl disable --now xrai.service`, then `stop --all`.

## Revision selection

Fresh setup defaults to the newest supported version tag published in `NVIDIA/xr-ai`,
including beta and release-candidate tags. The chooser offers **Latest release**,
individual tags, and **Current checkout**. It displays the selected tag above the
resolved commit. A saved selection survives restarts: it is not automatically
upgraded when a new release appears. Select **Latest release** again to upgrade.
To select a release from the terminal:

```bash
XR_AI_VERSION=v0.3.0-beta ./deployments/brev/launch.sh setup
# Or use the UI revision selector or:
./deployments/brev/launch.sh commit v0.2.0-beta.2
# Resolve the newest supported release, or use this launcher's development checkout:
./deployments/brev/launch.sh commit latest
./deployments/brev/launch.sh commit current
# To start a fresh development setup without release discovery:
XR_AI_VERSION=current ./deployments/brev/launch.sh setup
```

Release tags are fetched from `NVIDIA/xr-ai` and resolved to their commit, including
annotated tags. A full commit SHA is also accepted. The resolved commit is saved
locally, so restarts reuse that revision without resolving the tag again. Explicitly
selecting a tag resolves its current target; use a full SHA when selecting an exact
immutable revision. Select only trusted releases or reviewed commits: changing
revisions executes their code on this host. Loading a different revision stops the
old sample; select and deploy a sample afterward.

Launcher and runtime checkouts are separate so old releases do not need to contain
this deployment directory. Runtime worktrees and caches live in the state directory.
**Current checkout** runs this launcher's checkout, not a moving `main` branch;
`XR_AI_ROOT` can specify another development checkout for this option. A saved
release runtime takes precedence over `XR_AI_ROOT`, so switching versions never
checks out files over the running UI or its deployment script. Release discovery
uses Git tags, not the GitHub API, and offers tags from `v0.2.0-beta.2` onward.
If discovery is unavailable, use **Current checkout** or enter a known tag or SHA.
Current samples supply their own model-server projects; older shared-stack layouts
use a small adapter. The launcher records the active model stack for shutdown;
stop failures block switching rather than removing unrelated Docker containers.

The chooser discovers sample console commands from `agent-samples/*/pyproject.toml`
and runs them with their default options. **Record session** is off by default and
available only when the selected sample declares a recording option. It records
audio, video, and session data on the host. The launcher uses `--session-recording`
on current runtimes or `--capture` on older releases such as `v0.3.0-beta`.
From the terminal, use `./deployments/brev/launch.sh start simple-vlm --session-recording`;
the launcher selects the supported sample flag. Restart deployment to change this option.
Procedure capture and replay is a separate sample: its
default flow records narrated procedures; guided replay is not yet implemented.

## Download session recordings

The **Recordings** section lists finalized bundles under
`~/.local/share/xr-ai/captures`. Disconnect the recording client or stop the
experience to finalize a session. The list refreshes every 15 seconds; choose
**Refresh recordings** for an immediate update, then **Download session ZIP**.
The archive includes the session manifest, media, and
recorded data. Finalized incomplete sessions are labeled; in-progress sessions
are not offered. Downloads use the chooser's existing access controls and do not
follow symbolic links or serve arbitrary host paths. Custom recording directories
and the procedure-capture-replay sample's separate evidence packets are not listed.

## Deployment implementation

Deployment policy lives in `catalog.py` and `compat.py`; `launch.sh` handles host
setup and process lifecycle. CI names the releases covered by compatibility tests;
there is no separate tag-to-commit map. The UI caches hardware inventory for 30
seconds, refreshing immediately when the selected runtime changes.

Current runtimes publish aggregate readiness through `XR_AI_STACK_READY_FILE`,
which the launcher sets only for the sample. The runtime writes the run ID after
all managed processes signal readiness and removes it during shutdown. The chooser
also checks process liveness. Older releases use a log adapter; only that adapter
depends on the sample's ready banner. Ready describes process initialization,
not an end-to-end check that every model or remote client remains healthy.

## Networking and clients

Without an override, the public IPv4 address is discovered through `api.ipify.org`
before each launch. `XR_AI_PUBLIC_HOST` selects a reachable LAN address instead.
Private addresses disable LiveKit external-IP discovery; public addresses enable
it. The address supplies client links, certificate SANs, and the XR Render endpoint.

Defaults are HTTPS client TCP 8080, LiveKit WebSocket TCP 7880, media TCP 7881 and
UDP 7882, token service TCP 8000, and CloudXR media UDP 47998. Restrict these to
trusted clients. `XR_AI_WEB_PORT`, `XR_AI_LIVEKIT_WS_PORT`, `XR_AI_LIVEKIT_TCP_PORT`,
`XR_AI_LIVEKIT_UDP_PORT`, and `XR_AI_TOKEN_PORT` override their respective ports;
`XR_AI_UI_PORT` changes the chooser port. These settings configure listeners, not
host firewalls or cloud security groups. Never expose model ports or the underlying
events listener to the Internet.

XR AI owns its development root and server certificates. The launcher sets the
reachable IP in `web_server_extra_sans` and clears legacy certificate overrides.
Current DeviceIOHub accepts these settings through `DEVICE_IO_HUB_NETWORK`, a
validated JSON environment override, leaving sample YAML unchanged. The release
adapter edits the isolated runtime YAML only for older versions without this hook.
Install the root served by the sample's `/cert` endpoint once per host and verify
its SHA-256 fingerprint in the chooser. Never copy the root private key. Changing
the IP regenerates the leaf, not the root. This development root is for trusted
demo environments, not a substitute for production PKI.

## Validation

```bash
XR_AI_TEST_ROOT="$PWD" python3 -m unittest discover -s tests/brev -v
```

The **validate-deployment-brev** workflow checks release-layout compatibility and
local HTTP behavior with mocked GPU inventory and launch commands. Changes to
samples, the launcher, model-stack utilities, the hub, and web clients trigger
these checks as well as deployment changes. It does not
create a Brev VM, download models, or test GPU execution. Before rollout, smoke-test
startup, sample switching, stop, logout/shared memory, network reachability, and
client TLS trust on an actual GPU host and Brev VM.
