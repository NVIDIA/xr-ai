#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SELF="$SCRIPT_DIR/launch.sh"
STATE_DIR="${XR_AI_STATE:-$HOME/.cache/xr-ai-experiences}"
HOST_SETTINGS=(XR_AI_PUBLIC_HOST XR_AI_UI_BIND XR_AI_UI_AUTH XR_AI_TRUST_PROXY XR_AI_UI_PORT XR_AI_WEB_PORT XR_AI_LIVEKIT_WS_PORT XR_AI_LIVEKIT_TCP_PORT XR_AI_LIVEKIT_UDP_PORT XR_AI_TOKEN_PORT)
# Persist only access and network settings, never tokens. Explicit environment values win.
if [[ -f "$STATE_DIR/host.env" ]]; then
  while IFS='=' read -r name value; do
    case " ${HOST_SETTINGS[*]} " in
      *" $name "*) [[ -n "${!name+x}" ]] || export "$name=$value" ;;
    esac
  done < "$STATE_DIR/host.env"
fi
SOURCE_ROOT="$STATE_DIR/xr-ai-source"
ROOT_STATE="$STATE_DIR/runtime-root"
PID_FILE="$STATE_DIR/sample.pid"
CURRENT_FILE="$STATE_DIR/sample.current"
RUN_FILE="$STATE_DIR/sample.run"
READY_FILE="$STATE_DIR/sample.ready"
HF_FILE="$STATE_DIR/hf"
EXIT_FILE="$STATE_DIR/sample.exit"
APP_LOG="$STATE_DIR/sample.log"
UI_PID_FILE="$STATE_DIR/ui.pid"
UI_LOG="$STATE_DIR/ui.log"
LOG_ROOT="$STATE_DIR/logs"

UI_PORT="${XR_AI_UI_PORT:-8091}"
WEB_PORT="${XR_AI_WEB_PORT:-8080}"
LIVEKIT_WS_PORT="${XR_AI_LIVEKIT_WS_PORT:-7880}"
LIVEKIT_TCP_PORT="${XR_AI_LIVEKIT_TCP_PORT:-7881}"
LIVEKIT_UDP_PORT="${XR_AI_LIVEKIT_UDP_PORT:-7882}"
TOKEN_PORT="${XR_AI_TOKEN_PORT:-8000}"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_PYTHON=3.12
export XR_AI_LOG_ROOT="$LOG_ROOT"
RUN_USER="$(id -un)"
PUBLIC_IP=""

die() { echo "ERROR: $*" >&2; exit 1; }
note() { echo "XR AI Experiences: $*" >&2; }
# uv selects the managed interpreter; it need not install python3.12 on PATH.
deployment_python() {
  if [[ -n "${XR_AI_DEPLOYMENT_PYTHON:-}" ]]; then "$XR_AI_DEPLOYMENT_PYTHON" "$@"
  else uv run --no-project --python "$UV_PYTHON" python "$@"; fi
}
first_line() { [[ -s "$1" ]] && head -n 1 "$1"; }
runtime_root() {
  first_line "$ROOT_STATE" 2>/dev/null || printf '%s\n' "${XR_AI_ROOT:-}"
}
is_xr_ai() { [[ -f "$1/agent-samples/simple-vlm-example/pyproject.toml" ]]; }
model_project() {
  deployment_python "$SCRIPT_DIR/compat.py" models "$(runtime_root)" "${1:-}"
}
pid_alive() {
  local pid
  pid="$(first_line "$1" 2>/dev/null || true)"
  [[ "$pid" =~ ^[0-9]+$ ]] && { kill -0 "$pid" 2>/dev/null || [[ -d "/proc/$pid" ]]; }
}
valid_port() { [[ "$2" =~ ^[0-9]+$ ]] && (( 1 <= $2 && $2 <= 65535 )) || die "$1 is not a valid port: $2"; }
public_ip() { if [[ -n "${XR_AI_PUBLIC_HOST:-}" ]]; then echo "$XR_AI_PUBLIC_HOST"; else curl -4fsSm8 https://api.ipify.org; fi; }
refresh_public_ip() {
  PUBLIC_IP="$(public_ip)" || die "public IP unavailable"
  [[ "$PUBLIC_IP" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "invalid public IP: $PUBLIC_IP"
  printf '%s\n' "$PUBLIC_IP" > "$STATE_DIR/public-ip"
}

prepare_state() {
  local file
  [[ "$(id -u)" != 0 ]] || die "run as a regular user, not root"
  [[ ! -L "$STATE_DIR" ]] || die "state directory must not be a symlink"
  mkdir -p "$STATE_DIR"
  [[ -O "$STATE_DIR" ]] || die "state directory must belong to the current user"
  chmod 700 "$STATE_DIR"
  for file in "$HF_FILE" "$PID_FILE" "$UI_PID_FILE" "$CURRENT_FILE" "$ROOT_STATE" "$STATE_DIR/ui.password" "$STATE_DIR/host.env"; do
    [[ ! -L "$file" ]] || die "state files must not be symlinks"
    if [[ -e "$file" ]]; then
      [[ -f "$file" && -O "$file" ]] || die "state files must be owned regular files"
      chmod 600 "$file"
    fi
  done
  mkdir -p "$LOG_ROOT"
}

validate_ports() {
  local name value
  while read -r name value; do valid_port "$name" "$value"; done <<EOF
XR_AI_UI_PORT $UI_PORT
XR_AI_WEB_PORT $WEB_PORT
XR_AI_LIVEKIT_WS_PORT $LIVEKIT_WS_PORT
XR_AI_LIVEKIT_TCP_PORT $LIVEKIT_TCP_PORT
XR_AI_LIVEKIT_UDP_PORT $LIVEKIT_UDP_PORT
XR_AI_TOKEN_PORT $TOKEN_PORT
EOF
}

resolve_source() {
  # Isolated runtime worktrees let old releases run without modifying this launcher checkout.
  local revision="${1:-${XR_AI_VERSION:-latest}}" repository="https://github.com/NVIDIA/xr-ai.git" commit runtime fetch_ref
  if [[ "$revision" == current ]]; then
    runtime="${XR_AI_ROOT:-$SCRIPT_DIR/../..}"
    is_xr_ai "$runtime" || die "$runtime is not an XR-AI checkout"
    SOURCE_RUNTIME="$(cd "$runtime" && pwd -P)"; SOURCE_REVISION=current
    return
  fi
  [[ "$revision" != latest ]] || revision="$(deployment_python "$SCRIPT_DIR/releases.py" --latest)" || return 1
  SOURCE_REVISION="$revision"
  mkdir -p "$STATE_DIR" "$(dirname "$SOURCE_ROOT")"
  if [[ "$revision" =~ ^[0-9a-f]{40}$ ]]; then fetch_ref="$revision"
  elif [[ "$revision" =~ ^v[0-9][a-zA-Z0-9._-]*$ ]]; then fetch_ref="refs/tags/$revision"
  else die "use a release tag (v...) or full commit SHA"; fi
  if [[ ! -e "$SOURCE_ROOT" ]]; then
    mkdir -p "$SOURCE_ROOT"
    git -C "$SOURCE_ROOT" init -q
    git -C "$SOURCE_ROOT" remote add origin "$repository"
  fi
  git -C "$SOURCE_ROOT" rev-parse --git-dir >/dev/null 2>&1 || die "$SOURCE_ROOT exists but is not a Git checkout"
  git -C "$SOURCE_ROOT" remote get-url origin >/dev/null 2>&1 || git -C "$SOURCE_ROOT" remote add origin "$repository"
  note "fetching XR-AI revision $revision"
  git -C "$SOURCE_ROOT" fetch --quiet --depth 1 origin "$fetch_ref" || die "could not fetch XR-AI revision $revision"
  commit="$(git -C "$SOURCE_ROOT" rev-parse --verify 'FETCH_HEAD^{commit}')"
  [[ "$fetch_ref" == refs/tags/* || "$commit" == "$revision" ]] || die "fetched revision does not match the requested commit"
  runtime="$STATE_DIR/xr-ai-runtime/${commit:0:12}"
  if [[ ! -e "$runtime" ]]; then
    mkdir -p "$(dirname "$runtime")"
    git -C "$SOURCE_ROOT" worktree add --quiet --detach "$runtime" "$commit"
  fi
  is_xr_ai "$runtime" || die "$runtime is not a compatible XR-AI worktree"
  SOURCE_RUNTIME="$(cd "$runtime" && pwd -P)"
}

activate_source() {
  printf '%s\n' "$SOURCE_RUNTIME" > "$ROOT_STATE"
  printf '%s\n' "${SOURCE_REVISION:-}" > "$STATE_DIR/runtime-ref"
}

ensure_source() {
  if [[ -s "$ROOT_STATE" && -z "${XR_AI_VERSION:-}" ]]; then
    is_xr_ai "$(runtime_root)" || die "saved runtime is unavailable"
  else switch_commit "${XR_AI_VERSION:-}"; fi
}

save_host_settings() {
  local name value
  for name in "${HOST_SETTINGS[@]}"; do
    value="${!name-}"
    [[ "$value" != *[!a-zA-Z0-9_.:-]* ]] || die "$name contains unsupported characters"
  done
  for name in "${HOST_SETTINGS[@]}"; do
    value="${!name-}"
    if [[ -n "$value" || "$name" == XR_AI_PUBLIC_HOST ]]; then
      printf '%s=%s\n' "$name" "$value"
    fi
  done > "$STATE_DIR/host.env"
}

install_boot_service() {
  [[ "$SELF$HOME$STATE_DIR${XR_AI_PUBLIC_HOST:-}${XR_AI_UI_BIND:-}${XR_AI_TRUST_PROXY:-}" != *[[:space:]]* ]] || die "boot service paths and settings must not contain whitespace"
  sudo tee /etc/systemd/system/xrai.service >/dev/null <<EOF
[Unit]
After=network-online.target docker.service
[Service]
Type=oneshot
User=$RUN_USER
Environment=HOME=$HOME
Environment=XR_AI_STATE=$STATE_DIR
EnvironmentFile=$STATE_DIR/host.env
ExecStart=$SELF setup
RemainAfterExit=yes
[Install]
WantedBy=multi-user.target
EOF
  sudo systemctl daemon-reload
  sudo systemctl enable xrai.service >/dev/null
}

catalog() {
  local root
  root="$(runtime_root)"
  [[ -n "$root" ]] || die "XR-AI is not provisioned"
  deployment_python "$SCRIPT_DIR/catalog.py" "$root" "$@"
}

configure_runtime() {
  local project="$1" host="$2" root
  root="$(runtime_root)"
  export DEVICE_IO_HUB_NETWORK
  DEVICE_IO_HUB_NETWORK="$(deployment_python "$SCRIPT_DIR/compat.py" configure "$root" "$project" "$host" "$LIVEKIT_WS_PORT" "$LIVEKIT_TCP_PORT" "$LIVEKIT_UDP_PORT" "$WEB_PORT" "$TOKEN_PORT")"
}

# The kernel releases the lock even if an action is killed before its EXIT trap runs.
acquire_lock() { exec 9>"$STATE_DIR/action.flock"; flock -n 9 || die "another launcher action is in progress"; }
release_lock() { exec 9>&-; }

switch_commit() {
  local revision="$1" old
  prepare_state
  acquire_lock; trap release_lock EXIT
  resolve_source "$revision"; old="$(runtime_root)"
  [[ "$SOURCE_RUNTIME" == "$old" ]] || stop_locked
  activate_source; note "commit $(git -C "$SOURCE_RUNTIME" rev-parse --short=12 HEAD)"
  release_lock; trap - EXIT
}

start_sample() {
  local sample="${1:-auto}" recording="${2:-}" selection project command recording_flag pid run_id
  [[ -z "$recording" || "$recording" == --session-recording ]] || die "expected --session-recording"
  prepare_state
  acquire_lock; trap release_lock EXIT
  selection="$(catalog --resolve "$sample")" || return 1
  IFS=$'\t' read -r sample project command recording_flag <<< "$selection"
  [[ -n "$project" && -n "$command" ]] || die "could not resolve $sample"
  if [[ -n "$recording" ]]; then
    [[ -n "$recording_flag" ]] || die "$sample does not support optional session recording"
  else recording_flag=""; fi
  if [[ -n "${HF_TOKEN:-}" ]]; then printf %s "$HF_TOKEN">"$HF_FILE"; elif [[ -s "$HF_FILE" ]]; then export HF_TOKEN="$(<"$HF_FILE")"; fi
  if [[ -s "$CURRENT_FILE" ]]; then stop_locked; fi
  refresh_public_ip; configure_runtime "$project" "$PUBLIC_IP"
  run_id="$(date -u +%Y%m%dT%H%M%SZ)"
  export XR_AI_LOG_TIMESTAMP="$run_id"
  : > "$APP_LOG"; rm -f "$EXIT_FILE" "$READY_FILE"
  printf '%s\n' "$sample" > "$CURRENT_FILE"; printf '%s\n' "$run_id" > "$RUN_FILE"
  printf '%s\n' "$recording_flag" > "$STATE_DIR/sample.recording"
  # Background shells inherit ignored SIGINT; restore it so run_stack can cancel
  # during readiness waits. The detached sample must not retain the action lock.
  nohup env --default-signal=INT setsid "$SELF" _run "$project" "$command" "$recording_flag" >> "$APP_LOG" 2>&1 </dev/null 9>&- &
  pid=$!; printf '%s\n' "$pid" > "$PID_FILE"; sleep 1
  pid_alive "$PID_FILE" || { tail -n 80 "$APP_LOG" >&2; stop_locked; die "$sample failed to start"; }
  note "starting $sample in the background (PID $pid)"
  release_lock; trap - EXIT
}

run_sample() {
  local project="$1" command="$2" recording="${3:-}" models model_command rc=0 anonymous=""
  case "$recording" in ""|--session-recording|--capture) ;; *) die "invalid session recording flag" ;; esac
  set --; [[ -z "$recording" ]] || set -- "$recording"
  trap 'rc=$?; printf "%s\n" "$rc" > "$EXIT_FILE"' EXIT
  [[ -z "${HF_TOKEN:-}" ]] && anonymous=--allow-anonymous
  models="$(model_project "$project")"
  model_command="$(deployment_python "$SCRIPT_DIR/compat.py" command "$models")"
  [[ "$model_command" =~ ^[a-zA-Z0-9_-]+$ ]] || die "invalid model command"
  if [[ "$models" == */agent-samples/*/model-servers ]]; then
    anonymous=""
    [[ -n "${HF_TOKEN:-}" ]] || die "this revision requires HF_TOKEN; enter it in the UI"
  fi
  printf '%s\n%s\n' "$models" "$model_command" > "$STATE_DIR/models.active"
  set +e
  if [[ "$project" == agent-samples/xr-render-demo ]]; then
    export NV_CXR_ENDPOINT_IP="$(first_line "$STATE_DIR/public-ip")"
    [[ $NV_CXR_ENDPOINT_IP ]] || die "CloudXR public IP unavailable"
    export NV_CXR_MEDIA_PORT=47998
  fi
  uv run --quiet --python "$UV_PYTHON" --project "$models" "$model_command" $anonymous; rc=$?
  if (( rc == 0 )); then
    deployment_python "$SCRIPT_DIR/compat.py" readiness "$(runtime_root)" > "$STATE_DIR/sample.status-mode"
    echo 'XR_AI_SAMPLE_START'
    XR_AI_STACK_READY_FILE="$READY_FILE" uv run --quiet --python "$UV_PYTHON" --project "$(runtime_root)/$project" "$command" "$@"; rc=$?
  fi
  exit "$rc"
}

stop_locked() {
  local pid attempt sample root models model_command
  sample="$(first_line "$CURRENT_FILE" 2>/dev/null || true)"
  if pid_alive "$PID_FILE"; then
    pid="$(first_line "$PID_FILE")"; note "stopping $sample (PID $pid)"
    # SIGTERM is not handled during startup in older run_stack releases. SIGINT
    # enters their KeyboardInterrupt cleanup, including separately grouped children.
    kill -INT -- "-$pid" 2>/dev/null || kill -INT "$pid" 2>/dev/null || true
    for ((attempt=0; attempt<30; attempt++)); do kill -0 -- "-$pid" 2>/dev/null || break; sleep 1; done
    if kill -0 -- "-$pid" 2>/dev/null; then
      die "sample shutdown did not finish; refusing to switch while cleanup is pending"
    fi
  fi
  root="$(runtime_root)"
  if [[ -n "$root" && -s "$STATE_DIR/models.active" ]]; then
    # Stop the recorded stack, not whichever stack the new selection would resolve.
    { read -r models; read -r model_command; } < "$STATE_DIR/models.active"
    [[ "$models" == "$root/"* && "$model_command" =~ ^[a-zA-Z0-9_-]+$ ]] || die "invalid active model stack"
    uv run --quiet --python "$UV_PYTHON" --project "$models" "$model_command" --stop || die "model shutdown failed; refusing to switch"
    rm -f "$STATE_DIR/models.active"
  fi
  rm -f "$PID_FILE"
}

stop_ui() {
  local pid argument attempt
  local -a arguments=()
  if pid_alive "$UI_PID_FILE"; then
    pid="$(first_line "$UI_PID_FILE")"
    # A stale PID file must never terminate an unrelated process after PID reuse.
    while IFS= read -r -d '' argument; do arguments+=("$argument"); done < "/proc/$pid/cmdline"
    [[ "${arguments[1]:-}" == "$SCRIPT_DIR/server.py" && "${arguments[4]:-}" == "$STATE_DIR" ]] ||
      die "chooser PID $pid does not match this launcher; refusing to stop it"
    kill -TERM "$pid" 2>/dev/null || true
    for ((attempt=0; attempt<50; attempt++)); do
      if ! kill -0 "$pid" 2>/dev/null || [[ "$(ps -p "$pid" -o stat=)" == Z* ]]; then break; fi
      sleep 0.1
    done
    (( attempt < 50 )) || die "chooser shutdown did not finish (PID $pid)"
  fi
  rm -f "$UI_PID_FILE"
}

stop_sample() {
  prepare_state; acquire_lock; trap release_lock EXIT
  stop_locked; note "sample stopped"
  if [[ "${1:-}" == --all ]]; then stop_ui; note "chooser stopped; saved credentials, logs, and caches retained"; fi
  release_lock; trap - EXIT
}

serve_ui() {
  prepare_state
  exec python3 "$SCRIPT_DIR/server.py" "$SELF" "$SCRIPT_DIR/ui.html" "$STATE_DIR" "$UI_PORT"
}

start_ui() {
  local attempt
  if pid_alive "$UI_PID_FILE"; then kill "$(first_line "$UI_PID_FILE")"; sleep 0.5; fi
  rm -f "$UI_PID_FILE"
  nohup setsid "$SELF" serve > "$UI_LOG" 2>&1 </dev/null &
  for ((attempt=0; attempt<30; attempt++)); do [[ -s "$UI_PID_FILE" ]] && pid_alive "$UI_PID_FILE" && return; sleep 1; done
  tail -n 40 "$UI_LOG" >&2; die "UI failed to start"
}

setup() {
  local command ui_bind="${XR_AI_UI_BIND:-127.0.0.1}" ui_auth="${XR_AI_UI_AUTH:-password}" tunnel_target tunnel
  prepare_state
  [[ "$(uname -s)" == Linux ]] || die "XR AI Experiences requires a Linux GPU host"
  for command in curl docker git python3 setsid flock nvidia-smi uv; do command -v "$command" >/dev/null || die "$command is required"; done
  validate_ports
  save_host_settings
  loginctl show-user "$RUN_USER" -p Linger --value | grep -qx yes || die "enable lingering first: sudo loginctl enable-linger $RUN_USER"
  uv python install "$UV_PYTHON"
  export XR_AI_DEPLOYMENT_PYTHON
  XR_AI_DEPLOYMENT_PYTHON="$(uv python find "$UV_PYTHON")"
  ensure_source; refresh_public_ip; start_ui
  note "XR-AI commit: $(git -C "$(runtime_root)" rev-parse HEAD)"
  if [[ "$ui_auth" == proxy ]]; then
    note "Open the Brev Secure Link or your authenticated HTTPS proxy for the chooser."
    note "Chooser backend: $ui_bind:$UI_PORT; keep it accessible only to the proxy."
    note "Authentication is provided by the proxy; no chooser username or password is required."
  elif [[ "$ui_bind" == 127.0.0.1 || "$ui_bind" == ::1 ]]; then
    tunnel_target="$ui_bind"; [[ "$ui_bind" != ::1 ]] || tunnel_target='[::1]'
    printf -v tunnel 'ssh -N -o ExitOnForwardFailure=yes -L %q %q' \
      "127.0.0.1:$UI_PORT:$tunnel_target:$UI_PORT" "$RUN_USER@$PUBLIC_IP"
    note "The chooser listens on this host only; $PUBLIC_IP is the sample-client address."
    note "From your workstation, run this command and keep it open:"
    note "  $tunnel"
    note "Then open http://127.0.0.1:$UI_PORT in your workstation browser (HTTP, not HTTPS)."
  else
    note "Chooser HTTP listener: $ui_bind:$UI_PORT; open your configured HTTPS proxy or Brev Secure Link."
  fi
  [[ "$ui_auth" != password ]] || note "Login user: xrai; password file: $STATE_DIR/ui.password"
  if ! pid_alive "$PID_FILE"; then
    (start_sample auto) || note "no demo auto-started; review hardware, credentials, and logs in the UI"
  fi
  printf -v command '%q stop' "$SELF"
  note "To stop the active experience and its models, run on this host (the chooser stays running):"
  note "  $command"
  note "To stop the chooser too, run: $command --all"
}

usage() {
  echo "Usage: launch.sh [setup|catalog|releases|commit REVISION|start DEMO [--session-recording]|stop [--all]|forget-token|serve|provision-host]"
  echo "commit REVISION: latest, current, a release tag, or a full commit SHA."
  echo "setup (default): start the chooser and attempt a default sample on a prepared Linux GPU host."
  echo "Local clients: XR_AI_PUBLIC_HOST=<LAN-IP> ./deployments/brev/launch.sh setup"
  echo "Brev Secure Link: XR_AI_UI_AUTH=proxy XR_AI_TRUST_PROXY=1 ./deployments/brev/launch.sh setup"
  echo "stop: stop the sample and its model stack; keep the chooser running."
  echo "stop --all: also stop the chooser; retain credentials, logs, caches, and host settings."
  echo "provision-host: opt in to lingering and a boot service (requires sudo); does not reboot."
  echo "Normal setup does not change system packages, drivers, firewall rules, or boot services."
}

case "${1:-setup}" in
  setup) setup ;;
  provision-host) prepare_state; validate_ports; save_host_settings; sudo loginctl enable-linger "$RUN_USER"; loginctl show-user "$RUN_USER" -p Linger --value | grep -qx yes || die "linger failed"; install_boot_service ;;
  catalog) shift; catalog "$@" ;;
  releases) deployment_python "$SCRIPT_DIR/releases.py" ;;
  commit) switch_commit "${2:-}" ;;
  start) [[ $# -le 3 ]] || die "start accepts DEMO and optional --session-recording"; start_sample "${2:-auto}" "${3:-}" ;;
  stop) [[ $# == 1 || ( $# == 2 && "$2" == --all ) ]] || die "stop accepts only --all; use forget-token to delete credentials"; stop_sample "${2:-}" ;;
  forget-token)
    prepare_state; acquire_lock; trap release_lock EXIT
    stop_locked; rm -f "$HF_FILE"; note "sample stopped; token removed"
    release_lock; trap - EXIT ;;
  serve) serve_ui ;;
  _run) run_sample "${2:?sample project is required}" "${3:?sample command is required}" "${4:-}" ;;
  help|-h|--help) usage ;;
  *) usage >&2; die "unknown command: $1" ;;
esac
