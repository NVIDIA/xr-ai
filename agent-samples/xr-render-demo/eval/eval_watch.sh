#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Watch the xr-render-demo system prompt and re-run the eval on every
# change.  See eval/README.md for behaviour (debounce, single-instance,
# log location).
#
# Usage: ./eval_watch.sh [PROMPT_PATH]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROMPT="$HERE/../worker/prompts/system.txt"

PROMPT="${1:-$DEFAULT_PROMPT}"
LOG=/tmp/eval_loop.log
LOCK=/tmp/eval_watch.pid
EVAL="$HERE/eval.py"
WORKER="$HERE/../worker"
SELF="$(readlink -f "$0")"
DEBOUNCE_SECS=10
TIME_FMT='+%H:%M:%S'

last_hash=""
pending=0
pending_since=0
running_pid=""

# Identify watchers by exact argv from /proc — a substring match on the
# joined cmdline would also hit shells whose argv contains this script's
# path as a nested argument.
is_watcher() {
    local pid="$1" argv0 argv1
    [[ -r "/proc/$pid/cmdline" ]] || return 1
    { IFS= read -r -d '' argv0; IFS= read -r -d '' argv1 || true; } \
        < "/proc/$pid/cmdline" 2>/dev/null || return 1
    [[ "${argv0##*/}" == "bash" && "$argv1" == "$SELF" ]]
}

others=()
for d in /proc/[0-9]*/; do
    pid="${d#/proc/}"; pid="${pid%/}"
    [[ "$pid" == "$$" ]] && continue
    is_watcher "$pid" && others+=("$pid")
done
if (( ${#others[@]} > 0 )); then
    echo "ERROR: watcher already running at PID ${others[*]}" >&2
    echo "       to use it:     tail -f $LOG" >&2
    echo "       to replace it: kill ${others[*]} && $0 ${1:+$1}" >&2
    exit 1
fi
echo $$ > "$LOCK"

# Hash the file's content — using mtime alone causes spurious triggers because
# editors / language servers / git tools re-save the file without changing
# bytes (focus changes, refresh-on-blur, etc.).
file_hash() {
    sha1sum "$PROMPT" 2>/dev/null | awk '{print $1}' || true
    return 0
}

kill_running() {
    [[ -z "$running_pid" ]] && return
    # setsid put the eval in its own process group, so a negative PID
    # signals the whole tree (uv → python → child requests).
    if kill -0 "$running_pid" 2>/dev/null; then
        kill -TERM -- "-$running_pid" 2>/dev/null || true
        sleep 0.3
        kill -KILL -- "-$running_pid" 2>/dev/null || true
        echo "  ── aborted at $(date "$TIME_FMT") ──" >> "$LOG"
    fi
    wait "$running_pid" 2>/dev/null || true
    running_pid=""
    return 0
}

trigger() {
    {
        echo
        echo "═══════════════════════════════════════════════════════════════"
        echo "  $(date "$TIME_FMT")  prompt=$PROMPT"
        echo "═══════════════════════════════════════════════════════════════"
    } >> "$LOG"
    # Reuse the worker environment so schema discovery imports the exact NAT
    # function packages used by the live sample.
    setsid uv run --project "$WORKER" python "$EVAL" \
        --verbose --prompt "$PROMPT" >> "$LOG" 2>&1 &
    running_pid=$!
    return 0
}

cleanup() {
    kill_running
    rm -f "$LOCK"
    return 0
}
trap cleanup EXIT INT TERM HUP

echo "watching $PROMPT — log $LOG  debounce ${DEBOUNCE_SECS}s  (stop: kill $$)"
echo "started $(date "$TIME_FMT") (PID $$)" >> "$LOG"

# Baseline run on startup so the user sees a score immediately.
last_hash=$(file_hash)
trigger

while true; do
    hash=$(file_hash)
    # Editors / language servers can atomic-replace the file (write
    # tmp → rename), leaving sha1sum to briefly see ENOENT and emit
    # nothing.  Treat empty as "skip this tick" to avoid spurious
    # restarts.
    if [[ -z "$hash" ]]; then
        sleep 1
        continue
    fi
    now=$(date +%s)
    if [[ "$hash" != "$last_hash" ]]; then
        echo "  ── change detected at $(date "$TIME_FMT") (sha ${last_hash:0:8} → ${hash:0:8}) ──" >> "$LOG"
        last_hash="$hash"
        pending=1
        pending_since=$now
        kill_running
    elif (( pending == 1 && now - pending_since >= DEBOUNCE_SECS )); then
        pending=0
        trigger
    fi
    sleep 1
done
