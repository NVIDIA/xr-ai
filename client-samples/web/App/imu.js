// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @fileoverview IMU + camera-meta publisher for web clients.
 *
 * Why this exists: monocular SLAM (pose-mcp) is dramatically more robust
 * when it has a rough orientation prior between frames.  The phone's
 * built-in gyro + accel is plenty for that — even ~1 deg/s of integrated
 * error gives PnP a useful initial guess.  This module:
 *
 *  • Hooks `DeviceMotionEvent` (and the iOS permission gate) and
 *    publishes batched samples on data-channel topic `imu`.
 *  • Publishes a one-shot `camera_meta` message after camera start, with
 *    the active track's resolution / label / user agent — the server
 *    uses it to skip MoGe's per-frame FOV estimation when it can match
 *    the device against a known intrinsics table.
 *
 * Wire formats (both UTF-8 JSON on the LiveKit data channel):
 *
 *   topic = "imu"
 *   {
 *     "t":   <unix ms of the first sample in the batch>,
 *     "dt":  <ms between consecutive samples in the batch>,
 *     "a":   [[ax,ay,az], ...],          // device accelerometer, m/s² (with gravity)
 *     "alin":[[ax,ay,az], ...],          // accelerationIncludingGravity removed
 *                                        // ("acceleration" in the spec; m/s² gravity-free)
 *     "g":   [[gx,gy,gz], ...],          // gyro, rad/s   (zeros if not exposed)
 *   }
 *
 *   topic = "camera_meta"
 *   {
 *     "width": ..., "height": ..., "frame_rate": ...,
 *     "label": "<MediaStreamTrack.label or ''>",
 *     "user_agent": "<navigator.userAgent>",
 *     "facing": "user"|"environment"|null,
 *   }
 *
 * Both topics are best-effort (reliable=false) so they never back up the
 * data channel.
 */

const IMU_BATCH_MS = 100;     // flush samples this often
const IMU_TOPIC    = 'imu';
const META_TOPIC   = 'camera_meta';

let _running    = false;
let _onMotion   = null;
let _buffer     = [];           // [{t, a:[3], alin:[3]|null, g:[3]|null}]
let _flushTimer = null;
let _session    = null;
let _samplePeriodMs = 0;        // running estimate of inter-sample dt

/**
 * Best-effort accelerometer permission gate.  On iOS Safari `DeviceMotionEvent`
 * has a static `requestPermission` that has to be called on a user gesture
 * before any 'devicemotion' events arrive.  Everywhere else this is a no-op.
 *
 * Returns true when permission is granted (or not required), false when
 * the user denied.
 */
export async function ensureMotionPermission() {
  // @ts-ignore — requestPermission is iOS-only and not in the standard typings.
  const req = window.DeviceMotionEvent?.requestPermission;
  if (typeof req !== 'function') return true;
  try {
    const result = await req.call(DeviceMotionEvent);
    return result === 'granted';
  } catch {
    return false;
  }
}

/**
 * Start publishing IMU samples to the given session.  Idempotent — calling
 * twice is a no-op.
 *
 * @param {import('/StreamKit/index.js').StreamSession} session
 */
export function startImuPublisher(session) {
  if (_running) return;
  if (typeof window.DeviceMotionEvent === 'undefined') return;
  _session    = session;
  _running    = true;
  _buffer     = [];
  _flushTimer = setInterval(_flush, IMU_BATCH_MS);
  let lastTs  = 0;

  _onMotion = (ev) => {
    const now = ev.timeStamp ?? performance.now();
    // Running estimate of sample period — phones report at ~50–60 Hz.
    if (lastTs) _samplePeriodMs = 0.7 * _samplePeriodMs + 0.3 * (now - lastTs);
    lastTs = now;
    _buffer.push({
      t:    Date.now(),
      a:    ev.accelerationIncludingGravity
              ? [ev.accelerationIncludingGravity.x ?? 0,
                 ev.accelerationIncludingGravity.y ?? 0,
                 ev.accelerationIncludingGravity.z ?? 0]
              : null,
      alin: ev.acceleration
              ? [ev.acceleration.x ?? 0,
                 ev.acceleration.y ?? 0,
                 ev.acceleration.z ?? 0]
              : null,
      g:    ev.rotationRate
              ? [(ev.rotationRate.alpha ?? 0) * Math.PI / 180,
                 (ev.rotationRate.beta  ?? 0) * Math.PI / 180,
                 (ev.rotationRate.gamma ?? 0) * Math.PI / 180]
              : null,
    });
  };
  window.addEventListener('devicemotion', _onMotion);
}

/** Stop the publisher and discard any unsent samples. */
export function stopImuPublisher() {
  if (!_running) return;
  if (_onMotion) window.removeEventListener('devicemotion', _onMotion);
  if (_flushTimer) clearInterval(_flushTimer);
  _running    = false;
  _onMotion   = null;
  _flushTimer = null;
  _buffer     = [];
  _session    = null;
}

function _flush() {
  if (!_session || !_buffer.length) return;
  const batch = _buffer;
  _buffer = [];
  const payload = {
    t:    batch[0].t,
    dt:   Math.max(1, Math.round(_samplePeriodMs)),
    a:    batch.map(s => s.a ?? [0, 0, 0]),
    alin: batch.map(s => s.alin ?? [0, 0, 0]),
    g:    batch.map(s => s.g ?? [0, 0, 0]),
  };
  // Best-effort, never await — IMU is a one-way fire-and-forget stream
  // and a slow ack must never stall the publisher.
  _session.send(new TextEncoder().encode(JSON.stringify(payload)),
                { reliable: false, topic: IMU_TOPIC }).catch(() => {});
}

/**
 * Publish a one-shot description of the active camera so the server can
 * skip MoGe's FOV estimation if it can match against a known device.
 *
 * Pass the live `MediaStreamTrack` from the active camera (typically
 * via `session.getLocalVideoTrack()` or whichever StreamKit method
 * surfaces it).  Falls back to publishing only the user agent if no
 * track is available.
 *
 * @param {import('/StreamKit/index.js').StreamSession} session
 * @param {MediaStreamTrack | null} videoTrack
 */
export async function publishCameraMeta(session, videoTrack) {
  if (!session) return;
  const settings = videoTrack?.getSettings?.() ?? {};
  const payload = {
    width:      settings.width      ?? null,
    height:     settings.height     ?? null,
    frame_rate: settings.frameRate  ?? null,
    label:      videoTrack?.label   ?? '',
    facing:     settings.facingMode ?? null,
    user_agent: navigator.userAgent,
  };
  try {
    await session.send(new TextEncoder().encode(JSON.stringify(payload)),
                       { reliable: true, topic: META_TOPIC });
  } catch { /* ignore — meta is optional */ }
}
