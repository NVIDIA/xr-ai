// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @fileoverview Sample application — JavaScript equivalent of AppModel.swift + ContentView.swift.
 *
 * Wires observable model state to DOM elements using vanilla JS (no framework).
 * All model fields and action names mirror AppModel.swift exactly; DOM bindings
 * replace SwiftUI's @Observable / @Bindable machinery.
 *
 * Shared logic lives in /App/core.js; this file owns only the model instance,
 * the error toast, and the bootstrap call.
 *
 * @module App/app
 */

import {
  $,
  createBaseModel, renderBase,
  enumerateCameras  as _enumerateCameras,
  connect           as _connect,
  disconnect        as _disconnect,
  startAudio        as _startAudio,
  stopAudio         as _stopAudio,
  startCamera       as _startCamera,
  stopCamera        as _stopCamera,
  sendPing          as _sendPing,
  sendCustom        as _sendCustom,
  wireBaseEvents,
} from '/App/core.js';

// ─────────────────────────────────────────────────────────────────────────────
// Model state  (mirrors AppModel.swift field-for-field)
// ─────────────────────────────────────────────────────────────────────────────

const model = {
  ...createBaseModel(),
  /** @type {string|null} Most recent final agent reply text. */
  agentResponse: null,
};

// Topics carrying the agent's final text reply. Different samples publish on
// different topics (e.g. simple-vlm-example uses `vlm.response`, glasses-agent-nat
// uses `agent.response`); both route into the Agent panel and are suppressed
// from the "Received" list.
const AGENT_REPLY_TOPICS = new Set(['agent.response', 'vlm.response']);

// Local camera preview stream (separate from the LiveKit publish stream).
let _previewStream = null;

function releasePreviewStream() {
  if (!_previewStream) return;
  _previewStream.getTracks().forEach(t => t.stop());
  _previewStream = null;
  const videoEl = $('camera-preview');
  videoEl.srcObject = null;
  videoEl.style.transform = '';
}

// ─────────────────────────────────────────────────────────────────────────────
// Error toast
// ─────────────────────────────────────────────────────────────────────────────

let _toastTimer = null;

function showError(message) {
  model.lastError = message;
  const toast = $('error-toast');
  toast.textContent = message;
  toast.classList.add('visible');
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => {
    toast.classList.remove('visible');
    model.lastError = null;
  }, 4000);
}

// ─────────────────────────────────────────────────────────────────────────────
// Render + bound actions
// ─────────────────────────────────────────────────────────────────────────────

function render() {
  renderBase(model);

  // Camera preview elements.
  const video       = $('camera-preview');
  const placeholder = $('preview-placeholder');
  const liveBadge   = $('preview-live-badge');
  if (model.isCameraActive) {
    video.classList.add('active');
    placeholder.style.display = 'none';
    liveBadge.classList.add('active');
  } else {
    video.classList.remove('active');
    placeholder.style.display = '';
    liveBadge.classList.remove('active');
  }

  // Agent response.
  const responseEl = $('agent-response-text');
  if (model.agentResponse) {
    responseEl.textContent = model.agentResponse;
    responseEl.classList.remove('empty');
  } else {
    responseEl.textContent = 'Waiting for agent…';
    responseEl.classList.add('empty');
  }
}

function enumerateCameras() { return _enumerateCameras(model, render); }

async function stopCamera() {
  try {
    await _stopCamera(model, render, showError);
  } finally {
    releasePreviewStream();
  }
}

async function startCamera() {
  await _startCamera(model, { render, showError, enumerateCameras });
  if (model.isCameraActive) {
    try {
      releasePreviewStream();
      // No facingMode default — let the browser pick the same camera LiveKit
      // picks (both calls use `{video: true}` when no deviceId is selected,
      // so they converge on the system default). When the user has explicitly
      // chosen a camera in the dropdown, both pin to that deviceId.
      const constraints = model.selectedCameraId
        ? { video: { deviceId: { exact: model.selectedCameraId } } }
        : { video: true };
      _previewStream = await navigator.mediaDevices.getUserMedia(constraints);
      const videoEl = $('camera-preview');
      videoEl.srcObject = _previewStream;

      // Default: do NOT mirror — XR / glasses / mobile-back-camera capture
      // should preserve real-world orientation (left = left, right = right).
      // Only flip when the camera is explicitly user-facing (front mobile cam,
      // `facingMode === 'user'`). Desktop selfie webcams typically report no
      // facingMode and stay unmirrored; users who want the FaceTime-style
      // mirror UX on a desktop webcam can add a manual toggle later.
      const track = _previewStream.getVideoTracks()[0];
      const facingMode = track?.getSettings?.()?.facingMode ?? '';
      videoEl.style.transform = facingMode === 'user' ? 'scaleX(-1)' : '';
    } catch { /* preview failure is non-fatal */ }
  }
  render();
}

function startAudio()       { return _startAudio(model, render, showError); }
function stopAudio()        { return _stopAudio(model, render, showError); }
async function disconnect() {
  releasePreviewStream();
  try {
    await _disconnect(model, render);
  } finally {
    releasePreviewStream();
    render();
  }
}
function sendPing()         { return _sendPing(model, startCamera); }
function sendCustom(text)   { return _sendCustom(model, text, showError); }
function connect()          {
  return _connect(model, {
    render, showError, enumerateCameras, startCamera, stopCamera,
    onDataReceived(topic, data) {
      if (AGENT_REPLY_TOPICS.has(topic)) {
        model.agentResponse = new TextDecoder().decode(data);
        render();
        return true; // suppress from the received messages list
      }
      return false;
    },
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Bootstrap
// ─────────────────────────────────────────────────────────────────────────────

wireBaseEvents(model, { connect, disconnect, startAudio, stopAudio, startCamera, stopCamera, sendPing, sendCustom });
window.addEventListener('pagehide', () => {
  releasePreviewStream();
  const pendingDisconnect = model.session?.disconnect();
  if (pendingDisconnect) pendingDisconnect.catch(() => {});
});
render();
