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
  /** @type {string|null} Most recent agent.response message text. */
  agentResponse: null,
};

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
      // Match LiveKit's default selection (facingMode: 'user') when no specific
      // deviceId is chosen, so the preview and the published track land on the
      // same physical camera. The cleaner fix — cloning the LiveKit publish
      // track instead of opening a second getUserMedia — requires StreamKit
      // accessors and lands in PR #153.
      const constraints = model.selectedCameraId
        ? { video: { deviceId: { exact: model.selectedCameraId } } }
        : { video: { facingMode: 'user' } };
      _previewStream = await navigator.mediaDevices.getUserMedia(constraints);
      const videoEl = $('camera-preview');
      videoEl.srcObject = _previewStream;

      // Mirror by default (desktop/laptop webcams report no facingMode but are
      // selfie cameras). Only the explicit back-camera case skips the flip.
      const track = _previewStream.getVideoTracks()[0];
      const facingMode = track?.getSettings?.()?.facingMode ?? '';
      videoEl.style.transform = facingMode === 'environment' ? '' : 'scaleX(-1)';
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
    render, showError, enumerateCameras, stopCamera,
    onDataReceived(topic, data) {
      if (topic === 'agent.response') {
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
