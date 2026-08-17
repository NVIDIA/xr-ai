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
// different topics (e.g. simple-vlm-example uses `vlm.response`, xr-render-demo
// uses `agent.response`); both route into the Agent panel and are suppressed
// from the "Received" list.
const AGENT_REPLY_TOPICS = new Set(['agent.response', 'vlm.response']);

function clearCameraPreview() {
  const videoEl = $('camera-preview');
  videoEl.onresize = null;
  videoEl.srcObject = null;
  videoEl.style.transform = '';
  videoEl.closest('.preview-card').style.aspectRatio = '';
}

function showPublishedCameraPreview() {
  clearCameraPreview();
  const track = model.session?.cameraTrack;
  if (!track) return;

  const videoEl = $('camera-preview');
  const previewCard = videoEl.closest('.preview-card');
  videoEl.srcObject = new MediaStream([track]);

  const updateAspectRatio = () => {
    const settings = track.getSettings();
    const width = videoEl.videoWidth || settings.width;
    const height = videoEl.videoHeight || settings.height;
    if (width && height) {
      previewCard.style.aspectRatio = `${width} / ${height}`;
    }
  };
  videoEl.onresize = updateAspectRatio;
  updateAspectRatio();

  const facingMode = track.getSettings().facingMode ?? '';
  videoEl.style.transform = facingMode === 'user' ? 'scaleX(-1)' : '';
  console.info('Camera preview uses published track settings', track.getSettings());
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
    clearCameraPreview();
  }
}

async function startCamera() {
  await _startCamera(model, { render, showError, enumerateCameras });
  if (model.isCameraActive) {
    showPublishedCameraPreview();
  }
  render();
}

function startAudio()       { return _startAudio(model, render, showError); }
function stopAudio()        { return _stopAudio(model, render, showError); }
async function disconnect() {
  clearCameraPreview();
  try {
    await _disconnect(model, render);
  } finally {
    clearCameraPreview();
    render();
  }
}
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

wireBaseEvents(model, { connect, disconnect, startAudio, stopAudio, startCamera, stopCamera, sendCustom });
window.addEventListener('pagehide', () => {
  clearCameraPreview();
  const pendingDisconnect = model.session?.disconnect();
  if (pendingDisconnect) pendingDisconnect.catch(() => {});
});
render();
