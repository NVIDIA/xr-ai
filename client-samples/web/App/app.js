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

const model = { ...createBaseModel() };

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

function render()           { renderBase(model); }

function enumerateCameras() { return _enumerateCameras(model, render); }
function stopCamera()       { return _stopCamera(model, render, showError); }
function startCamera()      { return _startCamera(model, { render, showError, enumerateCameras }); }
function startAudio()       { return _startAudio(model, render, showError); }
function stopAudio()        { return _stopAudio(model, render, showError); }
function disconnect()       { return _disconnect(model, render); }
function sendPing()         { return _sendPing(model, startCamera); }
function sendCustom(text)   { return _sendCustom(model, text, showError); }
function connect()          {
  return _connect(model, { render, showError, enumerateCameras, stopCamera });
}

// ─────────────────────────────────────────────────────────────────────────────
// Bootstrap
// ─────────────────────────────────────────────────────────────────────────────

wireBaseEvents(model, { connect, disconnect, startAudio, stopAudio, startCamera, stopCamera, sendPing, sendCustom });
render();
