// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @fileoverview XR render demo application — extends the base StreamKit sample
 * with CloudXR streaming via the XR Stream panel.
 *
 * All base logic (camera enumeration, LiveKit connect/disconnect, audio,
 * camera, data channel) lives in /App/core.js (symlinked from web/App/core.js).
 * This file adds only the XR-specific model fields, render additions, and
 * CloudXR event wiring.
 *
 * @module App/app
 */

import { CloudXRStream } from '/App/cloudxr.js';
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

const dbg = (typeof window !== 'undefined' && window.__dbg) || {
  info: () => {}, warn: () => {}, err: () => {},
};

// ─────────────────────────────────────────────────────────────────────────────
// Model state — base fields + XR extensions
// ─────────────────────────────────────────────────────────────────────────────

// When served over HTTPS the hub proxies LiveKit signaling at wss://<host>:<port>/rtc.
const _pageIsSecure = window.location.protocol === 'https:';
const _defaultPort  = _pageIsSecure ? (Number(window.location.port) || 443) : 7880;

const model = {
  ...createBaseModel(),
  port:   _defaultPort,
  secure: _pageIsSecure,
  // CloudXR state
  xrHost:  window.location.hostname || 'localhost',
  xrPort:  48322,
  /** @type {'idle'|'connecting'|'streaming'|'stopping'|'error'} */
  xrState: 'idle',
  /** @type {string|null} */
  xrError: null,
};

// Singleton CloudXR wrapper — created in wireEvents().
/** @type {CloudXRStream|null} */
let xrStream = null;

// ─────────────────────────────────────────────────────────────────────────────
// Error toast
// ─────────────────────────────────────────────────────────────────────────────

let _toastTimer = null;

function _hideToast() {
  const toast = $('error-toast');
  toast.classList.remove('visible');
  toast.textContent = '';
  model.lastError = null;
}

function showError(message) {
  model.lastError = message;
  const toast = $('error-toast');
  toast.textContent = message;
  toast.classList.add('visible');
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(_hideToast, 4000);
}

// ─────────────────────────────────────────────────────────────────────────────
// Bound actions (close over local model / render / showError)
// ─────────────────────────────────────────────────────────────────────────────

function enumerateCameras() { return _enumerateCameras(model, render); }
function stopCamera()       { return _stopCamera(model, render, showError); }
function startCamera()      { return _startCamera(model, { render, showError, enumerateCameras }); }
function startAudio()       { return _startAudio(model, render, showError); }
function stopAudio()        { return _stopAudio(model, render, showError); }
function disconnect()       { return _disconnect(model, render); }
function sendPing()         { return _sendPing(model, startCamera); }
function sendCustom(text)   { return _sendCustom(model, text, showError); }

function connect() {
  dbg.info(`connect() ${model.secure ? 'wss' : 'ws'}://${model.host}:${model.port} identity=${model.identity}`);
  return _connect(model, {
    render,
    showError,
    enumerateCameras,
    stopCamera,
    onDataReceived: _onDataReceived,
  });
}

/**
 * Intercepts XR-specific data topics before they hit the message list.
 * Returns true if the topic was consumed (suppresses list append).
 *
 * @param {string} topic
 * @param {Uint8Array} _data
 * @returns {boolean}
 */
function _onDataReceived(topic, _data) {
  // render.ready: agent confirms the XR scene is ready.
  if (topic === 'render.ready') {
    dbg.info('render.ready received');
    return true;  // informational — don't clutter the message list
  }
  return false;
}

// ─────────────────────────────────────────────────────────────────────────────
// Render — base + XR Stream panel
// ─────────────────────────────────────────────────────────────────────────────

function render() {
  renderBase(model);

  // ── XR Stream panel ────────────────────────────────────────────────────────
  const xrDot       = $('xr-state-dot');
  const xrLabel     = $('xr-state-label');
  const xrLaunchBtn = $('xr-launch-btn');
  const xrHostInput = $('xr-host-input');
  const xrPortInput = $('xr-port-input');

  xrDot.className = 'state-dot';
  switch (model.xrState) {
    case 'idle':
      xrDot.classList.add('disconnected');
      xrLabel.textContent = 'Idle';
      break;
    case 'connecting':
      xrDot.classList.add('connecting');
      xrLabel.textContent = 'Connecting…';
      break;
    case 'streaming':
      xrDot.classList.add('connected');
      xrLabel.textContent = 'Streaming';
      break;
    case 'stopping':
      xrDot.classList.add('reconnecting');
      xrLabel.textContent = 'Stopping…';
      break;
    case 'error':
      xrDot.classList.add('disconnected');
      xrLabel.textContent = 'Error';
      break;
  }

  const xrBusy   = model.xrState === 'connecting' || model.xrState === 'stopping';
  const xrActive = model.xrState === 'streaming';
  const xrInputsDisabled = xrActive || xrBusy;
  xrHostInput.disabled = xrInputsDisabled;
  xrPortInput.disabled = xrInputsDisabled;
  xrHostInput.closest('.field-row')?.classList.toggle('dimmed', xrInputsDisabled);
  xrPortInput.closest('.field-row')?.classList.toggle('dimmed', xrInputsDisabled);

  const xrCertLink = $('xr-cert-link');
  if (xrCertLink) {
    const certHost = (model.xrHost || '').trim() || window.location.hostname || 'localhost';
    const certPort = Number(model.xrPort) || 48322;
    const url = `https://${certHost}:${certPort}/`;
    if (xrCertLink.href !== url) {
      xrCertLink.href = url;
      _setCertAccepted(false);
      verifyCert(url);
    }
  }

  if (xrActive) {
    xrLaunchBtn.textContent = 'Connected';
    xrLaunchBtn.className   = 'btn btn-secondary btn-full';
    xrLaunchBtn.disabled    = true;
  } else if (xrBusy) {
    xrLaunchBtn.textContent = model.xrState === 'connecting' ? 'Connecting…' : 'Stopping…';
    xrLaunchBtn.className   = 'btn btn-secondary btn-full';
    xrLaunchBtn.disabled    = true;
  } else if (!model.isAudioActive) {
    xrLaunchBtn.textContent = 'Launch XR (start mic first)';
    xrLaunchBtn.className   = 'btn btn-secondary btn-full';
    xrLaunchBtn.disabled    = true;
  } else {
    xrLaunchBtn.textContent = 'Launch XR';
    xrLaunchBtn.className   = 'btn btn-primary btn-full';
    xrLaunchBtn.disabled    = false;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// CloudXR cert verification
// ─────────────────────────────────────────────────────────────────────────────

let _certAcceptedUrl = null;
let _certVerifyAbort = null;

function _setCertAccepted(accepted) {
  const hint = $('xr-cert-hint');
  const link = $('xr-cert-link');
  if (!hint || !link) return;
  if (accepted) {
    hint.classList.add('cert-accepted');
    link.textContent = '✓ CloudXR cert accepted';
  } else {
    hint.classList.remove('cert-accepted');
    link.textContent = 'accept the CloudXR cert';
  }
}

async function verifyCert(url) {
  if (_certVerifyAbort) _certVerifyAbort.abort();
  _certVerifyAbort = new AbortController();
  try {
    await fetch(url, { mode: 'no-cors', signal: _certVerifyAbort.signal });
    _certAcceptedUrl = url;
    _setCertAccepted(true);
  } catch (err) {
    if (err?.name === 'AbortError') return;
    if (_certAcceptedUrl === url) _certAcceptedUrl = null;
    _setCertAccepted(false);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// XR actions
// ─────────────────────────────────────────────────────────────────────────────

async function startXR() {
  if (!xrStream) return;
  model.xrError = null;
  const host = model.xrHost.trim() || window.location.hostname || 'localhost';
  const port = Number(model.xrPort) || 48322;
  dbg.info(`startXR() ${host}:${port}`);
  try {
    await xrStream.startXR(host, port);
  } catch (err) {
    dbg.err('startXR failed: ' + (err?.stack ?? err));
    model.xrError = String(err?.message ?? err);
    model.xrState = 'error';
    render();
  }
}

async function stopXR() {
  if (!xrStream) return;
  dbg.info('stopXR()');
  try {
    await xrStream.stopXR();
  } catch (err) {
    dbg.err('stopXR failed: ' + (err?.stack ?? err));
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Event wiring
// ─────────────────────────────────────────────────────────────────────────────

function wireEvents() {
  // Re-verify CloudXR cert when the user returns from accepting it in a new tab.
  window.addEventListener('focus', () => {
    const link = $('xr-cert-link');
    if (link?.href) verifyCert(link.href);
  });

  wireBaseEvents(model, { connect, disconnect, startAudio, stopAudio, startCamera, stopCamera, sendPing, sendCustom });

  // ── XR Stream ──────────────────────────────────────────────────────────────
  const xrHostInput = $('xr-host-input');
  const xrPortInput = $('xr-port-input');
  xrHostInput.value = model.xrHost;
  xrPortInput.value = model.xrPort;

  xrHostInput.addEventListener('input', (e) => { model.xrHost = e.target.value; render(); });
  xrPortInput.addEventListener('input', (e) => { model.xrPort = Number(e.target.value) || 48322; render(); });

  xrStream = new CloudXRStream({
    canvasId: 'xr-canvas',
    dbg,
    onStateChange: (state, detail) => {
      model.xrState = state;
      if (state === 'error') model.xrError = detail ?? null;
      if (state === 'idle')  model.xrError = null;
      dbg.info(`xr state → ${state}${detail ? ': ' + detail : ''}`);

      // Notify the agent that an XR client is now connected to CloudXR.
      // render-mcp gates its LOVR launch on this signal.
      if (state === 'streaming') {
        model.session?.send(new Uint8Array(0), { topic: 'xr.session.started' })
          .catch((err) => dbg.warn('xr.session.started publish failed: ' + err));
      }

      render();
    },
  });

  $('xr-launch-btn').addEventListener('click', () => {
    dbg.info('xr-launch-btn clicked (state=' + model.xrState + ')');
    if (model.xrState === 'streaming') stopXR();
    else if (model.xrState === 'idle' || model.xrState === 'error') startXR();
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Bootstrap
// ─────────────────────────────────────────────────────────────────────────────

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => { wireEvents(); render(); });
} else {
  wireEvents();
  render();
}
