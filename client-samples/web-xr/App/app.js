// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @fileoverview XR render demo application — extends the base StreamKit sample
 * with CloudXR streaming via the XR Stream panel.
 *
 * Wires observable model state to DOM elements using vanilla JS (no framework).
 * All model fields and action names mirror AppModel.swift exactly; DOM bindings
 * replace SwiftUI's @Observable / @Bindable machinery.
 *
 * @module App/app
 */

import {
  StreamSession,
  ConnectionState,
  StreamError,
  AudioConfig,
  MicrophoneMode,
  CameraConfig,
  SessionConfig,
  BackendConfiguration,
  LiveKitConfig,
} from '/StreamKit/index.js';
import { CloudXRStream } from '/App/cloudxr.js';

const dbg = (typeof window !== 'undefined' && window.__dbg) || {
  info: () => {}, warn: () => {}, err: () => {},
};
dbg.info('app.js module loaded');

// ─────────────────────────────────────────────────────────────────────────────
// Camera enumeration
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Queries available video input devices and updates `model.cameras`.
 * Labels are only populated after the user has granted camera permission;
 * calling this again after the first `startCamera()` yields labelled entries.
 *
 * @returns {Promise<void>}
 */
async function enumerateCameras() {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  try {
    let devices = await navigator.mediaDevices.enumerateDevices();
    let cameras = devices.filter(d => d.kind === 'videoinput');

    // deviceId is an empty string until camera permission has been granted.
    // Request a brief permission probe so we get real device IDs and labels.
    if (cameras.length > 0 && !cameras.some(d => d.deviceId)) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        stream.getTracks().forEach(t => t.stop());
        devices  = await navigator.mediaDevices.enumerateDevices();
        cameras  = devices.filter(d => d.kind === 'videoinput');
      } catch { /* permission denied — proceed with anonymous devices */ }
    }

    const list = cameras.map((d, i) => ({
      deviceId: d.deviceId,
      label:    d.label || `Camera ${i + 1}`,
    }));

    model.cameras = list;
    // Preserve selection if it still exists; otherwise default to first device.
    if (list.length > 0 && !list.some(c => c.deviceId === model.selectedCameraId)) {
      model.selectedCameraId = list[0].deviceId;
    }
    render();
  } catch {
    // enumerateDevices not available — ignore.
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Model state  (mirrors AppModel.swift field-for-field)
// ─────────────────────────────────────────────────────────────────────────────

// When the page is served over HTTPS the hub proxies LiveKit signaling at
// wss://<host>:<page-port>/rtc — same origin as the page itself. Over plain
// HTTP the browser connects directly to ws://<host>:7880.
const _pageIsSecure = window.location.protocol === 'https:';
const _defaultPort  = _pageIsSecure
  ? Number(window.location.port) || 443
  : 7880;

const model = {
  // Connection settings
  host:           window.location.hostname || 'localhost',
  port:           _defaultPort,
  secure:         _pageIsSecure,
  tokenServerURL: '',       // defaults to /token relative to origin when blank
  token:          '',       // pre-signed JWT — alternative to tokenServerURL
  identity:       'web-client',

  // Media settings
  audioMode:      MicrophoneMode.RAW,

  // Live state
  /** @type {StreamSession|null} */
  session:         null,
  connectionState: ConnectionState.DISCONNECTED,
  isAudioActive:   false,
  isCameraActive:  false,
  /** @type {Array<{deviceId: string, label: string}>} */
  cameras:          [],
  /** @type {string|null} */
  selectedCameraId: null,
  /** @type {string|null} */
  agentStatus: null,
  /** When true, ``clientControl`` startCamera/stopCamera messages from the
   *  agent are honoured.  When false (default — always-on), they are ignored
   *  and the camera button is the sole control. */
  cameraOnDemand: false,
  /** @type {Array<{id: string, text: string, timestamp: Date}>} */
  receivedMessages: [],
  /** @type {string|null} */
  lastError:       null,

  // ── CloudXR state ──────────────────────────────────────────────────────
  xrHost:  window.location.hostname || 'localhost',
  xrPort:  48322,
  /** @type {'idle'|'connecting'|'streaming'|'stopping'|'error'} */
  xrState: 'idle',
  /** @type {string|null} */
  xrError: null,
};

// Singleton CloudXR wrapper — created after DOM is ready in wireEvents().
/** @type {CloudXRStream|null} */
let xrStream = null;

// ─────────────────────────────────────────────────────────────────────────────
// DOM helpers
// ─────────────────────────────────────────────────────────────────────────────

/** @param {string} id @returns {HTMLElement} */
const $ = (id) => document.getElementById(id);

// ─────────────────────────────────────────────────────────────────────────────
// render()  — updates every DOM element from current model state
// ─────────────────────────────────────────────────────────────────────────────

function render() {
  const state = model.connectionState;
  const isDisconnected  = state === ConnectionState.DISCONNECTED;
  const isConnected     = state === ConnectionState.CONNECTED;
  const isTransitioning = state === ConnectionState.CONNECTING
                       || state === ConnectionState.RECONNECTING;

  // ── State badge ────────────────────────────────────────────────────────────
  const dot   = $('state-dot');
  const label = $('state-label');

  dot.className = 'state-dot';   // reset
  switch (state) {
    case ConnectionState.DISCONNECTED:
      dot.classList.add('disconnected');
      label.textContent = 'Disconnected';
      break;
    case ConnectionState.CONNECTING:
      dot.classList.add('connecting');
      label.textContent = 'Connecting…';
      break;
    case ConnectionState.CONNECTED:
      dot.classList.add('connected');
      label.textContent = 'Connected';
      break;
    case ConnectionState.RECONNECTING:
      dot.classList.add('reconnecting');
      label.textContent = 'Reconnecting…';
      break;
  }

  // ── Config inputs — disabled / dimmed while not disconnected ───────────────
  const configDisabled = !isDisconnected;
  for (const id of ['host-input', 'port-input', 'token-input', 'token-url-input', 'identity-input']) {
    const el = $(id);
    el.disabled = configDisabled;
    el.closest('.field-row')?.classList.toggle('dimmed', configDisabled);
  }

  // ── Connect button ─────────────────────────────────────────────────────────
  const connectBtn = $('connect-btn');
  if (isDisconnected) {
    connectBtn.textContent = 'Connect';
    connectBtn.className   = 'btn btn-primary';
    connectBtn.disabled    = false;
  } else if (isTransitioning) {
    connectBtn.textContent = state === ConnectionState.CONNECTING
      ? 'Connecting…'
      : 'Reconnecting…';
    connectBtn.className   = 'btn btn-secondary';
    connectBtn.disabled    = true;
  } else {
    connectBtn.textContent = 'Disconnect';
    connectBtn.className   = 'btn btn-destructive';
    connectBtn.disabled    = false;
  }

  // ── Audio ──────────────────────────────────────────────────────────────────
  const audioBtn    = $('audio-btn');
  const audioStatus = $('audio-status');
  const audioModeSelect = $('audio-mode-select');

  audioBtn.disabled = !isConnected;
  audioModeSelect.disabled = model.isAudioActive;
  if (model.isAudioActive) {
    audioBtn.textContent     = 'Stop Microphone';
    audioBtn.className       = 'btn btn-destructive';
    audioStatus.textContent  = 'Live';
    audioStatus.className    = 'status-text status-active';
  } else {
    audioBtn.textContent     = 'Start Microphone';
    audioBtn.className       = 'btn btn-secondary';
    audioStatus.textContent  = isConnected ? 'Idle' : 'Not connected';
    audioStatus.className    = 'status-text status-idle';
  }

  // ── Camera ─────────────────────────────────────────────────────────────────
  const cameraBtn    = $('camera-btn');
  const cameraStatus = $('camera-status');

  cameraBtn.disabled = !isConnected;
  if (model.isCameraActive) {
    cameraBtn.textContent = 'Stop Camera';
    cameraBtn.className   = 'btn btn-destructive';
    cameraStatus.textContent = 'Streaming';
    cameraStatus.className   = 'status-text status-active';
  } else {
    cameraBtn.textContent = 'Start Camera';
    cameraBtn.className   = 'btn btn-secondary';
    cameraStatus.textContent = isConnected ? 'Idle' : 'Not connected';
    cameraStatus.className   = 'status-text status-idle';
  }

  // ── Camera on demand toggle ────────────────────────────────────────────────
  const codCheckbox = $('camera-on-demand');
  if (codCheckbox) codCheckbox.checked = model.cameraOnDemand;

  // ── Camera selector (shown only when multiple cameras detected) ────────────
  const selectRow = $('camera-select-row');
  const camSelect = $('camera-select');

  if (model.cameras.length > 1) {
    selectRow.style.display = '';
    // Rebuild options only when the list has changed.
    const currentIds = [...camSelect.options].map(o => o.value).join(',');
    const newIds     = model.cameras.map(c => c.deviceId).join(',');
    if (currentIds !== newIds) {
      camSelect.innerHTML = model.cameras
        .map(c => `<option value="${escapeHtml(c.deviceId)}">${escapeHtml(c.label)}</option>`)
        .join('');
    }
    camSelect.value    = model.selectedCameraId ?? '';
    camSelect.disabled = model.isCameraActive;
  } else {
    selectRow.style.display = 'none';
  }

  // ── Agent status ───────────────────────────────────────────────────────────
  const agentDot   = $('agent-status-dot');
  const agentLabel = $('agent-status-label');

  if (!isConnected) {
    agentDot.className    = 'state-dot disconnected';
    agentLabel.textContent = '—';
  } else if (model.agentStatus === 'processing') {
    agentDot.className    = 'state-dot connecting';  // orange pulse
    agentLabel.textContent = 'Processing…';
  } else if (model.agentStatus === 'idle') {
    agentDot.className    = 'state-dot connected';   // green
    agentLabel.textContent = 'Idle';
  } else {
    agentDot.className    = 'state-dot disconnected';
    agentLabel.textContent = 'Unknown';
  }

  // ── XR Stream ──────────────────────────────────────────────────────────────
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

  // Cert-accept link tracks the host/port inputs so user can tap it to
  // accept the cloudxr-runtime self-signed cert before clicking Launch XR.
  const xrCertLink = $('xr-cert-link');
  if (xrCertLink) {
    const certHost = (model.xrHost || '').trim() || window.location.hostname || 'localhost';
    const certPort = Number(model.xrPort) || 48322;
    const url = `https://${certHost}:${certPort}/`;
    if (xrCertLink.href !== url) {
      xrCertLink.href = url;
      // Reset the accepted state whenever the URL changes; verifyCert()
      // will re-flip it to green if the new URL's cert is also trusted.
      _setCertAccepted(false);
      verifyCert(url);
    }
  }

  if (xrActive) {
    // The 2D page can't meaningfully exit XR — the user is inside the
    // headset at this point. Show a disabled "Connected" state; the session
    // ends when the user exits via the headset UI or takes the headset off.
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

  // ── Data channel ───────────────────────────────────────────────────────────
  $('ping-btn').disabled = !isConnected;
  $('send-btn').disabled = !isConnected || $('message-input').value.trim() === '';

  // ── Received messages ──────────────────────────────────────────────────────
  const list = $('messages');
  if (model.receivedMessages.length === 0) {
    list.innerHTML = '<li class="empty-hint">No messages received yet.</li>';
  } else {
    list.innerHTML = model.receivedMessages
      .map(({ id, text, timestamp }) => {
        const timeStr = timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        return `<li class="message-item" data-id="${id}">
          <span class="message-text">${escapeHtml(text)}</span>
          <span class="message-time">${escapeHtml(timeStr)}</span>
        </li>`;
      })
      .join('');
  }
}

/** @param {string} s @returns {string} */
function escapeHtml(s) {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ─────────────────────────────────────────────────────────────────────────────
// CloudXR cert-accept verification
// ─────────────────────────────────────────────────────────────────────────────
//
// Mirrors the cloudxr-js/examples/simple cert-acceptance pattern: probe the
// WSS proxy URL with fetch(mode='no-cors'). If the fetch resolves without
// throwing, the browser has accepted the cert and we flip the hint to a
// green "✓ accepted" state. If it throws (TypeError on cert reject) we
// leave the hint blue so the user can tap the link to accept it.

let _certAcceptedUrl = null;     // URL whose cert is currently believed accepted
let _certVerifyAbort = null;     // AbortController for the in-flight probe

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
    if (err && err.name === 'AbortError') return;
    if (_certAcceptedUrl === url) _certAcceptedUrl = null;
    _setCertAccepted(false);
  }
}

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
// Actions  (mirror AppModel.swift methods)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Resolves the effective token-server URL.
 * When `model.tokenServerURL` is blank, defaults to `/token` relative to the
 * page's origin — matching the iOS pattern of a local dev server.
 *
 * @returns {string}
 */
function resolvedTokenURL() {
  if (model.tokenServerURL.trim() !== '') return model.tokenServerURL.trim();
  return `${window.location.origin}/token`;
}

/**
 * Connects to the LiveKit room.
 * Mirrors AppModel.connect() exactly.
 *
 * @returns {Promise<void>}
 */
async function connect() {
  model.lastError = null;
  model.receivedMessages = [];

  const scheme = model.secure ? 'wss' : 'ws';
  dbg.info('connect() ' + scheme + '://' + model.host + ':' + model.port +
           ' identity=' + model.identity +
           ' tokenURL=' + (model.token.trim() ? '(inline token)' : resolvedTokenURL()));

  const lkConfig = new LiveKitConfig({
    host:     model.host,
    port:     Number(model.port),
    secure:   model.secure,
    token:    model.token.trim()   || null,
    tokenURL: model.token.trim()   ? null : resolvedTokenURL(),
  });

  let newSession;
  try {
    newSession = await StreamSession.create(BackendConfiguration.liveKit(lkConfig));
    dbg.info('StreamSession.create() ok');
  } catch (err) {
    dbg.err('StreamSession.create failed: ' + (err && err.stack ? err.stack : err));
    showError(err instanceof StreamError ? err.message : String(err));
    render();
    return;
  }

  // Wire callbacks before connecting (mirrors iOS ordering).
  newSession.onConnectionStateChanged = (state) => {
    model.connectionState = state;
    if (state === ConnectionState.DISCONNECTED) {
      model.isAudioActive  = false;
      model.isCameraActive = false;
      model.agentStatus    = null;
    }
    render();
  };

  newSession.onAgentStatus = (status) => {
    model.agentStatus = status;
    render();
  };

  newSession.onDataReceived = (topic, data) => {
    // Camera on demand: agent sends {"action":"startCamera/stopCamera"} on
    // "clientControl".  In always-on mode (cameraOnDemand === false) these
    // are silently ignored so manual camera control still works uninterrupted.
    if (topic === 'clientControl') {
      if (model.cameraOnDemand) {
        try {
          const { action } = JSON.parse(new TextDecoder().decode(data));
          if (action === 'startCamera' && !model.isCameraActive) startCamera();
          if (action === 'stopCamera'  &&  model.isCameraActive) stopCamera();
        } catch { /* malformed — ignore */ }
      }
      return;  // never surface in the message list
    }

    const text = (() => {
      try {
        return new TextDecoder().decode(data);
      } catch {
        return `[${data.byteLength ?? data.length} bytes binary]`;
      }
    })();
    model.receivedMessages.unshift({
      id:        crypto.randomUUID(),
      text:      topic ? `[${topic}] ${text}` : text,
      timestamp: new Date(),
    });
    render();
  };

  model.session = newSession;

  const sessionConfig = new SessionConfig({
    audio:    new AudioConfig({ mode: MicrophoneMode.DISABLED }),  // user starts explicitly
    camera:   CameraConfig.disabled,
    identity: model.identity,
  });

  try {
    await newSession.connect(sessionConfig);
    dbg.info('session.connect() ok');
  } catch (err) {
    dbg.err('session.connect failed: ' + (err && err.stack ? err.stack : err));
    showError(err instanceof StreamError ? err.message : String(err));
    model.session = null;
    model.connectionState = ConnectionState.DISCONNECTED;
  }

  render();
}

/**
 * Disconnects from the room and resets live state.
 * Mirrors AppModel.disconnect().
 *
 * @returns {Promise<void>}
 */
async function disconnect() {
  await model.session?.disconnect();
  model.session          = null;
  model.connectionState  = ConnectionState.DISCONNECTED;
  model.isAudioActive    = false;
  model.isCameraActive   = false;
  render();
}

async function startAudio() {
  if (!navigator.mediaDevices) {
    const msg = window.isSecureContext
      ? 'mediaDevices API unavailable in this browser'
      : 'Mic/camera require a secure context. Use https:// or add this origin to chrome://flags "Insecure origins treated as secure".';
    dbg.err('startAudio blocked: ' + msg);
    showError(msg);
    return;
  }
  try {
    await model.session?.startAudio(new AudioConfig({ mode: model.audioMode }));
    model.isAudioActive = true;
  } catch (err) {
    dbg.err('startAudio failed: ' + (err && err.stack ? err.stack : err));
    showError(err instanceof StreamError ? err.message : String(err));
  }
  render();
}

async function stopAudio() {
  try {
    await model.session?.stopAudio();
  } catch (err) {
    showError(err instanceof StreamError ? err.message : String(err));
  }
  model.isAudioActive = false;
  render();
}

/**
 * Starts local camera capture and publishes it.
 * Mirrors AppModel.startCamera().
 *
 * @returns {Promise<void>}
 */
async function startCamera() {
  if (!navigator.mediaDevices) {
    const msg = window.isSecureContext
      ? 'mediaDevices API unavailable in this browser'
      : 'Mic/camera require a secure context. Use https:// or add this origin to chrome://flags "Insecure origins treated as secure".';
    dbg.err('startCamera blocked: ' + msg);
    showError(msg);
    return;
  }
  // Enumerate first (triggers permission prompt if needed) so the selector
  // is populated with real device names before the camera goes active.
  await enumerateCameras();

  const cameraConfig = model.selectedCameraId
    ? new CameraConfig({ deviceId: model.selectedCameraId })
    : CameraConfig.default;
  try {
    await model.session?.startCamera(cameraConfig);
    model.isCameraActive = true;
  } catch (err) {
    showError(err instanceof StreamError ? err.message : String(err));
  }
  render();
}

/**
 * Stops camera capture.
 * Mirrors AppModel.stopCamera().
 *
 * @returns {Promise<void>}
 */
async function stopCamera() {
  try {
    await model.session?.stopCamera();
  } catch (err) {
    showError(err instanceof StreamError ? err.message : String(err));
  }
  model.isCameraActive = false;
  render();
}

/**
 * Sends a ping payload as UTF-8 text.
 * Mirrors AppModel.sendPing().
 *
 * @returns {Promise<void>}
 */
async function sendPing() {
  // In on-demand mode, start the camera now so it warms up in parallel
  // with the ping's network round-trip and agent processing.  The camera
  // will be ready (or nearly so) by the time the agent needs a frame.
  if (model.cameraOnDemand && !model.isCameraActive) startCamera();

  try {
    await model.session?.send('ping');
  } catch (err) {
    showError(err instanceof StreamError ? err.message : String(err));
  }
}

/**
 * Sends arbitrary text as UTF-8.
 * Mirrors AppModel.sendCustom(text:).
 *
 * @param {string} text
 * @returns {Promise<void>}
 */
async function sendCustom(text) {
  if (!text.trim()) return;
  try {
    await model.session?.send(text);
  } catch (err) {
    showError(err instanceof StreamError ? err.message : String(err));
  }
}

/**
 * Launch (or exit) a CloudXR streaming session. The LiveKit room is
 * untouched — entering XR does not drop the agent connection.
 *
 * The always-visible "accept the CloudXR cert" link in the XR panel handles
 * the first-time self-signed-cert prompt; on failure we just surface the
 * error to the panel state and the dbg log.
 */
async function startXR() {
  if (!xrStream) return;
  model.xrError = null;
  const host = model.xrHost.trim() || window.location.hostname || 'localhost';
  const port = Number(model.xrPort) || 48322;

  dbg.info(`startXR() ${host}:${port}`);
  try {
    await xrStream.startXR(host, port);
  } catch (err) {
    dbg.err('startXR failed: ' + (err && err.stack ? err.stack : err));
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
    dbg.err('stopXR failed: ' + (err && err.stack ? err.stack : err));
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Event wiring
// ─────────────────────────────────────────────────────────────────────────────

function wireEvents() {
  // Re-verify CloudXR cert when the user returns from accepting it in the
  // new tab — flips the hint to green without requiring a Launch XR attempt.
  window.addEventListener('focus', () => {
    const link = $('xr-cert-link');
    if (link && link.href) verifyCert(link.href);
  });

  // Config inputs — sync to model on change
  $('host-input').addEventListener('input', (e) => { model.host = e.target.value; });
  $('port-input').addEventListener('input', (e) => { model.port = Number(e.target.value) || 7880; });
  $('token-input').addEventListener('input', (e) => { model.token = e.target.value; });
  $('token-url-input').addEventListener('input', (e) => { model.tokenServerURL = e.target.value; });
  $('identity-input').addEventListener('input', (e) => { model.identity = e.target.value; });

  // Reflect back initial model values into inputs
  $('host-input').value     = model.host;
  $('port-input').value     = model.port;
  $('identity-input').value = model.identity;

  // Populate audio mode dropdown
  const audioModeOptions = [
    { value: MicrophoneMode.SOFTWARE_PROCESSING, label: 'Software (AEC on)' },
    { value: MicrophoneMode.VOICE_PROCESSING,    label: 'Voice Processing' },
    { value: MicrophoneMode.RAW,                 label: 'Raw (no DSP)' },
  ];
  const audioModeSelect = $('audio-mode-select');
  audioModeSelect.innerHTML = audioModeOptions
    .map(o => `<option value="${o.value}">${o.label}</option>`)
    .join('');
  audioModeSelect.value = model.audioMode;
  audioModeSelect.addEventListener('change', (e) => {
    model.audioMode = e.target.value;
  });

  // Connect / disconnect button — toggles based on state
  $('connect-btn').addEventListener('click', () => {
    dbg.info('connect-btn clicked (state=' + model.connectionState + ')');
    if (model.connectionState === ConnectionState.DISCONNECTED) {
      connect();
    } else {
      disconnect();
    }
  });

  // Audio toggle
  $('audio-btn').addEventListener('click', () => {
    if (model.isAudioActive) {
      stopAudio();
    } else {
      startAudio();
    }
  });

  // Camera selector
  $('camera-select').addEventListener('change', (e) => {
    model.selectedCameraId = e.target.value || null;
  });

  // Camera on demand toggle
  $('camera-on-demand')?.addEventListener('change', (e) => {
    model.cameraOnDemand = e.target.checked;
  });

  // Camera toggle
  $('camera-btn').addEventListener('click', () => {
    if (model.isCameraActive) {
      stopCamera();
    } else {
      startCamera();
    }
  });

  // Data channel
  $('ping-btn').addEventListener('click', () => { sendPing(); });

  const msgInput = $('message-input');
  msgInput.addEventListener('input', () => { render(); });   // update send-btn disabled state
  msgInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      const text = msgInput.value;
      sendCustom(text);
      msgInput.value = '';
      render();
    }
  });

  $('send-btn').addEventListener('click', () => {
    const text = msgInput.value;
    sendCustom(text);
    msgInput.value = '';
    render();
  });

  // ── XR Stream ──────────────────────────────────────────────────────────────
  const xrHostInput = $('xr-host-input');
  const xrPortInput = $('xr-port-input');
  xrHostInput.value = model.xrHost;
  xrPortInput.value = model.xrPort;

  xrHostInput.addEventListener('input', (e) => {
    model.xrHost = e.target.value;
  });
  xrPortInput.addEventListener('input', (e) => {
    model.xrPort = Number(e.target.value) || 48322;
  });

  xrStream = new CloudXRStream({
    canvasId: 'xr-canvas',
    dbg,
    onStateChange: (state, detail) => {
      model.xrState = state;
      if (state === 'error') model.xrError = detail ?? null;
      if (state === 'idle')  model.xrError = null;
      dbg.info(`xr state → ${state}${detail ? ': ' + detail : ''}`);

      // Notify the agent stack that an XR client is now connected to CloudXR.
      // OpenXR apps (e.g. LOVR scenes hosted by render-mcp) cannot succeed
      // `xrGetSystem` until a streaming client is present — CloudXR returns
      // XR_ERROR_FORM_FACTOR_UNAVAILABLE otherwise. We send xr.session.started
      // on streaming-start so render-mcp gates its LOVR launch correctly.
      // Failure is swallowed: it's an optional coordination signal and some
      // sessions may not have an agent peer (StreamError.notConnected).
      //
      // We deliberately do NOT publish a corresponding xr.session.stopped:
      // no peer subscribes to it today (the worker only listens for the
      // start), and shipping unused topics into the LiveKit signalling
      // bandwidth just makes the protocol surface harder to reason about.
      // If a peer ever needs the stop signal, add the publish here and
      // document the matching subscriber.
      if (state === 'streaming') {
        model.session?.send(new Uint8Array(0), { topic: 'xr.session.started' })
          .catch((err) => dbg.warn('xr.session.started publish failed: ' + err));
      }

      render();
    },
  });

  $('xr-launch-btn').addEventListener('click', () => {
    dbg.info('xr-launch-btn clicked (state=' + model.xrState + ')');
    if (model.xrState === 'streaming') {
      stopXR();
    } else if (model.xrState === 'idle' || model.xrState === 'error') {
      startXR();
    }
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Bootstrap
// ─────────────────────────────────────────────────────────────────────────────

// Defer until the DOM is parsed so that wireEvents() can grab elements by id.
// In practice index.html ships this as a `<script type="module" defer>` (which
// already implies post-parse execution), but a stray `<script>` without
// `defer` would race the body — the guard makes the contract explicit.
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => { wireEvents(); render(); });
} else {
  wireEvents();
  render();
}
