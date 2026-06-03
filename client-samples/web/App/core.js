// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @fileoverview Shared StreamKit app core — logic common to all web clients.
 *
 * Exports parameterised versions of every action and render helper so that
 * each client app (web/App/app.js, web-xr/App/app.js, …) can import what it
 * needs and extend only what it needs to change.  No module-level state lives
 * here — callers own their `model` object and pass it in.
 *
 * All web clients (web/App/app.js, web-xr/App/app.js, …) import from this
 * module and extend only what they need to change.
 *
 * @module App/core
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

// ─────────────────────────────────────────────────────────────────────────────
// Pure helpers
// ─────────────────────────────────────────────────────────────────────────────

/** @param {string} id @returns {HTMLElement} */
export const $ = (id) => document.getElementById(id);

/** @param {string} s @returns {string} */
export function escapeHtml(s) {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/** @returns {boolean} */
export function isQuestBrowser() {
  return /OculusBrowser/.test(navigator.userAgent || '');
}

// ─────────────────────────────────────────────────────────────────────────────
// Camera enumeration
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Queries available video input devices and updates `model.cameras`.
 *
 * @param {object} model
 * @param {() => void} render
 * @returns {Promise<void>}
 */
export async function enumerateCameras(model, render) {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  try {
    let devices = await navigator.mediaDevices.enumerateDevices();
    let cameras = devices.filter(d => d.kind === 'videoinput');

    if (cameras.length > 0 && !cameras.some(d => d.deviceId)) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        stream.getTracks().forEach(t => t.stop());
        devices  = await navigator.mediaDevices.enumerateDevices();
        cameras  = devices.filter(d => d.kind === 'videoinput');
      } catch { /* permission denied — proceed with anonymous devices */ }
    }

    // Accessing a non-passthrough camera on Quest breaks the mediaDevices stack
    // until headset reboot, so this filters the camera picker to passthrough only.
    if (isQuestBrowser()) {
      cameras = cameras.filter(d => !/\bfront\b/i.test(d.label));
    }

    const list = cameras.map((d, i) => ({
      deviceId: d.deviceId,
      label:    d.label || `Camera ${i + 1}`,
    }));

    model.cameras = list;
    if (list.length > 0 && !list.some(c => c.deviceId === model.selectedCameraId)) {
      model.selectedCameraId = list[0].deviceId;
    }
    render();
  } catch { /* enumerateDevices not available */ }
}

// ─────────────────────────────────────────────────────────────────────────────
// Base model factory
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Returns the base model fields shared by all clients.
 * Callers spread this and add their own extensions.
 *
 * @returns {object}
 */
export function createBaseModel() {
  // Hub serves the page and the wss:// /rtc proxy on the same origin. Default
  // port + scheme to the page's own so the demo Just Works for both the local
  // https://localhost:8080 dev flow and a remote https://host:8080 deployment.
  const isPageSecure = typeof window !== 'undefined'
    && window.location.protocol === 'https:';
  const defaultPort = isPageSecure
    ? (Number(window.location.port) || 443)
    : 8080;
  return {
    host:            window.location.hostname || 'localhost',
    port:            defaultPort,
    secure:          isPageSecure,
    tokenServerURL:  '',
    token:           '',
    identity:        'web-client',
    audioMode:       MicrophoneMode.VOICE_PROCESSING,
    /** @type {import('/StreamKit/index.js').StreamSession|null} */
    session:          null,
    connectionState:  ConnectionState.DISCONNECTED,
    isAudioActive:    false,
    isCameraActive:   false,
    /** @type {Array<{deviceId: string, label: string}>} */
    cameras:           [],
    /** @type {string|null} */
    selectedCameraId:  null,
    /** @type {string|null} */
    agentStatus:       null,
    cameraOnDemand:    false,
    /** @type {Array<{id: string, text: string, timestamp: Date}>} */
    receivedMessages:  [],
    /** @type {string|null} */
    lastError:         null,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Base render
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Renders all base DOM elements from `model`.  Clients with extra panels call
 * this first, then render their own additions.
 *
 * @param {object} model
 */
export function renderBase(model) {
  const state = model.connectionState;
  const isDisconnected  = state === ConnectionState.DISCONNECTED;
  const isConnected     = state === ConnectionState.CONNECTED;
  const isTransitioning = state === ConnectionState.CONNECTING
                       || state === ConnectionState.RECONNECTING;

  // ── State badge ────────────────────────────────────────────────────────────
  const dot   = $('state-dot');
  const label = $('state-label');

  dot.className = 'state-dot';
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

  // ── Config inputs ──────────────────────────────────────────────────────────
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
    connectBtn.textContent = state === ConnectionState.CONNECTING ? 'Connecting…' : 'Reconnecting…';
    connectBtn.className   = 'btn btn-secondary';
    connectBtn.disabled    = true;
  } else {
    connectBtn.textContent = 'Disconnect';
    connectBtn.className   = 'btn btn-destructive';
    connectBtn.disabled    = false;
  }

  // ── Audio ──────────────────────────────────────────────────────────────────
  const audioBtn        = $('audio-btn');
  const audioStatus     = $('audio-status');
  const audioModeSelect = $('audio-mode-select');

  audioBtn.disabled        = !isConnected;
  audioModeSelect.disabled = model.isAudioActive;
  if (model.isAudioActive) {
    audioBtn.textContent    = 'Stop Microphone';
    audioBtn.className      = 'btn btn-destructive';
    audioStatus.textContent = 'Live';
    audioStatus.className   = 'status-text status-active';
  } else {
    audioBtn.textContent    = 'Start Microphone';
    audioBtn.className      = 'btn btn-secondary';
    audioStatus.textContent = isConnected ? 'Idle' : 'Not connected';
    audioStatus.className   = 'status-text status-idle';
  }

  // ── Camera ─────────────────────────────────────────────────────────────────
  const cameraBtn    = $('camera-btn');
  const cameraStatus = $('camera-status');

  cameraBtn.disabled = !isConnected;
  if (model.isCameraActive) {
    cameraBtn.textContent    = 'Stop Camera';
    cameraBtn.className      = 'btn btn-destructive';
    cameraStatus.textContent = 'Streaming';
    cameraStatus.className   = 'status-text status-active';
  } else {
    cameraBtn.textContent    = 'Start Camera';
    cameraBtn.className      = 'btn btn-secondary';
    cameraStatus.textContent = isConnected ? 'Idle' : 'Not connected';
    cameraStatus.className   = 'status-text status-idle';
  }

  // ── Camera on demand toggle ────────────────────────────────────────────────
  const codCheckbox = $('camera-on-demand');
  if (codCheckbox) codCheckbox.checked = model.cameraOnDemand;

  // ── Camera selector ────────────────────────────────────────────────────────
  const selectRow = $('camera-select-row');
  const camSelect = $('camera-select');

  if (model.cameras.length >= 1) {
    selectRow.style.display = '';
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

  // ── Agent status (optional — some clients replace this with their own UI) ──
  const agentDot   = $('agent-status-dot');
  const agentLabel = $('agent-status-label');

  if (agentDot && agentLabel) {
    if (!isConnected) {
      agentDot.className     = 'state-dot disconnected';
      agentLabel.textContent = '—';
    } else if (model.agentStatus === 'processing') {
      agentDot.className     = 'state-dot connecting';
      agentLabel.textContent = 'Processing…';
    } else if (model.agentStatus === 'idle') {
      agentDot.className     = 'state-dot connected';
      agentLabel.textContent = 'Idle';
    } else {
      agentDot.className     = 'state-dot disconnected';
      agentLabel.textContent = 'Unknown';
    }
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

// ─────────────────────────────────────────────────────────────────────────────
// Actions
// ─────────────────────────────────────────────────────────────────────────────

/** @param {object} model @returns {string} */
export function resolvedTokenURL(model) {
  if (model.tokenServerURL.trim() !== '') return model.tokenServerURL.trim();
  return `${window.location.origin}/token`;
}

/**
 * Connects to the LiveKit room.
 *
 * @param {object} model
 * @param {{
 *   render: () => void,
 *   showError: (msg: string) => void,
 *   enumerateCameras: () => Promise<void>,
 *   startCamera?: () => Promise<void>,
 *   stopCamera: () => Promise<void>,
 *   onStateChange?: (state: string) => void,
 *   onDataReceived?: (topic: string, data: Uint8Array) => boolean,
 * }} opts
 *   `startCamera` / `stopCamera` are caller-supplied wrappers used by the
 *   on-demand `clientControl` handler so client-specific side effects (e.g.
 *   the local `<video>` preview in `web/App/app.js`) run on agent-triggered
 *   start / stop. When omitted, the on-demand start falls back to the bare
 *   transport-level `startCamera` (sufficient for clients without a local
 *   preview, such as `web-xr`).
 */
export async function connect(model, {
  render, showError, enumerateCameras: _ec,
  startCamera: _startCamera, stopCamera: _sc,
  onStateChange, onDataReceived,
}) {
  model.lastError        = null;
  model.receivedMessages = [];

  const lkConfig = new LiveKitConfig({
    host:     model.host,
    port:     Number(model.port),
    secure:   model.secure ?? false,
    token:    model.token.trim()  || null,
    tokenURL: model.token.trim()  ? null : resolvedTokenURL(model),
  });

  let newSession;
  try {
    newSession = await StreamSession.create(BackendConfiguration.liveKit(lkConfig));
  } catch (err) {
    showError(err instanceof StreamError ? err.message : String(err));
    render();
    return;
  }

  newSession.onConnectionStateChanged = (state) => {
    const wasCameraActive = model.isCameraActive;
    model.connectionState = state;
    if (state === ConnectionState.CONNECTED) {
      // Enumerate cameras on connect so the selector is populated before the
      // user starts the camera for the first time.
      _ec?.();
    } else if (state === ConnectionState.DISCONNECTED) {
      if (wasCameraActive) _sc?.();
      model.isAudioActive  = false;
      model.isCameraActive = false;
      model.agentStatus    = null;
    } else if (state === ConnectionState.RECONNECTING) {
      // Stop the camera when the connection drops so the server and client
      // both start from a known-off state after reconnect.
      if (model.isCameraActive) _sc?.();
    }
    onStateChange?.(state);
    render();
  };

  newSession.onAgentStatus = (status) => {
    model.agentStatus = status;
    render();
  };

  newSession.onDataReceived = (topic, data) => {
    // Let the caller intercept topics first (returns true to suppress list append).
    if (onDataReceived?.(topic, data)) return;

    if (topic === 'clientControl') {
      if (model.cameraOnDemand) {
        try {
          const { action } = JSON.parse(new TextDecoder().decode(data));
          const reportAsyncError = (err) =>
            showError(err instanceof StreamError ? err.message : String(err));
          if (action === 'startCamera' && !model.isCameraActive) {
            // Prefer the caller's wrapper so client-specific side effects
            // (e.g. acquiring the local <video> preview stream in app.js)
            // run on agent-triggered start. Wrappers are async — surface
            // rejections via showError so agent-triggered failures aren't
            // swallowed as unhandled promise rejections.
            (_startCamera
              ? _startCamera()
              : startCamera(model, { render, showError, enumerateCameras: _ec })
            )?.catch?.(reportAsyncError);
          }
          if (action === 'stopCamera'  &&  model.isCameraActive) {
            _sc?.()?.catch?.(reportAsyncError);
          }
        } catch { /* malformed — ignore */ }
      }
      return;
    }

    const text = (() => {
      try { return new TextDecoder().decode(data); }
      catch { return `[${data.byteLength ?? data.length} bytes binary]`; }
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
    audio:    new AudioConfig({ mode: MicrophoneMode.DISABLED }),
    camera:   CameraConfig.disabled,
    identity: model.identity,
  });

  try {
    await newSession.connect(sessionConfig);
  } catch (err) {
    showError(err instanceof StreamError ? err.message : String(err));
    model.session         = null;
    model.connectionState = ConnectionState.DISCONNECTED;
  }

  render();
}

/** @param {object} model @param {() => void} render */
export async function disconnect(model, render) {
  await model.session?.disconnect();
  model.session          = null;
  model.connectionState  = ConnectionState.DISCONNECTED;
  model.isAudioActive    = false;
  model.isCameraActive   = false;
  render();
}

/**
 * @param {object} model
 * @param {() => void} render
 * @param {(msg: string) => void} showError
 */
export async function startAudio(model, render, showError) {
  if (!navigator.mediaDevices) {
    showError(window.isSecureContext
      ? 'mediaDevices API unavailable in this browser'
      : 'Mic/camera require a secure context (https:// or localhost).');
    return;
  }
  try {
    await model.session?.startAudio(new AudioConfig({ mode: model.audioMode }));
    model.isAudioActive = true;
  } catch (err) {
    showError(err instanceof StreamError ? err.message : String(err));
  }
  render();
}

/** @param {object} model @param {() => void} render @param {(msg: string) => void} showError */
export async function stopAudio(model, render, showError) {
  try {
    await model.session?.stopAudio();
  } catch (err) {
    showError(err instanceof StreamError ? err.message : String(err));
  }
  model.isAudioActive = false;
  render();
}

/**
 * @param {object} model
 * @param {{ render: () => void, showError: (msg: string) => void, enumerateCameras: () => Promise<void> }} opts
 */
export async function startCamera(model, { render, showError, enumerateCameras: _ec }) {
  if (!navigator.mediaDevices) {
    showError(window.isSecureContext
      ? 'mediaDevices API unavailable in this browser'
      : 'Mic/camera require a secure context (https:// or localhost).');
    return;
  }
  await _ec?.();
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

/** @param {object} model @param {() => void} render @param {(msg: string) => void} showError */
export async function stopCamera(model, render, showError) {
  try {
    await model.session?.stopCamera();
  } catch (err) {
    showError(err instanceof StreamError ? err.message : String(err));
  }
  model.isCameraActive = false;
  render();
}

/**
 * @param {object} model
 * @param {() => Promise<void>} _startCamera  bound startCamera for the caller
 */
export async function sendPing(model, _startCamera) {
  if (model.cameraOnDemand && !model.isCameraActive) _startCamera();
  try {
    await model.session?.send('ping');
  } catch { /* ignore */ }
}

/** @param {object} model @param {string} text @param {(msg: string) => void} showError */
export async function sendCustom(model, text, showError) {
  if (!text.trim()) return;
  try {
    await model.session?.send(text);
  } catch (err) {
    showError(err instanceof StreamError ? err.message : String(err));
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Base event wiring
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Wires all base DOM events.  Callers invoke this, then wire their own extras.
 *
 * @param {object} model
 * @param {{
 *   connect:     () => void,
 *   disconnect:  () => void,
 *   startAudio:  () => void,
 *   stopAudio:   () => void,
 *   startCamera: () => void,
 *   stopCamera:  () => void,
 *   sendPing:    () => void,
 *   sendCustom:  (text: string) => void,
 * }} actions
 */
export function wireBaseEvents(model, actions) {
  const { connect, disconnect, startAudio, stopAudio, startCamera, stopCamera, sendPing, sendCustom } = actions;

  $('host-input').addEventListener('input', (e) => { model.host = e.target.value; });
  $('port-input').addEventListener('input', (e) => { model.port = Number(e.target.value) || 8080; });
  $('token-input').addEventListener('input', (e) => { model.token = e.target.value; });
  $('token-url-input').addEventListener('input', (e) => { model.tokenServerURL = e.target.value; });
  $('identity-input').addEventListener('input', (e) => { model.identity = e.target.value; });

  $('host-input').value     = model.host;
  $('port-input').value     = model.port;
  $('identity-input').value = model.identity;

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
  audioModeSelect.addEventListener('change', (e) => { model.audioMode = e.target.value; });

  $('connect-btn').addEventListener('click', () => {
    if (model.connectionState === ConnectionState.DISCONNECTED) connect(); else disconnect();
  });

  $('audio-btn').addEventListener('click', () => {
    if (model.isAudioActive) stopAudio(); else startAudio();
  });

  $('camera-select').addEventListener('change', (e) => {
    model.selectedCameraId = e.target.value || null;
  });

  $('camera-on-demand')?.addEventListener('change', (e) => {
    model.cameraOnDemand = e.target.checked;
  });

  $('camera-btn').addEventListener('click', () => {
    if (model.isCameraActive) stopCamera(); else startCamera();
  });

  $('ping-btn').addEventListener('click', () => { sendPing(); });

  const msgInput = $('message-input');
  msgInput.addEventListener('input', () => {
    $('send-btn').disabled = !model.session || msgInput.value.trim() === '';
  });
  msgInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendCustom(msgInput.value);
      msgInput.value = '';
    }
  });
  $('send-btn').addEventListener('click', () => {
    sendCustom(msgInput.value);
    msgInput.value = '';
  });
}
