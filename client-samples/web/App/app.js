/**
 * @fileoverview Sample application — JavaScript equivalent of AppModel.swift + ContentView.swift.
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

const model = {
  // Connection settings
  host:           window.location.hostname || 'localhost',
  port:           7880,
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
};

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
      label.textContent = 'Connecting\u2026';
      break;
    case ConnectionState.CONNECTED:
      dot.classList.add('connected');
      label.textContent = 'Connected';
      break;
    case ConnectionState.RECONNECTING:
      dot.classList.add('reconnecting');
      label.textContent = 'Reconnecting\u2026';
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
      ? 'Connecting\u2026'
      : 'Reconnecting\u2026';
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
    agentLabel.textContent = 'Processing\u2026';
  } else if (model.agentStatus === 'idle') {
    agentDot.className    = 'state-dot connected';   // green
    agentLabel.textContent = 'Idle';
  } else {
    agentDot.className    = 'state-dot disconnected';
    agentLabel.textContent = 'Unknown';
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

  const lkConfig = new LiveKitConfig({
    host:     model.host,
    port:     Number(model.port),
    token:    model.token.trim()   || null,
    tokenURL: model.token.trim()   ? null : resolvedTokenURL(),
  });

  let newSession;
  try {
    newSession = await StreamSession.create(BackendConfiguration.liveKit(lkConfig));
  } catch (err) {
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
  } catch (err) {
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
  try {
    await model.session?.startAudio(new AudioConfig({ mode: model.audioMode }));
    model.isAudioActive = true;
  } catch (err) {
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

// ─────────────────────────────────────────────────────────────────────────────
// Event wiring
// ─────────────────────────────────────────────────────────────────────────────

function wireEvents() {
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
}

// ─────────────────────────────────────────────────────────────────────────────
// Bootstrap
// ─────────────────────────────────────────────────────────────────────────────

wireEvents();
render();
