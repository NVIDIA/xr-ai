// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @fileoverview CloudXR streaming wrapper — owns the WebGL canvas, the
 * `XRSession`, and the CloudXR `Session`.
 *
 * Why it is **not** a `StreamingBackend`: `StreamingBackend` models a
 * bidirectional audio/data/video session (LiveKit). CloudXR is a one-way
 * render stream into WebXR — a fundamentally different lifecycle. Keeping it
 * separate preserves the existing LiveKit `Room` so entering/leaving XR does
 * not disturb the agent connection running in the same page.
 *
 * Usage:
 *   const xr = new CloudXRStream({ canvasId: 'xr-canvas', onStateChange });
 *   await xr.startXR(host, port);
 *   // ... later ...
 *   await xr.stopXR();
 *
 * Frame/session logic adapted from
 * https://github.com/NVIDIA/cloudxr-js examples/simple/src/main.ts.
 *
 * @module App/cloudxr
 */

import { createSession, SessionState } from '@nvidia/cloudxr';

/** @typedef {'idle'|'connecting'|'streaming'|'stopping'|'error'} XRState */

// IWER = Immersive Web Emulator Runtime. Loaded only when the browser has no
// real XR device (desktop dev). Polyfills a fake Quest 3 so requestSession()
// succeeds. On an actual Quest these URLs are never hit.
//
// Pinned versions + SRI match the upstream cloudxr-js/examples/simple.
const _IWER_VERSION       = '2.2.1';
const _IWER_DEVUI_VERSION = '2.2.0';
const _IWER_URL = `https://unpkg.com/iwer@${_IWER_VERSION}/build/iwer.min.js`;
const _IWER_SRI = 'sha384-3G2UIBh0RX9Imd3PFwcHyXbqRYAeQo9FDMgQTOLcflo9H6LDHaxADB24vKC3b+OY';
const _IWER_DEVUI_URL = `https://unpkg.com/@iwer/devui@${_IWER_DEVUI_VERSION}/build/iwer-devui.min.js`;
const _IWER_DEVUI_SRI = 'sha384-gPhqycVT+bNyiNIH8kMEWFjaysw6xH9NGYwuduRzK71Ro0Tp3hXByxqAI9sWrc9T';

/**
 * Load a script tag with SRI. Resolves on load, rejects on error.
 * @param {string} src
 * @param {string} integrity
 */
function _loadScript(src, integrity) {
  return new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = src;
    s.async = true;
    s.integrity = integrity;
    s.crossOrigin = 'anonymous';
    s.onload  = () => resolve();
    s.onerror = () => reject(new Error(`failed to load ${src}`));
    document.head.appendChild(s);
  });
}

/** Module-scope: once one call succeeds, subsequent calls are no-ops. */
let _iwerLoaded = false;

/**
 * If the browser lacks WebXR immersive support, load IWER + DevUI from CDN
 * and install a fake Quest 3 XR device. Idempotent.
 *
 * @returns {Promise<{supportsImmersive: boolean, iwerLoaded: boolean}>}
 */
async function _ensureImmersiveSupport() {
  let supportsImmersive = false;
  if ('xr' in navigator) {
    try {
      const vr = await navigator.xr.isSessionSupported?.('immersive-vr');
      const ar = await navigator.xr.isSessionSupported?.('immersive-ar');
      supportsImmersive = Boolean(vr || ar);
    } catch { /* swallow */ }
  }
  if (supportsImmersive) return { supportsImmersive: true, iwerLoaded: _iwerLoaded };

  await _loadScript(_IWER_URL, _IWER_SRI);
  const IWER = /** @type {any} */ (window).IWER;
  if (!IWER) throw new Error('IWER script loaded but global not found');

  try {
    await _loadScript(_IWER_DEVUI_URL, _IWER_DEVUI_SRI);
  } catch { /* devui is optional */ }

  const device = new IWER.XRDevice(IWER.metaQuest3);
  const DevUI = /** @type {any} */ (window).IWER_DevUI?.DevUI;
  if (DevUI) device.installDevUI(DevUI);
  await device.installRuntime();
  /** @type {any} */ (window).xrDevice = device;

  _iwerLoaded = true;
  return { supportsImmersive: true, iwerLoaded: true };
}

export class CloudXRStream {
  /**
   * @param {{
   *   canvasId: string,
   *   onStateChange?: (state: XRState, detail?: string) => void,
   *   dbg?: { info: (m:string)=>void, warn: (m:string)=>void, err: (m:string)=>void },
   * }} opts
   */
  constructor(opts) {
    this._canvasId = opts.canvasId;
    this._onStateChange = opts.onStateChange || (() => {});
    this._dbg = opts.dbg || { info: () => {}, warn: () => {}, err: () => {} };

    /** @type {XRSession|null}               */ this._xrSession = null;
    /** @type {WebGL2RenderingContext|null}  */ this._gl = null;
    /** @type {XRWebGLLayer|null}            */ this._baseLayer = null;
    /** @type {import('@nvidia/cloudxr').Session|null} */ this._cxrSession = null;
  }

  // ── public API ────────────────────────────────────────────────────────────

  /**
   * Start a CloudXR session: request an immersive-vr WebXR session, build
   * the WebGL bridge, hand it all to CloudXR, and begin the render loop.
   *
   * Resolves as soon as `session.connect()` has been called — actual
   * "streaming" arrives via onStateChange('streaming') when the first
   * frame lands. Rejects if WebXR/WebGL setup fails.
   *
   * @param {string} host   CloudXR server IP/hostname (WSS proxy endpoint).
   * @param {number} port   WSS proxy port (48322 by default on HTTPS pages).
   */
  async startXR(host, port) {
    if (this._xrSession) throw new Error('XR session already running');
    this._emitState('connecting');

    try {
      const { iwerLoaded } = await _ensureImmersiveSupport();
      if (iwerLoaded) this._dbg.info('cloudxr: IWER emulator active (desktop dev)');
      await this._initializeWebGL();
      await this._createXRSession();
      const referenceSpace = await this._getReferenceSpace('local-floor');
      await this._createCloudXRSession(host, port, referenceSpace);

      this._cxrSession.connect();
      this._dbg.info(`cloudxr: connect() issued to ${host}:${port}`);
    } catch (err) {
      this._emitState('error', String(err?.message ?? err));
      await this._safeTeardown();
      throw err;
    }
  }

  /** Stop the current session. Idempotent. */
  async stopXR() {
    if (!this._xrSession && !this._cxrSession) return;
    this._emitState('stopping');
    try {
      await this._xrSession?.end();
    } catch (err) {
      this._dbg.warn('cloudxr: xrSession.end() threw: ' + err);
    }
    await this._safeTeardown();
    this._emitState('idle');
  }

  // ── internals ─────────────────────────────────────────────────────────────

  /** @param {XRState} state */
  _emitState(state, detail) {
    try { this._onStateChange(state, detail); } catch { /* swallow */ }
  }

  async _initializeWebGL() {
    const canvas = /** @type {HTMLCanvasElement} */ (document.getElementById(this._canvasId));
    if (!canvas) throw new Error(`canvas #${this._canvasId} not found`);
    const gl = canvas.getContext('webgl2', {
      alpha: true,
      depth: true,
      stencil: false,
      antialias: false,
      failIfMajorPerformanceCaveat: true,
      powerPreference: 'high-performance',
      premultipliedAlpha: false,
      preserveDrawingBuffer: false,
    });
    if (!gl) throw new Error('WebGL2 context unavailable');
    await gl.makeXRCompatible();
    this._gl = gl;
  }

  async _createXRSession() {
    if (!navigator.xr) throw new Error('WebXR not supported by this browser');
    // The mic-sphere scene renders a single sphere at a fixed world position
    // and reads no hand/body poses, so we ask only for what we actually use.
    // Keep local-floor so y=1.6 means "head height above the floor" rather
    // than "1.6 m above wherever the headset booted up".
    const opts = { requiredFeatures: ['local-floor'] };
    // AR first: an immersive-ar session lets the OpenXR runtime expose
    // ALPHA_BLEND environment blend mode, so LOVR's alpha=0 clears resolve
    // to real-world passthrough on a headset (or transparent-over-the-page
    // under IWER).  immersive-vr is opaque-by-spec, so it's a fallback only
    // for browsers/devices that lack AR support.
    try {
      this._xrSession = await navigator.xr.requestSession('immersive-ar', opts);
    } catch (err) {
      this._dbg.warn('cloudxr: immersive-ar failed, falling back to immersive-vr: ' + err);
      this._xrSession = await navigator.xr.requestSession('immersive-vr', opts);
    }

    // Values mirrored from cloudxr-js/examples/helpers/PerformanceProfiles.ts
    // (tuned for Quest 3). framebufferScaleFactor = 1.5 is required — leaving
    // it at the spec default (1.0) causes Quest Browser to use *more* frame
    // time than scaling up, and under IWER the XR framebuffer ends up
    // undersized to the point that nothing visible makes it to the DevUI
    // preview overlay.
    this._baseLayer = new XRWebGLLayer(this._xrSession, /** @type {WebGL2RenderingContext} */ (this._gl), {
      alpha: true,
      antialias: false,
      depth: true,
      ignoreDepthValues: false,
      stencil: false,
      framebufferScaleFactor: 1.5,
    });
    if ('fixedFoveation' in this._baseLayer) {
      // MEDIUM — balance between performance and visual quality.
      /** @type {any} */ (this._baseLayer).fixedFoveation = 0.666;
    }
    this._xrSession.updateRenderState({ baseLayer: this._baseLayer });
    this._xrSession.addEventListener('end', () => this._handleSessionEnd());
    this._xrSession.requestAnimationFrame((ts, frame) => this._onXRFrame(ts, frame));
  }

  /**
   * @param {XRReferenceSpaceType} type
   * @returns {Promise<XRReferenceSpace>}
   */
  async _getReferenceSpace(type) {
    const s = /** @type {XRSession} */ (this._xrSession);
    try                 { return await s.requestReferenceSpace(type);          }
    catch { try         { return await s.requestReferenceSpace('local-floor'); }
      catch { try       { return await s.requestReferenceSpace('local');       }
        catch           { return await s.requestReferenceSpace('viewer');      } } }
  }

  /**
   * @param {string} host
   * @param {number} port
   * @param {XRReferenceSpace} referenceSpace
   */
  async _createCloudXRSession(host, port, referenceSpace) {
    const secure = window.location.protocol === 'https:';
    /** @type {import('@nvidia/cloudxr').SessionOptions} */
    const sessionOptions = {
      serverAddress: host,
      serverPort:    port,
      useSecureConnection: secure,
      gl: /** @type {WebGL2RenderingContext} */ (this._gl),
      // Defaults mirrored from cloudxr-js/examples/simple (index.html) — these
      // reflect the settings the reference app ships with, which is the
      // configuration we know works against CloudXR Runtime under IWER.
      // perEyeWidth must be a multiple of 16, perEyeHeight a multiple of 64.
      codec: 'av1',
      perEyeWidth:  2048,
      perEyeHeight: 1792,
      deviceFrameRate: 90,
      maxStreamingBitrateKbps: 150_000,
      referenceSpace,
      enablePoseSmoothing: true,
      posePredictionFactor: 1.0,
      enableTexSubImage2D: false,
      useQuestColorWorkaround: false,
      telemetry: {
        enabled: true,
        appInfo: { version: '0.1.0', product: 'xr-ai web sample' },
      },
    };

    const delegates = {
      onStreamStarted: () => {
        this._dbg.info('cloudxr: stream started');
        this._emitState('streaming');
      },
      onStreamStopped: (err) => {
        if (err) {
          const code = err.code ? ` (0x${err.code.toString(16).toUpperCase()})` : '';
          this._dbg.err('cloudxr: stream stopped: ' + err.message + code);
          this._emitState('error', err.message + code);
        } else {
          this._dbg.info('cloudxr: stream stopped normally');
        }
        // End the XR session so the headset returns to 2D browser view;
        // _handleSessionEnd cleans everything up. If end() rejects (it can
        // throw when the session is already ending), null the handle so the
        // next startXR() doesn't hit "session already running".
        this._xrSession?.end().catch(() => { this._xrSession = null; });
      },
      onServerMessageReceived: (/** @type {Uint8Array} */ _msg) => {},
      onWebGLStateChangeBegin: () => {},
      onWebGLStateChangeEnd:   () => {},
    };

    this._cxrSession = createSession(sessionOptions, delegates);
  }

  /**
   * Per-frame render tick. Forwards tracking to the server, asks the SDK to
   * render the received video into the XR base layer's framebuffer.
   *
   * @param {DOMHighResTimeStamp} timestamp
   * @param {XRFrame} frame
   */
  _onXRFrame(timestamp, frame) {
    const xr = this._xrSession;
    if (!xr) return;
    xr.requestAnimationFrame((ts, f) => this._onXRFrame(ts, f));

    const cxr = this._cxrSession;
    if (!cxr || cxr.state !== SessionState.Connected) return;

    try {
      cxr.sendTrackingStateToServer(timestamp, frame);
      const gl = /** @type {WebGL2RenderingContext} */ (this._gl);
      const layer = /** @type {XRWebGLLayer} */ (this._baseLayer);
      gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
      cxr.render(timestamp, frame, layer);
    } catch (err) {
      this._dbg.err('cloudxr: frame error: ' + err);
    }
  }

  // TODO(cleanup): _handleSessionEnd and _safeTeardown duplicate the same
  // disconnect/null sequence — only the trailing _emitState('idle') differs.
  // Worth folding once we add a state machine for retry, but keeping them
  // explicit for now to avoid a "tear down again from inside a teardown"
  // recursion if the SDK's disconnect ever ends up calling back into us.
  _handleSessionEnd() {
    // XRSession ended — for any reason (user exit, headset removed, error,
    // stream failure). Tear everything down and return to idle.
    try { this._cxrSession?.disconnect(); } catch { /* swallow */ }
    this._cxrSession = null;
    this._baseLayer  = null;
    this._xrSession  = null;
    this._gl         = null;
    this._emitState('idle');
  }

  async _safeTeardown() {
    try { this._cxrSession?.disconnect(); } catch { /* swallow */ }
    this._cxrSession = null;
    this._baseLayer  = null;
    this._xrSession  = null;
    this._gl         = null;
  }
}
