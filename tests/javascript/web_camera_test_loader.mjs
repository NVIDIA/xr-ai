// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const LIVEKIT_URL = 'mock:livekit-client';
const APP_CORE_URL = 'mock:app-core';
const WEB_CLIENT_ROOT = new URL('../../client-samples/web/', import.meta.url).href;
const STREAMKIT_INDEX_URL = new URL(
  '../../client-samples/web/StreamKit/index.js', import.meta.url,
).href;

function isWebClientJavaScript(url) {
  return url.startsWith(WEB_CLIENT_ROOT) && new URL(url).pathname.endsWith('.js');
}

export async function resolve(specifier, context, nextResolve) {
  if (specifier === 'livekit-client') {
    return { url: LIVEKIT_URL, shortCircuit: true };
  }
  if (specifier === '/App/core.js') {
    return { url: APP_CORE_URL, shortCircuit: true };
  }
  if (specifier === '/StreamKit/index.js') {
    return { url: STREAMKIT_INDEX_URL, shortCircuit: true };
  }
  return nextResolve(specifier, context);
}

export async function load(url, context, nextLoad) {
  if (url === LIVEKIT_URL) {
    return {
      format: 'module',
      shortCircuit: true,
      source: `
        export class Room {
          constructor() {
            if (!globalThis.__livekitRoom) throw new Error('LiveKit room mock is not installed');
            return globalThis.__livekitRoom;
          }
        }
        // Published SDK2.17.2 camera preset references, not simulated encoding
        // computation: the backend must preserve geometry and clamp budgets.
        export const VideoPresets = {
          h180: { width: 320, height: 180, encoding: { maxBitrate: 160000, maxFramerate: 20 } },
          h360: { width: 640, height: 360, encoding: { maxBitrate: 450000, maxFramerate: 20 } },
        };
        export const VideoPresets43 = {
          h180: { width: 240, height: 180, encoding: { maxBitrate: 125000, maxFramerate: 20 } },
          h360: { width: 480, height: 360, encoding: { maxBitrate: 330000, maxFramerate: 20 } },
        };
        export const RoomEvent = Object.freeze({
          ConnectionStateChanged: 'connectionStateChanged',
          DataReceived: 'dataReceived',
          TrackPublished: 'trackPublished',
          TrackSubscribed: 'trackSubscribed',
          TrackUnsubscribed: 'trackUnsubscribed',
        });
        export const Track = Object.freeze({
          Kind: Object.freeze({ Audio: 'audio' }),
          Source: Object.freeze({ Camera: 'camera', Microphone: 'microphone' }),
        });
        export async function createLocalAudioTrack() {
          throw new Error('audio capture is not expected in camera tests');
        }
      `,
    };
  }

  if (url === APP_CORE_URL) {
    return {
      format: 'module',
      shortCircuit: true,
      source: `
        export const $ = id => globalThis.__elements.get(id);
        export const createBaseModel = () => globalThis.__appBaseModel;
        export const renderBase = () => {};
        export const enumerateCameras = async () => {};
        export const connect = async () => {};
        export async function disconnect(model) {
          await model.session?.disconnect();
          model.session = null;
          model.isCameraActive = false;
        }
        export const startAudio = async () => {};
        export const stopAudio = async () => {};
        export async function startCamera(model) {
          await model.session?.startCamera({ deviceId: model.selectedCameraId });
          model.isCameraActive = true;
        }
        export async function stopCamera(model) {
          await model.session?.stopCamera();
          model.isCameraActive = false;
        }
        export async function setCameraMode(model, mode, { startCamera, stopCamera }) {
          model.cameraMode = mode;
          if (mode === 'live') await startCamera();
          else if (model.isCameraActive) await stopCamera();
        }
        export const sendCustom = async () => {};
        export function wireBaseEvents(model, actions) {
          globalThis.__appModel = model;
          globalThis.__appActions = actions;
        }
      `,
    };
  }

  if (isWebClientJavaScript(url)) {
    return nextLoad(url, { ...context, format: 'module' });
  }

  return nextLoad(url, context);
}
