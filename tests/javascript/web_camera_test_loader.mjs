// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const LIVEKIT_URL = 'mock:livekit-client';
const APP_CORE_URL = 'mock:app-core';
const WEB_CLIENT_ROOT = new URL('../../client-samples/web/', import.meta.url).href;

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
