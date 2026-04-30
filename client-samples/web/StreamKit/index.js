// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @fileoverview Public API index for StreamKit (web).
 *
 * Re-exports every symbol a consumer needs to build a streaming application.
 * Import from this file rather than from individual module paths so that
 * internal layout can change without breaking call-sites.
 *
 * @example
 * import {
 *   StreamSession,
 *   ConnectionState,
 *   StreamError,
 *   AudioConfig, MicrophoneMode,
 *   CameraConfig, CameraFacing,
 *   SessionConfig,
 *   BackendConfiguration, LiveKitConfig,
 * } from '/StreamKit/index.js';
 *
 * @module StreamKit
 */

export { StreamSession }                          from './StreamSession.js';
export { ConnectionState }                        from './ConnectionState.js';
export { StreamError }                            from './StreamError.js';
export { AudioConfig, MicrophoneMode }            from './Config/AudioConfig.js';
export { CameraConfig, CameraFacing }             from './Config/CameraConfig.js';
export { SessionConfig }                          from './Config/SessionConfig.js';
export { BackendConfiguration, LiveKitConfig }    from './Config/BackendConfiguration.js';
