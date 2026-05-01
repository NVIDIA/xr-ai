// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Re-exports the pieces of @nvidia/cloudxr needed by /client-samples/web/.
 * Everything else in the SDK is still tree-shaken into the bundle because
 * the SDK ships a single CommonJS module — but we only need these symbols
 * in the page's own code.
 */
export { createSession, SessionState } from '@nvidia/cloudxr';
