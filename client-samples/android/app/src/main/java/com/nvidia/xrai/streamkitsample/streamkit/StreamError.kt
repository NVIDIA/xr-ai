// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.xrai.streamkitsample.streamkit

/**
 * Errors thrown by [StreamSession] and its backends.
 *
 * Mirror of Swift `StreamError` and the web `StreamError` class.
 */
sealed class StreamError(message: String) : Exception(message) {

    /** Host string could not be turned into a valid URL. */
    class InvalidHost(host: String) :
        StreamError("'$host' is not a valid hostname.")

    /** An operation that requires an active connection was called while disconnected. */
    object NotConnected :
        StreamError("Not connected. Call connect() first.")

    /** Neither a token nor a tokenURL was provided to the LiveKit backend. */
    object MissingToken :
        StreamError("Provide a token or tokenURL in LiveKitConfig.")

    /** Token-server request failed or returned an unparseable body. */
    class TokenFetchFailed(url: String, detail: String? = null) :
        StreamError(buildString {
            append("Token fetch failed: ").append(url)
            if (!detail.isNullOrBlank()) append(" — ").append(detail)
        })

    /** startCamera() was called while not connected. */
    object CameraRequiresConnection :
        StreamError("Connect before starting the camera.")
}
