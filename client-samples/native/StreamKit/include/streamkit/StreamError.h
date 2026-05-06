// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdexcept>
#include <string>

namespace streamkit {

/// Base class for all StreamKit errors.
///
/// Mirror of Swift `StreamError` and Kotlin `StreamError`.
class StreamError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/// Thrown when a host string cannot be turned into a valid WebSocket URL.
class InvalidHostError : public StreamError {
public:
    explicit InvalidHostError(const std::string& host)
        : StreamError("'" + host + "' is not a valid hostname.") {}
};

/// Thrown when an operation that requires an active connection is called
/// while disconnected.
class NotConnectedError : public StreamError {
public:
    NotConnectedError() : StreamError("Not connected. Call Connect() first.") {}
};

/// Thrown when neither a token nor a tokenURL was provided to the LiveKit backend.
class MissingTokenError : public StreamError {
public:
    MissingTokenError() : StreamError("Provide a token or token_url in LiveKitConfig.") {}
};

/// Thrown when the token-server request fails or returns an unparseable body.
class TokenFetchFailedError : public StreamError {
public:
    explicit TokenFetchFailedError(const std::string& url)
        : StreamError("Failed to fetch token from " + url + ".") {}
};

/// Thrown when StartCamera() is called while not connected.
class CameraRequiresConnectionError : public StreamError {
public:
    CameraRequiresConnectionError()
        : StreamError("Connect() before starting the camera.") {}
};

} // namespace streamkit
