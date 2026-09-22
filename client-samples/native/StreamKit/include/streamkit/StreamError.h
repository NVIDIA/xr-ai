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

/// Thrown when a custom backend does not implement file transfer.
class FileTransferUnsupportedError : public StreamError {
public:
    FileTransferUnsupportedError()
        : StreamError("The selected streaming backend does not support file transfer.") {}
};

/// Thrown when file-transfer metadata violates the StreamKit wire contract.
class InvalidFileMetadataError : public StreamError {
public:
    explicit InvalidFileMetadataError(const std::string& reason)
        : StreamError("Invalid file metadata: " + reason) {}
};

/// Thrown when the active connection changes before a file stream closes.
class FileTransferIncompleteError : public StreamError {
public:
    FileTransferIncompleteError()
        : StreamError(
              "The file stream did not close successfully on the active connection.") {}
};

} // namespace streamkit
