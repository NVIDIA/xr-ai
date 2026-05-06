// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace streamkit {

/// Connection lifecycle state reported by StreamSession.
///
/// Mirror of Swift `ConnectionState`, Kotlin `ConnectionState`,
/// and the JS `ConnectionState` constants.
enum class ConnectionState {
    kDisconnected,
    kConnecting,
    kConnected,
    kReconnecting,
};

} // namespace streamkit
