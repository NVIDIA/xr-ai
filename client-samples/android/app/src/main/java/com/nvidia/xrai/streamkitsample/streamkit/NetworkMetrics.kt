// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.xrai.streamkitsample.streamkit

/** Transport-neutral quality estimate from the active streaming backend. */
enum class NetworkQuality { UNKNOWN, EXCELLENT, GOOD, POOR, LOST }

/**
 * A once-per-second snapshot of LiveKit's native WebRTC telemetry.
 * RTT and jitter are milliseconds and remain null until stats are available.
 */
data class NetworkMetrics(
    val quality: NetworkQuality = NetworkQuality.UNKNOWN,
    val roundTripTimeMs: Double? = null,
    val receiveJitterMs: Double? = null,
)
