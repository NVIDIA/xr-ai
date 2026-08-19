// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.kotlin.android) apply false
    alias(libs.plugins.kotlin.compose) apply false
}

// AGP's UTP test tooling pulls io.netty 4.1.93.Final (via grpc-netty), which has known
// CVEs; lift anything older to the pinned version. Buildscript and project configurations
// resolve separately, so both need the rule.
val nettyPin = libs.versions.netty.get()
val nettyPinParts = nettyPin.split('.').mapNotNull { it.toIntOrNull() }

fun belowNettyPin(version: String): Boolean {
    if (!version.startsWith("4.")) return false
    val parts = version.split('.').mapNotNull { it.toIntOrNull() }
    nettyPinParts.forEachIndexed { i, pin ->
        val requested = parts.getOrElse(i) { 0 }
        if (requested != pin) return requested < pin
    }
    return false
}

fun Configuration.forceNettyVersion() = resolutionStrategy.eachDependency {
    if (requested.group == "io.netty" && belowNettyPin(requested.version.orEmpty())) {
        useVersion(nettyPin)
    }
}

allprojects {
    buildscript.configurations.configureEach { forceNettyVersion() }
    configurations.configureEach { forceNettyVersion() }
}
