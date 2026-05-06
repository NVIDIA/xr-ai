// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * StreamKit native sample app.
 *
 * Demonstrates the full StreamSession lifecycle:
 *   connect → start audio → start camera → send data → receive data → disconnect
 *
 * Usage:
 *   streamkit_sample --host <ip> --token <jwt> [--port 7880] [--identity <name>]
 *
 * NOTE: The LiveKitBackend is currently a stub, so Connect() reports
 * kConnected immediately without opening a real WebRTC session. Replace
 * the stub with a real implementation to use this against a live server.
 */

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "streamkit/StreamSession.h"
#include "streamkit/Config/BackendConfiguration.h"
#include "streamkit/ConnectionState.h"
#include "streamkit/StreamError.h"

// ─────────────────────────────────────────────────────────────────────────────
// Argument parsing
// ─────────────────────────────────────────────────────────────────────────────

struct Args {
    std::string host;
    std::string token;
    int         port     = 7880;
    std::string identity = "native-client";
};

static std::string GetArg(const std::vector<std::string>& argv,
                           const std::string& flag,
                           const std::string& fallback = "") {
    auto it = std::find(argv.begin(), argv.end(), flag);
    if (it != argv.end() && std::next(it) != argv.end()) {
        return *std::next(it);
    }
    return fallback;
}

static Args ParseArgs(int argc, char** argv) {
    std::vector<std::string> args(argv, argv + argc);
    Args result;
    result.host     = GetArg(args, "--host");
    result.token    = GetArg(args, "--token");
    result.port     = std::stoi(GetArg(args, "--port", "7880"));
    result.identity = GetArg(args, "--identity", "native-client");

    if (result.host.empty() || result.token.empty()) {
        std::cerr << "Usage: streamkit_sample --host <ip> --token <jwt>"
                  << " [--port 7880] [--identity <name>]\n";
        std::exit(1);
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static std::string StateToString(streamkit::ConnectionState state) {
    switch (state) {
        case streamkit::ConnectionState::kDisconnected:  return "disconnected";
        case streamkit::ConnectionState::kConnecting:    return "connecting";
        case streamkit::ConnectionState::kConnected:     return "connected";
        case streamkit::ConnectionState::kReconnecting:  return "reconnecting";
    }
    return "unknown";
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    const Args args = ParseArgs(argc, argv);

    // ── Build BackendConfiguration ────────────────────────────────────────
    streamkit::LiveKitConfig lk_config;
    lk_config.host  = args.host;
    lk_config.port  = args.port;
    lk_config.token = args.token;

    // ── Create StreamSession ──────────────────────────────────────────────
    streamkit::StreamSession session{streamkit::BackendConfiguration{lk_config}};

    // ── Wire event hooks ──────────────────────────────────────────────────
    session.on_connection_state_changed = [](streamkit::ConnectionState state) {
        std::cout << "[state] " << StateToString(state) << "\n";
    };

    session.on_data_received = [](std::string_view topic,
                                  std::span<const std::byte> data) {
        std::string payload(reinterpret_cast<const char*>(data.data()), data.size());
        std::cout << "[data] topic='" << topic << "' payload='" << payload << "'\n";
    };

    session.on_agent_status = [](std::string_view status) {
        std::cout << "[agent] status='" << status << "'\n";
    };

    // ── Connect ───────────────────────────────────────────────────────────
    try {
        std::cout << "Connecting to " << args.host << ":" << args.port
                  << " as '" << args.identity << "'...\n";

        session.Connect(streamkit::SessionConfig{args.identity});

        std::cout << "Connected.\n";
    } catch (const streamkit::StreamError& e) {
        std::cerr << "[error] Connect failed: " << e.what() << "\n";
        return 1;
    }

    // ── Start audio ───────────────────────────────────────────────────────
    try {
        session.StartAudio(streamkit::AudioConfig::Default());
        std::cout << "Audio started.\n";
    } catch (const streamkit::StreamError& e) {
        // Audio failure never drops the connection — log and continue.
        std::cerr << "[warn] StartAudio failed: " << e.what() << "\n";
    }

    // ── Start camera ──────────────────────────────────────────────────────
    try {
        session.StartCamera(streamkit::CameraConfig::Default());
        std::cout << "Camera started.\n";
    } catch (const streamkit::StreamError& e) {
        std::cerr << "[warn] StartCamera failed: " << e.what() << "\n";
    }

    // ── Send a test message ───────────────────────────────────────────────
    try {
        const std::string msg = R"({"event":"hello","from":"native-client"})";
        std::vector<std::byte> payload(msg.size());
        std::transform(msg.begin(), msg.end(), payload.begin(),
                       [](char c) { return static_cast<std::byte>(c); });

        session.Send(payload, /*reliable=*/true, "xr.session.started");
        std::cout << "Sent test message.\n";
    } catch (const streamkit::StreamError& e) {
        std::cerr << "[warn] Send failed: " << e.what() << "\n";
    }

    // ── Run for a few seconds then disconnect ─────────────────────────────
    std::cout << "Running for 5 seconds...\n";
    std::this_thread::sleep_for(std::chrono::seconds(5));

    session.StopAudio();
    session.StopCamera();
    session.Disconnect();

    std::cout << "Disconnected. Bye.\n";
    return 0;
}
