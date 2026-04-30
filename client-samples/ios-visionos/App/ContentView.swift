// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import SwiftUI
import StreamKit

// MARK: - ContentView

struct ContentView: View {

    @Environment(AppModel.self) private var model

    #if os(visionOS)
    @Environment(\.openImmersiveSpace)    var openImmersiveSpace
    @Environment(\.dismissImmersiveSpace) var dismissImmersiveSpace
    #endif

    @State private var sendText = ""

    var body: some View {
        NavigationStack {
            Form {
                connectionSection
                mediaSection
                dataSection
                if !model.receivedMessages.isEmpty {
                    messagesSection
                }
            }
            .navigationTitle("StreamKit Sample")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.large)
            .scrollDismissesKeyboard(.interactively)
            .toolbar {
                ToolbarItemGroup(placement: .keyboard) {
                    Spacer()
                    Button("Done") {
                        UIApplication.shared.sendAction(
                            #selector(UIResponder.resignFirstResponder),
                            to: nil, from: nil, for: nil
                        )
                    }
                }
            }
            #endif
        }
        .alert("Error", isPresented: Binding(
            get: { model.lastError != nil },
            set: { if !$0 { model.lastError = nil } }
        )) {
            Button("OK", role: .cancel) { model.lastError = nil }
        } message: {
            Text(model.lastError ?? "")
        }
    }

    // MARK: - Connection section

    @ViewBuilder
    private var connectionSection: some View {
        @Bindable var m = model

        Section("Connection") {
            LabeledContent("State") {
                ConnectionStateBadge(state: model.connectionState)
            }

            if model.connectionState == .connected {
                LabeledContent("Agent") {
                    AgentStatusBadge(status: model.agentStatus)
                }
            }

            if model.connectionState == .disconnected {
                TextField("Host / IP", text: $m.host)
                    .autocorrectionDisabled()
                    #if os(iOS)
                    .keyboardType(.decimalPad)
                    #endif

                TextField("Port", text: $m.port)
                    #if os(iOS)
                    .keyboardType(.numberPad)
                    #endif

                Toggle("Token server uses HTTPS", isOn: $m.secure)

                TextField("Token server URL (e.g. http://host/token)", text: $m.tokenServerURL)
                    .autocorrectionDisabled()
                    .onSubmit {}
                    #if os(iOS)
                    .keyboardType(.URL)
                    #endif

                TextField("Identity", text: $m.identity)
                    .autocorrectionDisabled()
                    .onSubmit {}

                Button(model.isConnecting ? "Connecting…" : "Connect") {
                    Task { await model.connect() }
                }
                .buttonStyle(.borderedProminent)
                .disabled(model.isConnecting)
            } else {
                Button("Disconnect", role: .destructive) {
                    Task { await model.disconnect() }
                }
            }
        }
    }

    // MARK: - Media section

    @ViewBuilder
    private var mediaSection: some View {
        Section("Media") {

            // visionOS: immersive space must be open before camera can start.
            #if os(visionOS)
            LabeledContent("Immersive Space") {
                immersiveSpaceToggle
            }
            #endif

            // Audio
            audioRow

            // Camera
            cameraRow
        }
    }

    @ViewBuilder
    private var audioRow: some View {
        if model.connectionState == .connected {
            @Bindable var m = model

            // Audio mode picker — mirrors the web client's dropdown.
            // Disabled while the mic is actively streaming (same as web).
            Picker("Audio Mode", selection: $m.audioMode) {
                Text("Voice Processing").tag(AudioConfig.MicrophoneMode.voiceProcessing)
                Text("Software (AEC on)").tag(AudioConfig.MicrophoneMode.softwareProcessing)
                Text("Raw (no DSP)").tag(AudioConfig.MicrophoneMode.raw)
            }
            .disabled(model.isAudioActive)

            LabeledContent("Microphone") {
                HStack {
                    Text(model.isAudioActive ? "Streaming" : "Idle")
                        .foregroundStyle(model.isAudioActive ? .green : .secondary)
                    if model.isAudioActive {
                        Button("Stop", role: .destructive) {
                            Task { await model.stopAudio() }
                        }
                        .buttonStyle(.bordered)
                    } else {
                        Button("Start") {
                            Task { await model.startAudio() }
                        }
                        .buttonStyle(.bordered)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private var cameraRow: some View {
        @Bindable var m = model

        // Camera on demand toggle — always visible so the user can set the
        // mode before connecting.  Ignored by the agent in always-on mode.
        Toggle("Camera on demand", isOn: $m.cameraOnDemand)

        if model.connectionState == .connected {
            Picker("Camera Mode", selection: $m.cameraPosition) {
                Text("Front").tag(CameraConfig.Position.front)
                Text("Back").tag(CameraConfig.Position.back)
            }
            #if os(visionOS)
            .disabled(true)
            #endif
            .onChange(of: model.cameraPosition) { _, newValue in
                Task { await model.switchCamera(to: newValue) }
            }

            LabeledContent("Camera") {
                HStack {
                    Text(model.isCameraActive ? "Streaming" : "Idle")
                        .foregroundStyle(model.isCameraActive ? .green : .secondary)
                    if model.isCameraActive {
                        Button("Stop", role: .destructive) {
                            Task { await model.stopCamera() }
                        }
                        .buttonStyle(.bordered)
                    } else {
                        #if os(visionOS)
                        Button("Start") {
                            Task { await model.startCamera() }
                        }
                        .buttonStyle(.bordered)
                        .disabled(!model.immersiveSpaceIsOpen)
                        .help(model.immersiveSpaceIsOpen ? "" : "Open the immersive space first.")
                        #else
                        Button("Start") {
                            Task { await model.startCamera() }
                        }
                        .buttonStyle(.bordered)
                        #endif
                    }
                }
            }
        }
    }

    // MARK: - Data section

    @ViewBuilder
    private var dataSection: some View {
        Section("Data Channel") {
            Button("Send Ping") {
                Task { await model.sendPing() }
            }
            .disabled(model.connectionState != .connected)

            HStack {
                TextField("Custom message…", text: $sendText)
                    .autocorrectionDisabled()
                    .onSubmit {}
                Button("Send") {
                    Task {
                        await model.sendCustom(text: sendText)
                        sendText = ""
                    }
                }
                .disabled(model.connectionState != .connected || sendText.isEmpty)
            }
        }
    }

    // MARK: - Received messages

    @ViewBuilder
    private var messagesSection: some View {
        Section("Received (\(model.receivedMessages.count))") {
            ForEach(model.receivedMessages) { msg in
                VStack(alignment: .leading, spacing: 2) {
                    Text(msg.text)
                        .font(.system(.footnote, design: .monospaced))
                    Text(msg.timestamp, style: .time)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    // MARK: - visionOS immersive space toggle

    #if os(visionOS)
    @ViewBuilder
    private var immersiveSpaceToggle: some View {
        Button(model.immersiveSpaceIsOpen ? "Close Space" : "Open Space") {
            Task {
                if model.immersiveSpaceIsOpen {
                    await dismissImmersiveSpace()
                } else {
                    await openImmersiveSpace(id: AppModel.immersiveSpaceID)
                }
            }
        }
        .tint(model.immersiveSpaceIsOpen ? .red : .blue)
    }
    #endif
}

// MARK: - ConnectionStateBadge

private struct ConnectionStateBadge: View {
    let state: ConnectionState

    var body: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(color)
                .frame(width: 8, height: 8)
            Text(label)
                .font(.footnote)
                .foregroundStyle(color)
        }
        .animation(.easeInOut, value: state)
    }

    private var color: Color {
        switch state {
        case .disconnected:  return .secondary
        case .connecting:    return .orange
        case .connected:     return .green
        case .reconnecting:  return .yellow
        }
    }

    private var label: String {
        switch state {
        case .disconnected:  return "Disconnected"
        case .connecting:    return "Connecting…"
        case .connected:     return "Connected"
        case .reconnecting:  return "Reconnecting…"
        }
    }
}

// MARK: - AgentStatusBadge

private struct AgentStatusBadge: View {
    let status: String?

    var body: some View {
        HStack(spacing: 6) {
            if status == "processing" {
                Circle()
                    .fill(Color.orange)
                    .frame(width: 8, height: 8)
                    .opacity(pulseOpacity)
                    .animation(
                        .easeInOut(duration: 0.7).repeatForever(autoreverses: true),
                        value: pulseOpacity
                    )
                    .onAppear { pulseOpacity = 0.3 }
            } else {
                Circle()
                    .fill(dotColor)
                    .frame(width: 8, height: 8)
            }
            Text(label)
                .font(.footnote)
                .foregroundStyle(dotColor)
        }
        .animation(.easeInOut, value: status)
    }

    @State private var pulseOpacity: Double = 1.0

    private var dotColor: Color {
        switch status {
        case "idle":       return .green
        case "processing": return .orange
        default:           return .secondary
        }
    }

    private var label: String {
        switch status {
        case "idle":       return "Idle"
        case "processing": return "Processing…"
        case nil:          return "—"
        default:           return "Unknown"
        }
    }
}

// MARK: - Preview

#Preview {
    ContentView()
        .environment(AppModel())
}
