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
                cameraPreviewSection
                agentSection
                connectionSection
                mediaSection
                dataSection
                if !model.receivedMessages.isEmpty {
                    messagesSection
                }
            }
            .navigationTitle("NVIDIA XR-AI Sample")
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
        .overlay(alignment: .bottom) {
            ErrorToast(message: model.lastError) {
                model.lastError = nil
            }
            .padding(.bottom, 24)
        }
    }

    // MARK: - Camera preview section

    @ViewBuilder
    private var cameraPreviewSection: some View {
        Section {
            CameraPreviewCard(isActive: model.isCameraActive)
                .frame(maxWidth: .infinity)
                .listRowInsets(EdgeInsets())
                .listRowBackground(Color.clear)
        }
    }

    // MARK: - Agent section

    @ViewBuilder
    private var agentSection: some View {
        Section("Agent") {
            if let response = model.agentResponse, !response.isEmpty {
                Text(response)
                    .font(.body)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
            } else {
                Text("Waiting for agent…")
                    .font(.subheadline)
                    .italic()
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
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

            if model.connectionState == .disconnected {
                TextField("Host / IP", text: $m.host)
                    .autocorrectionDisabled()
                    .textInputAutocapitalization(.never)
                    #if os(iOS)
                    .keyboardType(.decimalPad)
                    #endif

                TextField("Port", text: $m.port)
                    #if os(iOS)
                    .keyboardType(.numberPad)
                    #endif

                TextField("Token (paste JWT directly)", text: $m.token)
                    .autocorrectionDisabled()
                    .textInputAutocapitalization(.never)

                TextField("Token URL (e.g. https://host:8080/token)", text: $m.tokenServerURL)
                    .autocorrectionDisabled()
                    .textInputAutocapitalization(.never)
                    .onSubmit {}
                    #if os(iOS)
                    .keyboardType(.URL)
                    #endif

                TextField("Identity", text: $m.identity)
                    .autocorrectionDisabled()
                    .textInputAutocapitalization(.never)
                    .onSubmit {}

                Button("Install hub certificate") {
                    if let url = URL(string: "https://\(m.host):\(m.port)/cert") {
                        UIApplication.shared.open(url)
                    }
                }
                .disabled(m.host.isEmpty)

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
            #if os(visionOS)
            LabeledContent("Immersive Space") {
                immersiveSpaceToggle
            }
            #endif

            audioRow
            cameraRow
        }
    }

    @ViewBuilder
    private var audioRow: some View {
        @Bindable var m = model
        let isConnected = model.connectionState == .connected

        // Mic Mode picker is always visible (matches web). Disabled until
        // connected, or while the mic is live (mode can't change mid-stream).
        Picker("Mic Mode", selection: $m.audioMode) {
            Text("Voice Processing").tag(AudioConfig.MicrophoneMode.voiceProcessing)
            Text("Software (AEC on)").tag(AudioConfig.MicrophoneMode.softwareProcessing)
            Text("Raw (no DSP)").tag(AudioConfig.MicrophoneMode.raw)
        }
        .disabled(!isConnected || model.isAudioActive)

        LabeledContent("Microphone") {
            HStack {
                Text(micStatusLabel)
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
                    .disabled(!isConnected)
                }
            }
        }
    }

    private var micStatusLabel: String {
        if model.isAudioActive { return "Live" }
        return model.connectionState == .connected ? "Idle" : "Not connected"
    }

    @ViewBuilder
    private var cameraRow: some View {
        @Bindable var m = model
        let isConnected = model.connectionState == .connected

        Toggle("Camera on demand", isOn: $m.cameraOnDemand)

        Picker("Camera", selection: $m.cameraPosition) {
            ForEach(CameraConfig.Position.allCases, id: \.self) { position in
                Text(position.displayName).tag(position)
            }
        }
        #if os(visionOS)
        .disabled(true)
        #else
        .disabled(!isConnected || model.isCameraActive)
        #endif
        .onChange(of: model.cameraPosition) { _, newValue in
            Task { await model.switchCamera(to: newValue) }
        }

        LabeledContent("Camera") {
            HStack {
                Text(cameraStatusLabel)
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
                    .disabled(!isConnected || !model.immersiveSpaceIsOpen)
                    .help(model.immersiveSpaceIsOpen ? "" : "Open the immersive space first.")
                    #else
                    Button("Start") {
                        Task { await model.startCamera() }
                    }
                    .buttonStyle(.bordered)
                    .disabled(!isConnected)
                    #endif
                }
            }
        }
    }

    private var cameraStatusLabel: String {
        if model.isCameraActive { return "Streaming" }
        return model.connectionState == .connected ? "Idle" : "Not connected"
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
                    Text(msg.timestamp, format: .dateTime.hour(.twoDigits(amPM: .omitted)).minute(.twoDigits).second(.twoDigits))
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

// MARK: - CameraPreviewCard

/// Camera preview card mirroring the web client's `.preview-card`.
///
/// Aspect ratio follows the live LiveKit track dimensions once frames are
/// flowing (so portrait phone capture renders as 9:16, landscape cameras
/// as 16:9). Before the first frame arrives — including the entire
/// "Camera off" placeholder state — the card uses the matching phone-
/// camera orientation (9:16 in portrait, 16:9 in landscape) on iOS so the
/// placeholder has the same footprint the live preview will once frames
/// flow. visionOS has no meaningful "phone display" to mirror, so it
/// falls back to 16:9. Width is capped so the Agent panel below stays
/// visible without scrolling.
private struct CameraPreviewCard: View {
    let isActive: Bool

    @Environment(AppModel.self) private var model
    #if os(iOS)
    @Environment(\.verticalSizeClass) private var verticalSizeClass
    #endif
    @StateObject private var aspect = CameraPreviewAspect()

    private static let previewMaxWidth: CGFloat = 540

    /// Fallback aspect used until the first camera frame arrives.
    /// On iOS/iPadOS, matches the typical phone-camera frame orientation:
    /// 9:16 in portrait (compact width / regular height) and 16:9 in
    /// landscape (compact height). Recomputes on rotation because the
    /// size class environment value is observed.
    private var fallbackAspectRatio: CGFloat {
        #if os(iOS)
        verticalSizeClass == .compact ? 16.0 / 9.0 : 9.0 / 16.0
        #else
        16.0 / 9.0
        #endif
    }

    var body: some View {
        ZStack(alignment: .topLeading) {
            Color.black

            if isActive {
                LocalCameraView(model: model)
            } else {
                Text("Camera off")
                    .font(.subheadline)
                    .foregroundStyle(.white.opacity(0.6))
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }

            if isActive {
                LiveBadge()
                    .padding(10)
            }
        }
        .aspectRatio(aspect.value(default: fallbackAspectRatio), contentMode: .fit)
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        // Cap width at 80% of the containing list row, never exceeding
        // 540pt (the web client's `.page-content` cap).
        .containerRelativeFrame(.horizontal, alignment: .center) { width, _ in
            min(width * 0.8, Self.previewMaxWidth)
        }
        .onAppear { aspect.attach(to: model.session) }
        .onChange(of: model.isCameraActive) { _, _ in
            aspect.attach(to: model.session)
        }
        .onChange(of: model.cameraPosition) { _, _ in
            aspect.attach(to: model.session)
        }
    }
}

private struct LiveBadge: View {
    @State private var pulse: Double = 1.0

    var body: some View {
        HStack(spacing: 5) {
            Circle()
                .fill(Color.red)
                .frame(width: 6, height: 6)
                .opacity(pulse)
                .animation(
                    .easeInOut(duration: 0.6).repeatForever(autoreverses: true),
                    value: pulse
                )
                .onAppear { pulse = 0.35 }
            Text("LIVE")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.white)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 3)
        .background(Color.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 6))
    }
}

// MARK: - LocalCameraView

/// Renders the outgoing camera feed via StreamKit's `CameraPreviewView`,
/// which wraps LiveKit's `SwiftUIVideoView` so this file stays LiveKit-free.
///
/// On visionOS the ARKit passthrough track does not surface frames to a 2D
/// SwiftUI view, so the camera card stays on its "Camera off" placeholder
/// even while capture is active — the `LIVE` badge alone signals capture.
private struct LocalCameraView: View {
    let model: AppModel

    var body: some View {
        CameraPreviewView(session: model.session)
    }
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

// MARK: - ErrorToast

/// Bottom-anchored auto-dismiss toast (mirrors the web client's `#error-toast`).
/// Visible whenever `message` is non-nil; clears `message` after 4 seconds.
private struct ErrorToast: View {
    let message: String?
    let onDismiss: () -> Void

    @State private var dismissTask: Task<Void, Never>? = nil

    var body: some View {
        Group {
            if let message {
                Text(message)
                    .font(.subheadline)
                    .foregroundStyle(.white)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 10)
                    .background(Color.red.opacity(0.92),
                                in: Capsule(style: .continuous))
                    .shadow(radius: 8, y: 2)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
                    .onAppear { scheduleDismiss() }
                    .onTapGesture {
                        dismissTask?.cancel()
                        onDismiss()
                    }
                    .accessibilityAddTraits(.isStaticText)
            }
        }
        .animation(.easeOut(duration: 0.2), value: message)
        .onChange(of: message) { _, newValue in
            if newValue != nil { scheduleDismiss() }
        }
    }

    private func scheduleDismiss() {
        dismissTask?.cancel()
        dismissTask = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 4 * 1_000_000_000)
            if !Task.isCancelled {
                onDismiss()
            }
        }
    }
}

// MARK: - CameraConfig.Position + CaseIterable

extension CameraConfig.Position: CaseIterable {
    public static var allCases: [CameraConfig.Position] { [.back, .front] }

    var displayName: String {
        switch self {
        case .front: "Front"
        case .back:  "Back"
        }
    }
}

// MARK: - Preview

#Preview {
    ContentView()
        .environment(AppModel())
}
