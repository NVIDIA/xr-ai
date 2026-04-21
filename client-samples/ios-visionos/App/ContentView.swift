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

                TextField("Token server URL (e.g. http://host/token)", text: $m.tokenServerURL)
                    .autocorrectionDisabled()
                    #if os(iOS)
                    .keyboardType(.URL)
                    #endif

                TextField("Identity", text: $m.identity)
                    .autocorrectionDisabled()

                Button("Connect") {
                    Task { await model.connect() }
                }
                .buttonStyle(.borderedProminent)
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

            // visionOS: immersive space must be open before camera can start
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
        if model.connectionState == .connected {
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

// MARK: - Preview

#Preview {
    ContentView()
        .environment(AppModel())
}
