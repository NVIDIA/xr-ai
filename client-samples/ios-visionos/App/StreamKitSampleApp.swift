// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import SwiftUI
import StreamKit

#if os(visionOS)
import CloudXRKit
#endif

@main
struct StreamKitSampleApp: App {

    @State private var model = AppModel()

    init() {
        #if os(visionOS)
        // Must run before any RealityKit content creates a CloudXRSessionComponent.
        CloudXRKit.registerSystems()
        #endif
        // Pre-configure AVAudioSession + mic permission so the first publish
        // doesn't race the system prompt and stall in republish retry loops.
        LiveKitBackend.prepareAudio()
        MediaSessionDiagnostics.shared.start()
    }

    var body: some Scene {

        WindowGroup {
            ContentView()
                .environment(model)
        }

        #if os(visionOS)
        ImmersiveSpace(id: AppModel.immersiveSpaceID) {
            ImmersiveView()
                .environment(model)
                .onAppear { model.immersiveSpaceIsOpen = true }
                .onDisappear {
                    model.immersiveSpaceIsOpen = false
                    // CloudXR renders into this RealityView, so stop a live session on disappear rather than leak it.
                    if model.xrState.isLive {
                        Task { await model.stopXR() }
                    }
                }
        }
        .immersionStyle(selection: .constant(.mixed), in: .mixed)
        #endif
    }
}
