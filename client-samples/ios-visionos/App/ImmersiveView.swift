// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#if os(visionOS)
import RealityKit
import SwiftUI
import CloudXRKit

// MARK: - ImmersiveView

/// The visionOS immersive space scene.
///
/// Owns two things:
/// 1. The immersive space that ARKit's `CameraFrameProvider` requires for the
///    StreamKit camera path.
/// 2. The `RealityKit` entity that hosts the active `CloudXRSessionComponent`
///    when CloudXR is streaming. The component pairing must exist inside the
///    immersive space for CloudXRKit to render frames.
struct ImmersiveView: View {

    @Environment(AppModel.self) private var model

    @State private var sessionEntity = Entity()

    var body: some View {
        RealityView { content in
            sessionEntity.name = "Session"
            content.add(sessionEntity)

            let mesh   = MeshResource.generateSphere(radius: 0.05)
            let material = SimpleMaterial(color: .systemBlue.withAlphaComponent(0.6), isMetallic: false)
            let placeholder = ModelEntity(mesh: mesh, materials: [material])
            placeholder.position = [0, 1.5, -0.5]
            placeholder.name = "Placeholder"
            content.add(placeholder)
        } update: { content in
            if model.xrState == .streaming, let s = model.cloudxrSession {
                sessionEntity.components[CloudXRSessionComponent.self] = .init(session: s)
            } else {
                sessionEntity.components.remove(CloudXRSessionComponent.self)
                // CloudXRKit parents the streaming mesh under this entity; clear it so the stale geometry doesn't linger after Stop.
                sessionEntity.children.removeAll()
            }

            if let placeholder = content.entities.first(where: { $0.name == "Placeholder" }) {
                placeholder.isEnabled = (model.xrState != .streaming)
            }
        }
    }
}

#Preview(immersionStyle: .mixed) {
    ImmersiveView()
        .environment(AppModel())
}
#endif
