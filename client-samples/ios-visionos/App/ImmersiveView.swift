#if os(visionOS)
import RealityKit
import SwiftUI
import StreamKit

// MARK: - ImmersiveView

/// The visionOS immersive space scene.
///
/// This view owns the immersive space that ARKit's `CameraFrameProvider` requires.
/// The SDK's `startCamera()` runs the ARKit session internally once this space is open;
/// you can add your own `RealityView` content here alongside the camera stream.
struct ImmersiveView: View {

    @Environment(AppModel.self) private var model

    var body: some View {
        RealityView { content in
            // Add your own RealityKit entities here.
            // Example: a simple sphere anchored in front of the user.
            let mesh   = MeshResource.generateSphere(radius: 0.05)
            let material = SimpleMaterial(color: .systemBlue.withAlphaComponent(0.6), isMetallic: false)
            let entity = ModelEntity(mesh: mesh, materials: [material])
            entity.position = [0, 1.5, -0.5]
            content.add(entity)
        }
    }
}

#Preview(immersionStyle: .mixed) {
    ImmersiveView()
        .environment(AppModel())
}
#endif
