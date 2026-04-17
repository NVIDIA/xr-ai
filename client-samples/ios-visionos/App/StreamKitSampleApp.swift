import SwiftUI
import StreamKit

@main
struct StreamKitSampleApp: App {

    @State private var model = AppModel()

    var body: some Scene {

        WindowGroup {
            ContentView()
                .environment(model)
        }

        #if os(visionOS)
        ImmersiveSpace(id: AppModel.immersiveSpaceID) {
            ImmersiveView()
                .environment(model)
                // Notify the model when the space appears/disappears.
                .onAppear  { model.immersiveSpaceIsOpen = true  }
                .onDisappear { model.immersiveSpaceIsOpen = false }
        }
        .immersionStyle(selection: .constant(.mixed), in: .mixed)
        #endif
    }
}
