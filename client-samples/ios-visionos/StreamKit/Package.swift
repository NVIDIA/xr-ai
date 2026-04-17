// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "StreamKit",
    platforms: [
        .iOS(.v18),
        .visionOS(.v26),
    ],
    products: [
        .library(name: "StreamKit", targets: ["StreamKit"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/livekit/client-sdk-swift",
            from: "2.13.0"
        ),
    ],
    targets: [
        .target(
            name: "StreamKit",
            dependencies: [
                .product(name: "LiveKit", package: "livekit-client-sdk-swift"),
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "StreamKitTests",
            dependencies: ["StreamKit"],
            path: "Tests/StreamKitTests"
        ),
    ]
)
