// swift-tools-version: 6.2

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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
                .product(name: "LiveKit", package: "client-sdk-swift"),
            ],
            resources: [
                // SimulatorFeed.gif is used as the fake camera feed on the iOS simulator.
                // Replace it with any animated GIF to customise what gets streamed.
                .copy("Resources/SimulatorFeed.gif"),
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
