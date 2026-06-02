// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * StreamKit — CameraPreviewView
 *
 * SwiftUI view that renders the local camera track of a StreamSession.
 *
 * Encapsulates the LiveKit `SwiftUIVideoView` so application code does not
 * need to import the LiveKit SDK.  Renders a transparent view when the
 * camera is not active or the session is backed by a non-LiveKit transport.
 */

import SwiftUI
import LiveKit

// MARK: - CameraPreviewView

/// Renders the local camera feed published by a ``StreamSession``.
///
/// Place this view anywhere in your layout to give the user a "what the
/// camera sees" preview that matches the web client's `<video>` element.
/// The view is empty (transparent) when ``StreamSession/localCameraTrack``
/// is `nil` (camera stopped, or non-LiveKit backend).
///
/// ```swift
/// CameraPreviewView(session: model.session)
///     .aspectRatio(16/9, contentMode: .fit)
/// ```
public struct CameraPreviewView: View {
    private let session: StreamSession?

    public init(session: StreamSession?) {
        self.session = session
    }

    public var body: some View {
        if let track = session?.localCameraTrack {
            // LiveKit's SwiftUIVideoView handles mirroring for front-facing
            // capture and resizes the underlying RTC video sink as the
            // SwiftUI layout changes.
            SwiftUIVideoView(track, layoutMode: .fill, mirrorMode: .auto)
        } else {
            Color.clear
        }
    }
}
