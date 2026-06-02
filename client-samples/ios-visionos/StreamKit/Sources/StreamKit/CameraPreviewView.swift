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

import Combine
import LiveKit
import SwiftUI

// MARK: - CameraPreviewView

/// Renders the local camera feed published by a ``StreamSession``.
///
/// Place this view anywhere in your layout to give the user a "what the
/// camera sees" preview that matches the web client's `<video>` element.
/// The view is empty (transparent) when ``StreamSession/localCameraTrack``
/// is `nil` (camera stopped, or non-LiveKit backend).
///
/// Pair with ``CameraPreviewAspect`` to size the surrounding container to
/// the actual capture aspect ratio (so portrait phone capture renders as
/// 9:16, landscape sensors as 16:9):
///
/// ```swift
/// @StateObject private var aspect = CameraPreviewAspect()
///
/// CameraPreviewView(session: model.session)
///     .aspectRatio(aspect.value(default: 16.0 / 9.0), contentMode: .fit)
///     .onAppear { aspect.attach(to: model.session) }
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

// MARK: - CameraPreviewAspect

/// Observes the live camera track's frame dimensions and exposes them as
/// an aspect ratio (`width / height`) for SwiftUI layout.
///
/// Returns `nil` until the first frame has been published by LiveKit; pair
/// it with a sensible fallback so the layout is stable before capture starts:
///
/// ```swift
/// @StateObject private var aspect = CameraPreviewAspect()
///
/// var body: some View {
///     CameraPreviewView(session: model.session)
///         .aspectRatio(aspect.value(default: 16.0 / 9.0), contentMode: .fit)
///         .onChange(of: model.session?.localCameraTrack === nil) { _, _ in
///             aspect.attach(to: model.session)
///         }
/// }
/// ```
///
/// Avoids leaking LiveKit types into app code — the only public surface is
/// a `CGFloat` aspect ratio.
@MainActor
public final class CameraPreviewAspect: ObservableObject {
    @Published public private(set) var aspectRatio: CGFloat?

    private var observer: TrackDelegateObserver?
    private var cancellable: AnyCancellable?

    public init() {}

    /// Returns the live aspect ratio if known, otherwise `defaultValue`.
    public func value(default defaultValue: CGFloat) -> CGFloat {
        aspectRatio ?? defaultValue
    }

    /// Attaches to the session's current camera track. Call this whenever
    /// the camera is started, stopped, or switched (front ↔ back).
    public func attach(to session: StreamSession?) {
        cancellable = nil
        observer = nil
        aspectRatio = nil

        guard let track = session?.localCameraTrack else { return }

        let observer = TrackDelegateObserver(track: track)
        self.observer = observer

        update(from: observer.dimensions)
        cancellable = observer.$dimensions.sink { [weak self] dimensions in
            Task { @MainActor [weak self] in
                self?.update(from: dimensions)
            }
        }
    }

    private func update(from dimensions: Dimensions?) {
        guard let dimensions, dimensions.width > 0, dimensions.height > 0 else {
            aspectRatio = nil
            return
        }
        aspectRatio = CGFloat(dimensions.width) / CGFloat(dimensions.height)
    }
}
