// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import AVFoundation
import Foundation
import os

/// Library-owned subsystem so the category is stable across any app that adopts
/// StreamKit, rather than tracking the host app's bundle id.
private let mediaLogSubsystem = "com.nvidia.streamkit"

/// Shared log for media-session lifecycle events. Filter Console.app / Xcode by
/// category `MediaSession`. Always on (not DEBUG-gated) so intermittent,
/// untethered failures are still captured in the unified log.
let mediaLog = Logger(subsystem: mediaLogSubsystem, category: "MediaSession")

// MARK: - MediaSessionDiagnostics

/// Notification-based observer (no polling) that logs microphone/camera session
/// interruptions, route changes, and resets. Start once at app launch.
///
/// `@unchecked Sendable`: the only mutable state (`tokens`, `started`) is touched
/// from `start()`/`stop()`, called on the main thread at launch and never
/// concurrently; the notification handlers read immutable payloads and log.
public final class MediaSessionDiagnostics: @unchecked Sendable {

    public static let shared = MediaSessionDiagnostics()

    private var tokens: [NSObjectProtocol] = []
    private var started = false

    private init() {}

    /// Begin logging. Idempotent.
    public func start() {
        guard !started else { return }
        started = true
        let nc = NotificationCenter.default

        #if os(iOS) || os(visionOS)
        tokens.append(nc.addObserver(forName: AVAudioSession.interruptionNotification,
                                     object: nil, queue: .main) { [weak self] in
            self?.logAudioInterruption($0)
        })
        tokens.append(nc.addObserver(forName: AVAudioSession.routeChangeNotification,
                                     object: nil, queue: .main) { [weak self] in
            self?.logAudioRouteChange($0)
        })
        tokens.append(nc.addObserver(forName: AVAudioSession.mediaServicesWereResetNotification,
                                     object: nil, queue: .main) { [weak self] _ in
            self?.logAudioServicesReset()
        })
        #endif

        // AVCaptureSession is the iOS/iPadOS camera path. visionOS uses ARKit
        // (ARCameraFrameProvider) instead, so these never fire there.
        #if os(iOS)
        tokens.append(nc.addObserver(forName: AVCaptureSession.wasInterruptedNotification,
                                     object: nil, queue: .main) { [weak self] in
            self?.logCaptureInterrupted($0)
        })
        tokens.append(nc.addObserver(forName: AVCaptureSession.interruptionEndedNotification,
                                     object: nil, queue: .main) { _ in
            mediaLog.error("capture session interruption ENDED (camera can resume)")
        })
        tokens.append(nc.addObserver(forName: AVCaptureSession.runtimeErrorNotification,
                                     object: nil, queue: .main) { [weak self] in
            self?.logCaptureRuntimeError($0)
        })
        #endif

        mediaLog.info("MediaSessionDiagnostics started")
    }

    public func stop() {
        guard started else { return }
        tokens.forEach { NotificationCenter.default.removeObserver($0) }
        tokens.removeAll()
        started = false
    }

    deinit { stop() }

    // MARK: - Audio handlers

    #if os(iOS) || os(visionOS)
    private func logAudioInterruption(_ note: Notification) {
        let info = note.userInfo ?? [:]
        guard let raw = info[AVAudioSessionInterruptionTypeKey] as? UInt,
              let type = AVAudioSession.InterruptionType(rawValue: raw) else {
            mediaLog.error("audio interruption: malformed userInfo")
            return
        }
        switch type {
        case .began:
            mediaLog.error("audio interruption BEGAN: system suspended mic capture")
        case .ended:
            var resume = false
            if let optRaw = info[AVAudioSessionInterruptionOptionKey] as? UInt {
                resume = AVAudioSession.InterruptionOptions(rawValue: optRaw).contains(.shouldResume)
            }
            mediaLog.error("audio interruption ENDED, shouldResume=\(resume, privacy: .public)")
        @unknown default:
            mediaLog.error("audio interruption: unknown type \(raw, privacy: .public)")
        }
    }

    private func logAudioRouteChange(_ note: Notification) {
        let info = note.userInfo ?? [:]
        var reasonStr = "unknown"
        if let raw = info[AVAudioSessionRouteChangeReasonKey] as? UInt,
           let reason = AVAudioSession.RouteChangeReason(rawValue: raw) {
            reasonStr = Self.routeReason(reason)
        }
        let route = AVAudioSession.sharedInstance().currentRoute
        let inputs = route.inputs.map(\.portType.rawValue).joined(separator: ",")
        let outputs = route.outputs.map(\.portType.rawValue).joined(separator: ",")
        mediaLog.info("audio route change, reason=\(reasonStr, privacy: .public) inputs=[\(inputs, privacy: .public)] outputs=[\(outputs, privacy: .public)]")
    }

    private func logAudioServicesReset() {
        mediaLog.error("audio mediaServicesWereReset: engine + session must be rebuilt from scratch")
    }

    private static func routeReason(_ reason: AVAudioSession.RouteChangeReason) -> String {
        switch reason {
        case .unknown:                  return "unknown"
        case .newDeviceAvailable:       return "newDeviceAvailable"
        case .oldDeviceUnavailable:     return "oldDeviceUnavailable"
        case .categoryChange:           return "categoryChange"
        case .override:                 return "override"
        case .wakeFromSleep:            return "wakeFromSleep"
        case .noSuitableRouteForCategory: return "noSuitableRouteForCategory"
        case .routeConfigurationChange: return "routeConfigurationChange"
        @unknown default:               return "unhandled(\(reason.rawValue))"
        }
    }
    #endif

    // MARK: - Capture handlers

    #if os(iOS)
    private func logCaptureInterrupted(_ note: Notification) {
        var reasonStr = "unknown"
        if let raw = note.userInfo?[AVCaptureSessionInterruptionReasonKey] as? Int,
           let reason = AVCaptureSession.InterruptionReason(rawValue: raw) {
            reasonStr = Self.captureReason(reason)
        }
        mediaLog.error("capture session INTERRUPTED, reason=\(reasonStr, privacy: .public)")
    }

    private func logCaptureRuntimeError(_ note: Notification) {
        let err = note.userInfo?[AVCaptureSessionErrorKey] as? NSError
        mediaLog.error("capture session RUNTIME ERROR, domain=\(err?.domain ?? "?", privacy: .public) code=\(err?.code ?? 0, privacy: .public)")
    }

    private static func captureReason(_ reason: AVCaptureSession.InterruptionReason) -> String {
        switch reason {
        case .videoDeviceNotAvailableInBackground:           return "videoDeviceNotAvailableInBackground"
        case .audioDeviceInUseByAnotherClient:               return "audioDeviceInUseByAnotherClient"
        case .videoDeviceInUseByAnotherClient:               return "videoDeviceInUseByAnotherClient"
        case .videoDeviceNotAvailableWithMultipleForegroundApps: return "videoDeviceNotAvailableWithMultipleForegroundApps"
        case .videoDeviceNotAvailableDueToSystemPressure:    return "videoDeviceNotAvailableDueToSystemPressure"
        @unknown default:                                    return "unhandled(\(reason.rawValue))"
        }
    }
    #endif
}
