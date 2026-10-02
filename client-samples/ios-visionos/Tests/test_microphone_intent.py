# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side app audio tests: python3 -m unittest discover -s client-samples/ios-visionos/Tests.

Compile AppModel's unmodified audio methods with a suspended fake session. This
tests intent/state interleavings without an iOS device or LiveKit binary; it does
not exercise hardware capture, notifications, or the full observable app model.
"""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SAMPLE = Path(__file__).resolve().parents[1]

HARNESS = r"""
import Foundation

struct AudioConfig {
    enum MicrophoneMode { case voiceProcessing }
    var mode: MicrophoneMode
}
enum ConnectionState { case connected }
enum CaptureFailure: Error { case unavailable }

@MainActor
final class FakeSession {
    var starts = 0
    var capturing = false
    var pending: CheckedContinuation<Void, Error>?

    func startAudio(config: AudioConfig) async throws {
        starts += 1
        try await withCheckedThrowingContinuation { pending = $0 }
        capturing = true
    }

    func stopAudio() async throws { capturing = false }

    func finishStart(failing: Bool = false) {
        let continuation = pending!
        pending = nil
        if failing { continuation.resume(throwing: CaptureFailure.unavailable) }
        else { continuation.resume() }
    }
}

@MainActor
final class AppModel {
    var session: FakeSession? = FakeSession()
    var connectionState = ConnectionState.connected
    var audioMode = AudioConfig.MicrophoneMode.voiceProcessing
    var lastError: String?
    var isTearingDown = false
    var micEnabledByUser = false
    var isAudioStarting = false
    var isAudioStopping = false
    var isAudioActive = false
    var micIntentGeneration: UInt64 = 0
    var micOperationGeneration: UInt64 = 0
    var micRecoveryTask: Task<Void, Never>?

    // APP_AUDIO_METHODS
}

@main
struct MicrophoneIntentTests {
    @MainActor
    static func main() async throws {
        let model = AppModel()
        let session = model.session!
        let first = Task { await model.enableMic() }
        while session.pending == nil { await Task.yield() }
        await model.enableMic()
        precondition(session.starts == 1, "Duplicate enable started a second capture")
        session.finishStart()
        await first.value
        precondition(session.capturing, "Fake session did not start capture")
        precondition(model.isAudioActive, "Successful overlapping enable left the app idle")
        precondition(!model.isAudioStarting, "Startup state was not cleared")
        await model.enableMic()
        precondition(session.starts == 1 && model.isAudioActive, "Active enable was not idempotent")

        await model.disableMic()
        precondition(!session.capturing && !model.isAudioActive, "Stop did not clear capture")
        let restart = Task { await model.enableMic() }
        while session.pending == nil { await Task.yield() }
        session.finishStart()
        await restart.value
        precondition(session.starts == 2 && model.isAudioActive, "Stop/start no longer works")
        print("PASS: overlapping enable, active enable, and stop/start")

        let retryModel = AppModel()
        let retrySession = retryModel.session!
        let failure = Task { await retryModel.enableMic() }
        while retrySession.pending == nil { await Task.yield() }
        await retryModel.enableMic()
        retrySession.finishStart(failing: true)
        await failure.value
        precondition(!retryModel.isAudioActive && !retryModel.isAudioStarting)
        precondition(retryModel.lastError != nil, "Overlapping enable swallowed startup failure")
        let retry = Task { await retryModel.enableMic() }
        while retrySession.pending == nil { await Task.yield() }
        retrySession.finishStart()
        await retry.value
        precondition(retrySession.starts == 2 && retryModel.isAudioActive, "Failed startup cannot retry")
        print("PASS: overlapping failure and explicit retry")

        let stoppingModel = AppModel()
        let stoppingSession = stoppingModel.session!
        let starting = Task { await stoppingModel.enableMic() }
        while stoppingSession.pending == nil { await Task.yield() }
        await stoppingModel.disableMic()
        stoppingSession.finishStart()
        await starting.value
        precondition(!stoppingModel.micEnabledByUser && !stoppingModel.isAudioActive,
                     "Late startup overrode the user's stop intent")
        print("PASS: stop intent rejects late startup state")
    }
}
"""


class MicrophoneIntentTests(unittest.TestCase):
    def test_app_audio_intent_interleavings(self):
        swiftc = shutil.which("swiftc")
        if swiftc is None:
            self.skipTest("Swift compiler required")
        source = (SAMPLE / "App" / "AppModel.swift").read_text()
        audio = source.split("    // MARK: - Audio\n", 1)[1].split(
            "    // MARK: - Camera\n", 1
        )[0]
        with tempfile.TemporaryDirectory(prefix="xr-ai-mic-intent-") as directory:
            work = Path(directory)
            harness = work / "MicrophoneIntentTests.swift"
            harness.write_text(HARNESS.replace("    // APP_AUDIO_METHODS", audio))
            executable = work / "microphone-intent-tests"
            compiled = subprocess.run(
                [swiftc, "-swift-version", "6", "-parse-as-library",
                 str(harness), "-o", str(executable)],
                capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            result = subprocess.run(
                [str(executable)], capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(result.stdout.count("PASS:"), 3, result.stdout)


if __name__ == "__main__":
    unittest.main()
