# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the production reconciler/queue tests without Apple or LiveKit frameworks."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SAMPLE = Path(__file__).resolve().parents[1]
PACKAGE = '''// swift-tools-version: 6.0
import PackageDescription
let package = Package(
    name: "MicrophoneLifecycleTests",
    platforms: [.macOS(.v13)],
    targets: [
        .target(name: "MicrophoneLifecycle"),
        .testTarget(name: "MicrophoneLifecycleTests", dependencies: ["MicrophoneLifecycle"]),
    ]
)
'''


class MicrophoneLifecycleTests(unittest.TestCase):
    def test_production_reconciler_and_backend_queue(self):
        swift = shutil.which("swift")
        self.assertIsNotNone(swift, "Swift 6 is required for microphone lifecycle tests")
        with tempfile.TemporaryDirectory(prefix="xr-ai-mic-lifecycle-") as directory:
            work = Path(directory)
            (work / "Package.swift").write_text(PACKAGE)
            sources = work / "Sources" / "MicrophoneLifecycle"
            tests = work / "Tests" / "MicrophoneLifecycleTests"
            sources.mkdir(parents=True)
            tests.mkdir(parents=True)
            shutil.copy2(SAMPLE / "App" / "MicrophoneReconciler.swift", sources)
            shutil.copy2(
                SAMPLE / "StreamKit/Sources/StreamKit/Backends/LiveKit/MicrophoneOperations.swift", sources
            )
            shutil.copy2(SAMPLE / "Tests" / "MicrophoneLifecycleTests.swift", tests)
            # Only the module name differs in the dependency-free host test target.
            queue_tests = SAMPLE / "StreamKit/Tests/StreamKitTests/MicrophoneOperationsTests.swift"
            (tests / queue_tests.name).write_text(
                queue_tests.read_text().replace("@testable import StreamKit", "@testable import MicrophoneLifecycle")
            )
            result = subprocess.run(
                [swift, "test", "--package-path", str(work)],
                capture_output=True, text=True, timeout=180,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            print(result.stdout, end="")


if __name__ == "__main__":
    unittest.main()
