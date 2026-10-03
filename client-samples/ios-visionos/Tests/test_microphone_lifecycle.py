# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run microphone lifecycle and app-wiring tests without Apple or LiveKit frameworks."""

import re
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
            shutil.copy2(SAMPLE / "StreamKit/Sources/StreamKit/Config/AudioConfig.swift", tests)
            app = (SAMPLE / "App/AppModel.swift").read_text()
            methods = []
            for name in ("makeMicrophone", "enableMic", "disableMic", "recoverMic", "handleMicrophoneConnectionState"):
                matches = re.findall(rf"^    (?:private )?func {name}\(.*?^    }}\n", app, re.MULTILINE | re.DOTALL)
                self.assertEqual(len(matches), 1, f"Expected one production {name} method")
                methods.append(matches[0].replace("#if os(visionOS)", "#if os(visionOS) || TEST_VISIONOS"))
            template = (SAMPLE / "Tests/AppModelMicrophoneTests.swift.in").read_text()
            xr_state = re.findall(r"^enum XRState:.*?^}\n", app, re.MULTILINE | re.DOTALL)
            self.assertEqual(len(xr_state), 1)
            (tests / "AppModelMicrophoneTests.swift").write_text(
                template.replace("    // APP_MODEL_METHODS", "\n".join(methods)).replace("// XR_STATE", xr_state[0])
            )
            backend = (SAMPLE / "StreamKit/Sources/StreamKit/Backends/LiveKit/LiveKitBackend.swift").read_text()
            stop = re.findall(r"^    private func stopMicrophone\(.*?^    }\n", backend, re.MULTILINE | re.DOTALL)
            self.assertEqual(len(stop), 1)
            cleanup_template = (SAMPLE / "Tests/MicrophoneCleanupTests.swift.in").read_text()
            (tests / "MicrophoneCleanupTests.swift").write_text(
                cleanup_template.replace("    // BACKEND_STOP_METHOD", stop[0])
            )
            # Only the module name differs in the dependency-free host test target.
            queue_tests = SAMPLE / "StreamKit/Tests/StreamKitTests/MicrophoneOperationsTests.swift"
            (tests / queue_tests.name).write_text(
                queue_tests.read_text().replace("@testable import StreamKit", "@testable import MicrophoneLifecycle")
            )
            # Exercise the XR guards on the host without importing CloudXR/ARKit.
            for flags in ([], ["-Xswiftc", "-DTEST_VISIONOS"]):
                result = subprocess.run(
                    [swift, "test", "--package-path", str(work), *flags],
                    capture_output=True, text=True, timeout=180,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                print(result.stdout, end="")


if __name__ == "__main__":
    unittest.main()
