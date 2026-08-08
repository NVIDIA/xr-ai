# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the sample-local live activity viewer."""

import argparse
import logging
import signal
import threading
from pathlib import Path

from xr_ai_logging import setup_logging

from .config import load_config
from .server import ActivityServer
from .store import EventStore, JsonlWatcher


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path)
    args = parser.parse_args()

    setup_logging("activity-viewer", namespace="tea-making-sample")
    config = load_config(args.config)
    store = EventStore()
    watcher = JsonlWatcher(config.sources, store)
    watcher.baseline()
    server = ActivityServer((config.host, config.port), store)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    stopped = threading.Event()

    def shutdown(_signum: int, _frame: object) -> None:
        stopped.set()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    if args.ready_file is not None:
        args.ready_file.touch()
    logging.getLogger(__name__).info(
        "Activity viewer ready at http://%s:%d",
        config.host,
        config.port,
    )
    try:
        while not stopped.wait(config.poll_interval_s):
            watcher.scan()
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join()


if __name__ == "__main__":
    run()
