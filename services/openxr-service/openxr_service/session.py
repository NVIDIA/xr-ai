# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Own the native headless OpenXR and DeviceIO sessions."""

import threading
from typing import Any

from loguru import logger
from xr_ai_nat.functions.xr_tracking import HeadPose, OpenXRHealth

from .pose import from_native_pose, unavailable_pose


class HardwarePoseSource:
    """Acquire a fresh head pose while lazily retrying OpenXR startup."""

    _OPEN_LOG_AT_ATTEMPTS = (1, 2, 4, 8, 16, 32, 64, 128)

    def __init__(self) -> None:
        import isaacteleop.deviceio as deviceio

        self._deviceio = deviceio
        self._lock = threading.Lock()
        self._tracker = deviceio.HeadTracker()
        self._oxr_session: Any | None = None
        self._device_session: Any | None = None
        self._open_attempts = 0
        self._last_open_error: str | None = None
        self._next_log_index = 0

    def _try_open(self) -> str | None:
        if self._oxr_session is not None:
            return None
        import isaacteleop.oxr as oxr

        self._open_attempts += 1
        session = None
        try:
            extensions = self._deviceio.DeviceIOSession.get_required_extensions([self._tracker])
            session = oxr.OpenXRSession("openxr-service", extensions=list(extensions))
            session.__enter__()
            handles = session.get_handles()
            device_session = self._deviceio.DeviceIOSession.run([self._tracker], handles)
            device_session.__enter__()
            self._oxr_session = session
            self._device_session = device_session
            self._last_open_error = None
            logger.info(
                "openxr-service: sessions opened (attempt={}, instance={:#x}, session={:#x})",
                self._open_attempts,
                handles.instance,
                handles.session,
            )
            return None
        except Exception as exc:
            if session is not None:
                session.__exit__(None, None, None)
            error = f"{type(exc).__name__}: {exc}"
            self._last_open_error = error
            self._log_open_failure(error)
            return error

    def _log_open_failure(self, error: str) -> None:
        schedule = self._OPEN_LOG_AT_ATTEMPTS
        should_log = False
        if self._next_log_index < len(schedule):
            should_log = self._open_attempts >= schedule[self._next_log_index]
            if should_log:
                self._next_log_index += 1
        else:
            should_log = self._open_attempts % schedule[-1] == 0
        if should_log:
            logger.warning(
                "openxr-service: open attempt {} failed: {} (waiting for streaming client)",
                self._open_attempts,
                error,
            )

    def get_pose(self) -> HeadPose:
        with self._lock:
            if self._device_session is None:
                error = self._try_open()
                if error is not None:
                    return unavailable_pose(error)
            self._device_session.update()
            data = self._tracker.get_head(self._device_session).data
        return from_native_pose(data)

    def health(self) -> OpenXRHealth:
        return OpenXRHealth(
            session_open=self._oxr_session is not None,
            open_attempts=self._open_attempts,
            last_open_error=self._last_open_error,
        )

    def close(self) -> None:
        with self._lock:
            if self._device_session is not None:
                try:
                    self._device_session.__exit__(None, None, None)
                except Exception:
                    logger.exception("openxr-service: failed to close DeviceIO session")
                self._device_session = None
            if self._oxr_session is not None:
                try:
                    self._oxr_session.__exit__(None, None, None)
                except Exception:
                    logger.exception("openxr-service: failed to close OpenXR session")
                self._oxr_session = None
