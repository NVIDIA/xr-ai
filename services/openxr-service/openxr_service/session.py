# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Own the headless OpenXR session that tracks the user's head."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any

import xr
from loguru import logger

from .pose import from_native_pose, unavailable_pose

# Headless: no graphics binding. Overlay: run beside the application that
# renders to the headset. Timespec: convert the monotonic clock to XrTime.
_EXTENSIONS = (
    xr.MND_HEADLESS_EXTENSION_NAME,
    xr.EXTX_OVERLAY_EXTENSION_NAME,
    xr.KHR_CONVERT_TIMESPEC_TIME_EXTENSION_NAME,
)
_POSE_VALID = xr.SpaceLocationFlags.POSITION_VALID_BIT | xr.SpaceLocationFlags.ORIENTATION_VALID_BIT


class HeadlessSession:
    """A running headless OpenXR overlay session with a head tracker.

    Ported from IsaacTeleop 1.3.131 (``oxr_session.cpp`` and
    ``live_head_tracker_impl.cpp``, Apache-2.0): poses are reported in the
    STAGE reference space, and the head is the VIEW space located in it.
    Construction raises :class:`xr.XrException` while no headset is streaming.
    """

    def __init__(self, app_name: str) -> None:
        # (handle, destroy function), in creation order.
        self._handles: list[tuple[Any, Any]] = []
        try:
            self._open(app_name)
        except BaseException:
            self.close()
            raise

    def _open(self, app_name: str) -> None:
        self.instance = self._own(
            xr.destroy_instance,
            xr.create_instance(
                xr.InstanceCreateInfo(
                    application_info=xr.ApplicationInfo(
                        application_name=app_name,
                        engine_name="xr-ai",
                        api_version=xr.Version(1, 0, 0),
                    ),
                    enabled_extension_names=list(_EXTENSIONS),
                )
            ),
        )
        self.system = xr.get_system(
            self.instance, xr.SystemGetInfo(form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY)
        )
        # The overlay struct must outlive create_session, which reads it.
        overlay = xr.SessionCreateInfoOverlayEXTX()
        self.session = self._own(
            xr.destroy_session,
            xr.create_session(self.instance, xr.SessionCreateInfo(system_id=self.system, next=overlay)),
        )
        self.stage = self._own(xr.destroy_space, self._reference_space(xr.ReferenceSpaceType.STAGE))
        self._view = self._own(xr.destroy_space, self._reference_space(xr.ReferenceSpaceType.VIEW))
        view_configurations = xr.enumerate_view_configurations(self.instance, self.system)
        primary = (
            xr.ViewConfigurationType.PRIMARY_STEREO
            if xr.ViewConfigurationType.PRIMARY_STEREO in view_configurations
            else view_configurations[0]
        )
        xr.begin_session(self.session, xr.SessionBeginInfo(primary_view_configuration_type=primary))

    def _own(self, destroy: Any, handle: Any) -> Any:
        self._handles.append((handle, destroy))
        return handle

    def _reference_space(self, space_type: xr.ReferenceSpaceType) -> xr.Space:
        return xr.create_reference_space(
            self.session, xr.ReferenceSpaceCreateInfo(reference_space_type=space_type)
        )

    def now(self) -> xr.Time:
        """Return the current time on the runtime's clock."""
        seconds, nanoseconds = divmod(time.clock_gettime_ns(time.CLOCK_MONOTONIC), 1_000_000_000)
        return xr.convert_timespec_time_to_time_khr(
            self.instance, xr.timespec(tv_sec=seconds, tv_nsec=nanoseconds)
        )

    def head(self) -> Any:
        """Locate the head now, as an ``is_valid``/``pose`` pair."""
        location = xr.locate_space(self._view, self.stage, self.now())
        is_valid = location.location_flags & _POSE_VALID == _POSE_VALID
        return SimpleNamespace(is_valid=is_valid, pose=location.pose if is_valid else xr.Posef())

    def close(self) -> None:
        """Destroy every handle, newest first."""
        while self._handles:
            handle, destroy = self._handles.pop()
            destroy(handle)


class HardwarePoseSource:
    """Acquire a fresh head pose while lazily retrying OpenXR startup."""

    _OPEN_LOG_AT_ATTEMPTS = (1, 2, 4, 8, 16, 32, 64, 128)

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session: HeadlessSession | None = None
        self._open_attempts = 0
        self._last_open_error: str | None = None
        self._next_log_index = 0

    def _try_open(self) -> str | None:
        if self._session is not None:
            return None
        self._open_attempts += 1
        try:
            session = HeadlessSession("openxr-service")
        except xr.XrException as exc:
            # Expected until a headset connects to the runtime.
            error = f"{type(exc).__name__}: {exc}"
            self._last_open_error = error
            self._log_open_failure(error)
            return error
        self._session = session
        self._last_open_error = None
        logger.info("openxr-service: session opened (attempt={})", self._open_attempts)
        return None

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

    def get_pose(self) -> dict[str, Any]:
        with self._lock:
            if self._session is None:
                error = self._try_open()
                if error is not None:
                    return unavailable_pose(error)
            data = self._session.head()
        return from_native_pose(data)

    def health(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": "ok",
                "session_open": self._session is not None,
                "open_attempts": self._open_attempts,
                "last_open_error": self._last_open_error,
            }

    def close(self) -> None:
        with self._lock:
            if self._session is not None:
                self._session.close()
                self._session = None
