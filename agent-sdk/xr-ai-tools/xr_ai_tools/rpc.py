# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private correlated msgpack transport for typed capability services."""

from __future__ import annotations

import asyncio
import inspect
import uuid
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any

import msgpack
import zmq
import zmq.asyncio

_PROTOCOL_VERSION = 1

Dispatch = Callable[[str, dict[str, Any]], dict[str, Any] | Awaitable[dict[str, Any]]]


class RPCError(RuntimeError):
    """A transport, protocol, or remote execution failure."""

    def __init__(self, message: str, *, code: str = "rpc_error") -> None:
        super().__init__(message)
        self.code = code


def _pack(value: dict[str, Any]) -> bytes:
    packed = msgpack.packb(value, use_bin_type=True)
    if not isinstance(packed, bytes):
        raise RPCError("msgpack encoder returned no bytes", code="encoding_error")
    return packed


def _unpack(value: bytes) -> dict[str, Any]:
    decoded = msgpack.unpackb(value, raw=False)
    if not isinstance(decoded, dict):
        raise RPCError("RPC frame must contain a map", code="invalid_frame")
    return decoded


class RPCClient:
    """Concurrent DEALER client with request correlation and bounded calls."""

    def __init__(self, endpoint: str, *, timeout_s: float = 10.0) -> None:
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self._socket: zmq.asyncio.Socket | None = None
        self._receiver: asyncio.Task[None] | None = None
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._send_lock = asyncio.Lock()

    def _ensure_started(self) -> None:
        if self._socket is not None:
            return
        socket = zmq.asyncio.Context.instance().socket(zmq.DEALER)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self.endpoint)
        self._socket = socket
        self._receiver = asyncio.create_task(
            self._receive(),
            name="xr-service-rpc-client",
        )

    async def call(
        self,
        operation: str,
        arguments: dict[str, Any] | None = None,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        self._ensure_started()
        assert self._socket is not None
        request_id = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            async with self._send_lock:
                await self._socket.send(
                    _pack(
                        {
                            "version": _PROTOCOL_VERSION,
                            "request_id": request_id,
                            "operation": operation,
                            "arguments": arguments or {},
                        }
                    )
                )
            try:
                response = await asyncio.wait_for(
                    future,
                    timeout=self.timeout_s if timeout_s is None else timeout_s,
                )
            except TimeoutError as exc:
                raise RPCError(
                    f"{operation} timed out calling {self.endpoint}",
                    code="timeout",
                ) from exc
        finally:
            self._pending.pop(request_id, None)
        if not response.get("ok"):
            error = response.get("error") or {}
            raise RPCError(
                str(error.get("message") or "remote operation failed"),
                code=str(error.get("code") or "remote_error"),
            )
        result = response.get("result")
        if not isinstance(result, dict):
            raise RPCError("RPC result must contain a map", code="invalid_response")
        return result

    async def _receive(self) -> None:
        assert self._socket is not None
        try:
            while True:
                response = _unpack(await self._socket.recv())
                request_id = response.get("request_id")
                if not isinstance(request_id, str):
                    continue
                future = self._pending.get(request_id)
                if future is not None and not future.done():
                    future.set_result(response)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = RPCError(f"RPC receive failed: {exc}", code="connection_error")
            for future in tuple(self._pending.values()):
                if not future.done():
                    future.set_exception(error)
            socket, self._socket = self._socket, None
            self._receiver = None
            if socket is not None:
                socket.close(linger=0)

    async def close(self) -> None:
        receiver, self._receiver = self._receiver, None
        if receiver is not None:
            receiver.cancel()
            with suppress(asyncio.CancelledError):
                await receiver
        for future in tuple(self._pending.values()):
            if not future.done():
                future.set_exception(RPCError("RPC client closed", code="closed"))
        self._pending.clear()
        socket, self._socket = self._socket, None
        if socket is not None:
            socket.close(linger=0)


class RPCServer:
    """ROUTER server dispatching correlated requests without framework coupling."""

    def __init__(self, endpoint: str, dispatch: Dispatch) -> None:
        self.endpoint = endpoint
        self._dispatch = dispatch
        self._socket: zmq.asyncio.Socket | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._send_lock = asyncio.Lock()

    async def serve(self, *, ready: Callable[[], None] | None = None) -> None:
        if self._socket is not None:
            raise RuntimeError("RPC server is already running")
        socket = zmq.asyncio.Context.instance().socket(zmq.ROUTER)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(self.endpoint)
        self._socket = socket
        if ready is not None:
            ready()
        try:
            while True:
                frames = await socket.recv_multipart()
                if len(frames) != 2:
                    continue
                identity, payload = frames
                task = asyncio.create_task(
                    self._handle(identity, payload),
                    name="xr-service-rpc-request",
                )
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
        finally:
            await self.close()

    async def _handle(self, identity: bytes, payload: bytes) -> None:
        request_id = ""
        try:
            request = _unpack(payload)
            request_id = str(request.get("request_id") or "")
            if request.get("version") != _PROTOCOL_VERSION:
                raise RPCError(
                    "unsupported RPC protocol version",
                    code="unsupported_version",
                )
            operation = request.get("operation")
            arguments = request.get("arguments")
            if not isinstance(operation, str) or not isinstance(arguments, dict):
                raise RPCError("invalid RPC request", code="invalid_request")
            result = self._dispatch(operation, arguments)
            if inspect.isawaitable(result):
                result = await result
            if not isinstance(result, dict):
                raise RPCError("RPC handler must return a map", code="invalid_result")
            response = {
                "version": _PROTOCOL_VERSION,
                "request_id": request_id,
                "ok": True,
                "result": result,
            }
        except RPCError as exc:
            response = {
                "version": _PROTOCOL_VERSION,
                "request_id": request_id,
                "ok": False,
                "error": {"code": exc.code, "message": str(exc)},
            }
        except Exception as exc:
            response = {
                "version": _PROTOCOL_VERSION,
                "request_id": request_id,
                "ok": False,
                "error": {"code": "internal_error", "message": str(exc)},
            }
        if self._socket is not None:
            async with self._send_lock:
                await self._socket.send_multipart([identity, _pack(response)])

    async def close(self) -> None:
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        socket, self._socket = self._socket, None
        if socket is not None:
            socket.close(linger=0)


__all__ = ["RPCClient", "RPCError", "RPCServer"]
