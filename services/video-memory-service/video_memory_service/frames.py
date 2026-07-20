# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decode recorded H.264 and normalize hub pixels for PNG output."""

import ctypes
from pathlib import Path

import numpy as np
from PIL import Image
from xr_ai_agent import FrameData, PixelFormat


def _yuv_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    y = y.astype(np.float32) - 16
    cb = cb.astype(np.float32) - 128
    cr = cr.astype(np.float32) - 128
    rgb = np.stack(
        (
            1.164 * y + 1.596 * cr,
            1.164 * y - 0.392 * cb - 0.813 * cr,
            1.164 * y + 2.017 * cb,
        ),
        axis=-1,
    )
    return np.clip(rgb, 0, 255).astype(np.uint8)


def nv12_to_rgb(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[:height, :]
    uv = frame[height:, :].reshape(height // 2, width // 2, 2)
    cb = np.repeat(np.repeat(uv[:, :, 0], 2, axis=0), 2, axis=1)
    cr = np.repeat(np.repeat(uv[:, :, 1], 2, axis=0), 2, axis=1)
    return _yuv_to_rgb(y, cb, cr)


def live_frame_to_rgb(frame: FrameData) -> np.ndarray:
    array = np.frombuffer(frame.data, dtype=np.uint8)
    if frame.fmt == PixelFormat.RGB24:
        return array.reshape(frame.height, frame.width, 3).copy()
    if frame.fmt == PixelFormat.RGBA:
        return array.reshape(frame.height, frame.width, 4)[:, :, :3].copy()
    if frame.fmt == PixelFormat.BGRA:
        return array.reshape(frame.height, frame.width, 4)[:, :, [2, 1, 0]].copy()
    if frame.fmt == PixelFormat.NV12:
        return nv12_to_rgb(
            array.reshape(frame.height * 3 // 2, frame.width),
            frame.width,
            frame.height,
        )
    if frame.fmt == PixelFormat.I420:
        y_size = frame.width * frame.height
        uv_size = (frame.width // 2) * (frame.height // 2)
        y = array[:y_size].reshape(frame.height, frame.width)
        u = array[y_size : y_size + uv_size].reshape(frame.height // 2, frame.width // 2)
        v = array[y_size + uv_size :].reshape(frame.height // 2, frame.width // 2)
        u = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1)
        v = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)
        return _yuv_to_rgb(y, u, v)
    raise ValueError(f"Unsupported PixelFormat for PNG export: {frame.fmt!r}")


def save_png(rgb: np.ndarray, path: Path) -> None:
    Image.fromarray(rgb, "RGB").save(path, "PNG")


def _copy_decoded_frame(frame) -> np.ndarray:
    size = frame.shape[0] * frame.shape[1]
    view = (ctypes.c_uint8 * size).from_address(frame.GetPtrToPlane(0))
    return np.ctypeslib.as_array(view).reshape(frame.shape).copy()


def decode_h264(data: bytes, gpu_id: int) -> list[np.ndarray]:
    import PyNvVideoCodec as nvc

    decoder = nvc.CreateDecoder(
        gpuid=gpu_id,
        codec=nvc.cudaVideoCodec.H264,
        cudacontext=0,
        cudastream=0,
        usedevicememory=False,
    )
    source = np.frombuffer(data, dtype=np.uint8)
    packet = nvc.PacketData()
    packet.bsl = int(source.size)
    packet.bsl_data = int(source.ctypes.data)
    frames = [_copy_decoded_frame(frame) for frame in decoder.Decode(packet)]
    end = nvc.PacketData()
    end.bsl = 0
    end.bsl_data = 0
    end.decode_flag = int(nvc.VideoPacketFlag.ENDOFSTREAM)
    frames.extend(_copy_decoded_frame(frame) for frame in decoder.Decode(end))
    return frames
