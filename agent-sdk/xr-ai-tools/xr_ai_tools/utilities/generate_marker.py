# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "opencv-contrib-python-headless>=4.8,<5",
# ]
# ///

"""Generate QR codes and ArUco markers as PNG images."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def _qr_image(value: str, size: int, border: int):
    if not value:
        raise ValueError("QR value must not be empty")
    if border < 0:
        raise ValueError("QR border must not be negative")

    modules = cv2.QRCodeEncoder_create().encode(value)
    module_count = modules.shape[0] + 2 * border
    scale = size // module_count
    if scale < 1:
        raise ValueError(
            f"size must be at least {module_count} pixels for this QR value and border"
        )

    bordered = cv2.copyMakeBorder(
        modules,
        border,
        border,
        border,
        border,
        cv2.BORDER_CONSTANT,
        value=255,
    )
    rendered_size = module_count * scale
    rendered = cv2.resize(
        bordered,
        (rendered_size, rendered_size),
        interpolation=cv2.INTER_NEAREST,
    )
    padding = size - rendered_size
    leading = padding // 2
    return cv2.copyMakeBorder(
        rendered,
        leading,
        padding - leading,
        leading,
        padding - leading,
        cv2.BORDER_CONSTANT,
        value=255,
    )


def _aruco_image(marker_id: int, dictionary_name: str, size: int, margin: int):
    dictionary_id = getattr(cv2.aruco, dictionary_name, None)
    if not dictionary_name.startswith("DICT_") or not isinstance(dictionary_id, int):
        raise ValueError(f"unknown ArUco dictionary: {dictionary_name}")
    if margin < 0:
        raise ValueError("ArUco margin must not be negative")

    marker_size = size - 2 * margin
    if marker_size < 1:
        raise ValueError("size must be greater than twice the ArUco margin")

    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    marker_count = dictionary.bytesList.shape[0]
    if marker_id < 0 or marker_id >= marker_count:
        raise ValueError(
            f"marker ID must be between 0 and {marker_count - 1} for {dictionary_name}"
        )

    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size)
    return cv2.copyMakeBorder(
        marker,
        margin,
        margin,
        margin,
        margin,
        cv2.BORDER_CONSTANT,
        value=255,
    )


def _write_png(image, output: Path) -> None:
    if output.suffix.lower() != ".png":
        raise ValueError("output path must end in .png")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise RuntimeError(f"failed to write {output}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a QR code or ArUco marker as a square PNG image."
    )
    subparsers = parser.add_subparsers(dest="marker_type", required=True)

    qr = subparsers.add_parser("qr", help="generate a QR code")
    qr.add_argument("value", help="text encoded by the QR code")
    qr.add_argument("--output", type=Path, default=Path("qr.png"))
    qr.add_argument("--size", type=int, default=512, help="PNG side length in pixels")
    qr.add_argument(
        "--border",
        type=int,
        default=4,
        help="quiet-zone width in QR modules (default: 4)",
    )

    aruco = subparsers.add_parser("aruco", help="generate an ArUco marker")
    aruco.add_argument("marker_id", type=int, help="marker identifier")
    aruco.add_argument("--output", type=Path, default=Path("aruco.png"))
    aruco.add_argument("--size", type=int, default=512, help="PNG side length in pixels")
    aruco.add_argument(
        "--dictionary",
        default="DICT_4X4_50",
        help="OpenCV predefined ArUco dictionary name (default: DICT_4X4_50)",
    )
    aruco.add_argument(
        "--margin",
        type=int,
        default=32,
        help="white margin around the marker in pixels (default: 32)",
    )
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.size < 1:
        parser.error("size must be positive")

    try:
        if args.marker_type == "qr":
            image = _qr_image(args.value, args.size, args.border)
        else:
            image = _aruco_image(
                args.marker_id,
                args.dictionary,
                args.size,
                args.margin,
            )
        _write_png(image, args.output)
    except (cv2.error, RuntimeError, ValueError) as error:
        parser.error(str(error))

    print(args.output)


if __name__ == "__main__":
    main()
