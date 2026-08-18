<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Lab instrument sample markers

These markers match `yaml/device_map.yaml`. The image encodes the raw marker ID;
the sample resolves that ID to the device name shown in the filename and table.

| File | Marker family | Encoded ID | Device name |
|---|---|---|---|
| `qr/Device1_QR_device-1.png` | QR code | `device-1` | `Device1` |
| `qr/Device2_QR_device-2.png` | QR code | `device-2` | `Device2` |
| `qr/Device3_QR_device-3.png` | QR code | `device-3` | `Device3` |
| `qr/Device4_QR_device-4.png` | QR code | `device-4` | `Device4` |
| `qr/Device5_QR_device-5.png` | QR code | `device-5` | `Device5` |
| `aruco/Device1_ArUco_0.png` | ArUco `DICT_4X4_50` | `0` | `Device1` |
| `aruco/Device2_ArUco_1.png` | ArUco `DICT_4X4_50` | `1` | `Device2` |
| `aruco/Device3_ArUco_2.png` | ArUco `DICT_4X4_50` | `2` | `Device3` |
| `aruco/Device4_ArUco_3.png` | ArUco `DICT_4X4_50` | `3` | `Device4` |
| `aruco/Device5_ArUco_4.png` | ArUco `DICT_4X4_50` | `4` | `Device5` |

Keep the white border around each marker, print without interpolation or
cropping, and place the marker close enough to its instrument display for both
to appear clearly in the same camera frame.
