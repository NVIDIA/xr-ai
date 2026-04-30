// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.xrai.streamkitsample

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import com.nvidia.xrai.streamkitsample.streamkit.config.CameraConfig

/**
 * One row in the camera picker.
 *
 * @property id          Android Camera2 camera id (also accepted by LiveKit's
 *                       `LocalVideoTrackOptions.deviceId`).
 * @property displayName Human-readable label shown in the dropdown.
 * @property facing      Coarse facing direction; used to pick a sensible default.
 */
data class CameraInfo(
    val id: String,
    val displayName: String,
    val facing: CameraConfig.CameraFacing?,
)

/**
 * Enumerate all cameras visible to Camera2 on this device.
 *
 * Returns an empty list if the platform refuses to expose any (unusual). Listing
 * cameras and reading their characteristics does **not** require the
 * `CAMERA` permission — only opening the camera does — so this can be called
 * before the permission dialog has been shown.
 */
fun enumerateCameras(context: Context): List<CameraInfo> {
    val manager = context.getSystemService(Context.CAMERA_SERVICE) as? CameraManager
        ?: return emptyList()

    val ids = try {
        manager.cameraIdList
    } catch (_: Exception) {
        return emptyList()
    }

    return ids.mapNotNull { id ->
        try {
            val chars = manager.getCameraCharacteristics(id)
            val facing = when (chars.get(CameraCharacteristics.LENS_FACING)) {
                CameraCharacteristics.LENS_FACING_FRONT -> CameraConfig.CameraFacing.FRONT
                CameraCharacteristics.LENS_FACING_BACK -> CameraConfig.CameraFacing.BACK
                else -> null  // EXTERNAL or unknown
            }

            // Some devices expose multiple cameras per direction (wide / ultra-wide /
            // telephoto). Use focal length to give the user a hint about which is which.
            val focalLengths = chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
            val lensHint = focalLengths?.firstOrNull()?.let { fl ->
                when {
                    fl < 2.5f  -> "ultra-wide"
                    fl > 6.0f  -> "telephoto"
                    else       -> "wide"
                }
            }

            val facingLabel = when (facing) {
                CameraConfig.CameraFacing.FRONT -> "Front"
                CameraConfig.CameraFacing.BACK  -> "Back"
                null                            -> "External"
            }
            val label = buildString {
                append(facingLabel)
                if (lensHint != null) append(" — ").append(lensHint)
                append("  (id ").append(id).append(")")
            }

            CameraInfo(id = id, displayName = label, facing = facing)
        } catch (_: Exception) {
            null
        }
    }
}
