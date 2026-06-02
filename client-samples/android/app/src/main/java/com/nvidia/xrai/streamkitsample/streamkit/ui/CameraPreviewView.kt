// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * StreamKit — CameraPreviewView
 *
 * Jetpack Compose view that renders the local camera track of a
 * StreamSession.  Wraps LiveKit's TextureViewRenderer in an AndroidView so
 * application code does not have to depend on the LiveKit SDK directly.
 */

package com.nvidia.xrai.streamkitsample.streamkit.ui

import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import com.nvidia.xrai.streamkitsample.streamkit.StreamSession
import io.livekit.android.renderer.TextureViewRenderer

/**
 * Renders the local camera feed published by [session].
 *
 * Place anywhere in your layout to give the user a "what the camera sees"
 * preview matching the web client's `<video>` element.  The view is empty
 * when [StreamSession.localCameraTrack] is `null` (camera stopped, or
 * non-LiveKit backend).
 *
 * Example:
 * ```kotlin
 * Box(Modifier.aspectRatio(16f / 9f)) {
 *     CameraPreviewView(session = streamSession)
 * }
 * ```
 */
@Composable
fun CameraPreviewView(
    session: StreamSession?,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val track = session?.localCameraTrack ?: return

    val renderer = remember(context) {
        TextureViewRenderer(context).also { session?.initVideoRenderer(it) }
    }

    DisposableEffect(track, renderer) {
        track.addRenderer(renderer)
        onDispose {
            track.removeRenderer(renderer)
            renderer.release()
        }
    }

    AndroidView(
        factory = { renderer },
        modifier = modifier.fillMaxSize(),
    )
}
