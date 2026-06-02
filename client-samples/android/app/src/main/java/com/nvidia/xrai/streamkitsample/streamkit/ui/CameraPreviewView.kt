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
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import com.nvidia.xrai.streamkitsample.streamkit.StreamSession
import io.livekit.android.renderer.TextureViewRenderer
import livekit.org.webrtc.VideoSink

/**
 * Renders the local camera feed published by [session].
 *
 * Place anywhere in your layout to give the user a "what the camera sees"
 * preview matching the web client's `<video>` element.  The view is empty
 * when [StreamSession.localCameraTrack] is `null` (camera stopped, or
 * non-LiveKit backend).
 *
 * Pair with [rememberCameraPreviewAspectRatio] to size the surrounding
 * container to the actual capture aspect ratio (so portrait phone capture
 * renders as 9:16, landscape sensors as 16:9):
 *
 * ```kotlin
 * val aspect = rememberCameraPreviewAspectRatio(session) ?: (16f / 9f)
 * Box(Modifier.aspectRatio(aspect)) {
 *     CameraPreviewView(session = session)
 * }
 * ```
 */
@Composable
fun CameraPreviewView(
    session: StreamSession?,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val track = session?.localCameraTrack

    // Renderer lifetime is keyed to the track instance — recreate (and
    // release) the TextureViewRenderer whenever the track identity changes
    // so the addRenderer/removeRenderer pair always operates on a live
    // renderer. Track-less compositions still execute the hooks below to
    // keep the slot table stable across null ↔ non-null transitions.
    val renderer = remember(context, session, track) {
        if (track == null) null
        else TextureViewRenderer(context).also { session.initVideoRenderer(it) }
    }

    DisposableEffect(track, renderer) {
        if (track != null && renderer != null) {
            track.addRenderer(renderer)
        }
        onDispose {
            if (track != null && renderer != null) {
                track.removeRenderer(renderer)
                renderer.release()
            }
        }
    }

    if (renderer != null) {
        AndroidView(
            factory = { renderer },
            modifier = modifier.fillMaxSize(),
        )
    }
}

/**
 * Observes the live camera track's frame dimensions and returns the latest
 * aspect ratio (`width / height`, rotation-corrected) for Compose layout.
 *
 * Returns `null` until the first frame arrives, so callers should pair it
 * with a sensible fallback (typically 16:9):
 *
 * ```kotlin
 * val aspect = rememberCameraPreviewAspectRatio(session) ?: (16f / 9f)
 * Box(Modifier.aspectRatio(aspect)) {
 *     CameraPreviewView(session = session)
 * }
 * ```
 *
 * Internally attaches a lightweight [VideoSink] to the track that only
 * inspects the frame's rotated dimensions — no decoding or rendering work.
 */
@Composable
fun rememberCameraPreviewAspectRatio(session: StreamSession?): Float? {
    val track = session?.localCameraTrack
    val aspect = remember(track) { mutableStateOf<Float?>(null) }

    DisposableEffect(track) {
        if (track == null) {
            aspect.value = null
            return@DisposableEffect onDispose { }
        }

        val sink = VideoSink { frame ->
            val w = frame.rotatedWidth
            val h = frame.rotatedHeight
            if (w > 0 && h > 0) {
                val newAspect = w.toFloat() / h.toFloat()
                if (aspect.value != newAspect) {
                    aspect.value = newAspect
                }
            }
        }
        track.addRenderer(sink)

        onDispose { track.removeRenderer(sink) }
    }

    return aspect.value
}
