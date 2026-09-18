// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.xrai.streamkitsample.streamkit.backends.livekit

import io.livekit.android.room.Room
import io.livekit.android.room.datastream.StreamBytesOptions
import io.livekit.android.room.datastream.outgoing.ByteStreamSender
import io.livekit.android.room.datastream.outgoing.writeFile
import io.livekit.android.room.participant.Participant
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import java.io.File

internal data class ByteStreamWireOptions(
    val topic: String,
    val attributes: Map<String, String> = emptyMap(),
    val destinationIdentities: List<Participant.Identity> = emptyList(),
    val mimeType: String,
    val name: String,
    val totalSize: Long,
)

internal data class ByteStreamConnection(
    val room: Room,
    val generation: Long,
)

internal class ByteStreamConnectionChanged : IllegalStateException(
    "The byte stream is no longer on the active connection",
)

internal class LiveKitByteStreamWriter(
    private val snapshotConnection: () -> ByteStreamConnection?,
    private val isConnectionActive: (ByteStreamConnection) -> Boolean,
) {
    suspend fun sendBytes(data: ByteArray, options: ByteStreamWireOptions): String =
        send(options) { sender -> sender.write(data).getOrThrow() }

    suspend fun sendFile(file: File, options: ByteStreamWireOptions): String =
        send(options) { sender -> sender.writeFile(file).getOrThrow() }

    private suspend fun send(
        options: ByteStreamWireOptions,
        write: suspend (ByteStreamSender) -> Unit,
    ): String {
        val connection = snapshotConnection() ?: throw ByteStreamConnectionChanged()
        if (!isConnectionActive(connection)) throw ByteStreamConnectionChanged()
        val sender = connection.room.localParticipant.streamBytes(
            StreamBytesOptions(
                topic = options.topic,
                attributes = options.attributes,
                destinationIdentities = options.destinationIdentities,
                mimeType = options.mimeType,
                name = options.name,
                totalSize = options.totalSize,
            ),
        )
        try {
            write(sender)
            sender.close()
            if (sender.isOpen || !isConnectionActive(connection)) {
                throw ByteStreamConnectionChanged()
            }
        } catch (error: Throwable) {
            closeAfterFailure(sender)
            throw error
        }
        return sender.info.id
    }

    private suspend fun closeAfterFailure(sender: ByteStreamSender) {
        withContext(NonCancellable) {
            withTimeoutOrNull(2_000) {
                runCatching { sender.close("StreamKit send failed") }
            }
        }
    }
}
