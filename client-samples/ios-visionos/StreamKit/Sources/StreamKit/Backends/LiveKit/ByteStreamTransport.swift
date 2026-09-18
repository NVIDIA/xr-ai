// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import Foundation
import LiveKit

internal struct ByteStreamWireOptions: Sendable {
    let topic: String
    let attributes: [String: String]
    let destinationIdentities: [Participant.Identity]
    let mimeType: String?
    let name: String?
    let totalSize: Int?
}

internal struct ByteStreamConnection: @unchecked Sendable {
    let room: Room
    let generation: UInt64
}

internal enum ByteStreamTransportError: Error {
    case connectionChanged
}

private actor ByteStreamCloseRace {
    private var completed = false
    private var waiter: CheckedContinuation<Void, Never>?

    func wait() async {
        if completed { return }
        await withCheckedContinuation { waiter = $0 }
    }

    func finish() {
        guard !completed else { return }
        completed = true
        waiter?.resume()
        waiter = nil
    }
}

internal enum LiveKitByteStreamWriter {
    static func sendBytes(
        _ data: Data,
        options: ByteStreamWireOptions,
        connection: ByteStreamConnection,
        isConnectionActive: (ByteStreamConnection) -> Bool
    ) async throws -> String {
        try await send(
            source: .data(data),
            options: options,
            connection: connection,
            isConnectionActive: isConnectionActive
        )
    }

    static func sendFile(
        _ fileURL: URL,
        options: ByteStreamWireOptions,
        connection: ByteStreamConnection,
        isConnectionActive: (ByteStreamConnection) -> Bool
    ) async throws -> String {
        try await send(
            source: .file(fileURL),
            options: options,
            connection: connection,
            isConnectionActive: isConnectionActive
        )
    }

    private enum Source {
        case data(Data)
        case file(URL)
    }

    private static func send(
        source: Source,
        options: ByteStreamWireOptions,
        connection: ByteStreamConnection,
        isConnectionActive: (ByteStreamConnection) -> Bool
    ) async throws -> String {
        guard isConnectionActive(connection) else {
            throw ByteStreamTransportError.connectionChanged
        }
        let writer = try await connection.room.localParticipant.streamBytes(
            options: StreamByteOptions(
                topic: options.topic,
                attributes: options.attributes,
                destinationIdentities: options.destinationIdentities,
                mimeType: options.mimeType,
                name: options.name,
                totalSize: options.totalSize
            )
        )
        do {
            switch source {
            case .data(let data):
                try await writer.write(data)
            case .file(let fileURL):
                let handle = try FileHandle(forReadingFrom: fileURL)
                defer { try? handle.close() }
                while let chunk = try handle.read(upToCount: 64 * 1024), !chunk.isEmpty {
                    try Task.checkCancellation()
                    guard isConnectionActive(connection) else {
                        throw ByteStreamTransportError.connectionChanged
                    }
                    try await writer.write(chunk)
                }
            }
            try Task.checkCancellation()
            try await writer.close()
            guard await writer.isOpen == false, isConnectionActive(connection) else {
                throw ByteStreamTransportError.connectionChanged
            }
        } catch {
            await closeAfterFailure(writer)
            throw error
        }
        return writer.info.id
    }

    private static func closeAfterFailure(_ writer: ByteStreamWriter) async {
        let race = ByteStreamCloseRace()
        let closeTask = Task {
            _ = try? await writer.close(reason: "StreamKit send failed")
            await race.finish()
        }
        let timeoutTask = Task {
            _ = try? await Task.sleep(nanoseconds: 2_000_000_000)
            await race.finish()
        }
        await race.wait()
        closeTask.cancel()
        timeoutTask.cancel()
    }
}
