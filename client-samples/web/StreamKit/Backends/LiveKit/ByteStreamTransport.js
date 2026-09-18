// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Internal capability used by StreamSession without adding a public backend API.
export const INTERNAL_SEND_BYTE_STREAM = Symbol('StreamKit.internalSendByteStream');

export class ByteStreamConnectionChanged extends Error {
  constructor() {
    super('The byte stream is no longer on the active connection');
    this.name = 'ByteStreamConnectionChanged';
  }
}

export class LiveKitByteStreamWriter {
  #snapshotConnection;
  #isConnectionActive;

  constructor(snapshotConnection, isConnectionActive) {
    this.#snapshotConnection = snapshotConnection;
    this.#isConnectionActive = isConnectionActive;
  }

  async sendBytes(data, options) {
    const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);
    return this.#send(options, async (writer, connection) => {
      this.#requireActive(connection);
      await writer.write(bytes);
    });
  }

  async sendFile(file, options) {
    return this.#send(options, async (writer, connection) => {
      const reader = file.stream().getReader();
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          this.#requireActive(connection);
          await writer.write(value);
        }
      } finally {
        reader.releaseLock();
      }
    });
  }

  async #send(options, write) {
    const connection = this.#snapshotConnection();
    if (!connection) throw new ByteStreamConnectionChanged();
    this.#requireActive(connection);
    const writer = await connection.room.localParticipant.streamBytes(options);
    try {
      await write(writer, connection);
      this.#requireActive(connection);
      await writer.close();
      this.#requireActive(connection);
    } catch (error) {
      let closeTimeout;
      try {
        await Promise.race([
          writer.close('StreamKit send failed').catch(() => {}),
          new Promise(resolve => {
            closeTimeout = setTimeout(resolve, 2000);
          }),
        ]);
      } finally {
        clearTimeout(closeTimeout);
      }
      throw error;
    }
    return writer.info.id;
  }

  #requireActive(connection) {
    if (!this.#isConnectionActive(connection)) {
      throw new ByteStreamConnectionChanged();
    }
  }
}
