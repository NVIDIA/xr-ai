// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';

import {
  ByteStreamConnectionChanged,
  LiveKitByteStreamWriter,
} from '../../client-samples/web/StreamKit/Backends/LiveKit/ByteStreamTransport.js';


function makeTransport({ onWrite = () => {} } = {}) {
  let active = true;
  const writes = [];
  const closeReasons = [];
  const options = [];
  const sdkWriter = {
    info: { id: 'stream-1' },
    async write(chunk) {
      writes.push([...chunk]);
      onWrite();
    },
    async close(reason) {
      closeReasons.push(reason);
    },
  };
  const room = {
    localParticipant: {
      async streamBytes(value) {
        options.push(value);
        return sdkWriter;
      },
    },
  };
  const connection = { room, generation: 1 };
  const transport = new LiveKitByteStreamWriter(
    () => active ? connection : null,
    candidate => active && candidate === connection,
  );
  return {
    transport,
    writes,
    closeReasons,
    options,
    disconnect() {
      active = false;
    },
  };
}


test('byte-stream writer sends bytes with raw wire options', async () => {
  const state = makeTransport();
  const wireOptions = {
    topic: 'camera.capture.response',
    attributes: { request_id: 'request-1' },
    totalSize: 3,
  };

  assert.equal(
    await state.transport.sendBytes(new Uint8Array([1, 2, 3]), wireOptions),
    'stream-1',
  );
  assert.deepEqual(state.options, [wireOptions]);
  assert.deepEqual(state.writes, [[1, 2, 3]]);
  assert.deepEqual(state.closeReasons, [undefined]);
});


test('byte-stream writer streams file chunks', async () => {
  const state = makeTransport();
  const chunks = [new Uint8Array([1, 2]), new Uint8Array([3])];
  const file = {
    stream() {
      return new ReadableStream({
        start(controller) {
          chunks.forEach(chunk => controller.enqueue(chunk));
          controller.close();
        },
      });
    },
  };

  await state.transport.sendFile(file, { topic: '_streamkit.file', totalSize: 3 });

  assert.deepEqual(state.writes, [[1, 2], [3]]);
  assert.deepEqual(state.closeReasons, [undefined]);
});


test('byte-stream writer closes a stream interrupted by disconnect', async () => {
  let state;
  state = makeTransport({ onWrite: () => state.disconnect() });

  await assert.rejects(
    state.transport.sendBytes(new Uint8Array([1]), { topic: 'capture' }),
    ByteStreamConnectionChanged,
  );
  assert.deepEqual(state.closeReasons, ['StreamKit send failed']);
});
