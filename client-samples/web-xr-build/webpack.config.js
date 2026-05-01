// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const path = require('path');

/**
 * Bundles @nvidia/cloudxr (CommonJS) into a single ESM file so the browser
 * can load it directly via <script type="module"> + import map.
 *
 * Output: ../web/vendor/cloudxr-sdk.esm.mjs  (committed)
 */
module.exports = {
  entry: './src/index.js',
  mode: 'production',
  target: ['web', 'es2020'],

  experiments: {
    outputModule: true,
  },

  output: {
    path: path.resolve(__dirname, '..', 'web', 'vendor'),
    filename: 'cloudxr-sdk.esm.mjs',
    library: { type: 'module' },
    module: true,
    environment: { module: true, dynamicImport: true },
    clean: false,
  },

  resolve: {
    extensions: ['.js'],
  },

  performance: {
    // The SDK bundle is legitimately large (~2 MB). Don't warn about it —
    // it ships once and is cached by the browser.
    hints: false,
  },
};
