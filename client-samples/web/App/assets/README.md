<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Capture connection-loss audio

`capture-connection-lost.wav` is locally synthesized speech saying:
“Connection lost. Pause your demonstration and check the connection.”
It contains no user recording. The web client decodes it before an outage so
warning playback does not depend on the return-audio connection or TTS server.

Regenerate with FFmpeg built with libflite:

```bash
ffmpeg -f lavfi -i "flite=text='Connection lost. Pause your demonstration and check the connection.':voice=slt" -ar 16000 -ac 1 capture-connection-lost.wav
```

Playback requires a browser audio context unlocked by the Connect gesture. A
text warning remains available if loading or audio playback is unavailable.
