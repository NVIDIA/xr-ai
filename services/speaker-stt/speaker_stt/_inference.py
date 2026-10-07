# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo speaker-conditioned ASR for one enrolled speaker."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from math import ceil
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
from xr_ai_voicegate._speaker import _SpeakerConfig

from ._selection import _Selection

if TYPE_CHECKING:
    from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
    from torch import Tensor


@dataclass
class _AsrState:
    cache: tuple[Tensor, Tensor, Tensor]
    previous_hypotheses: list[Hypothesis] | None = None
    previous_pred: list[Tensor] | None = None


class _Models:
    def __init__(self, cfg: dict) -> None:
        import torch
        from nemo.collections.asr.models import ASRModel, SortformerEncLabelModel
        from nemo.collections.asr.parts.utils.multispk_transcribe_utils import (
            configure_diar_streaming,
            validate_feature_frame_strides,
        )
        from omegaconf import OmegaConf

        self.device = torch.device(cfg.get("device", "cuda"))
        self.precision = cfg.get("precision", "bf16")
        if self.precision not in {"bf16", "float32"}:
            raise ValueError("precision must be bf16 or float32")
        self.asr = ASRModel.from_pretrained(cfg["asr_model"]).eval().to(self.device)
        if not hasattr(self.asr, "set_speaker_targets"):
            raise ValueError("ASR model must support speaker conditioning")
        self.diar = SortformerEncLabelModel.from_pretrained(cfg["diar_model"]).eval().to(self.device)
        self.asr.encoder.set_default_att_context_size([70, 13])
        self.cfg = OmegaConf.create(
            {
                "deploy_mode": True,
                "streaming_mode": True,
                "max_num_of_spks": 8,
                "batch_size": 1,
                "fix_prev_words_count": 5,
                "update_prev_words_sentence": 5,
                "ignored_initial_frame_steps": 0,
                "att_context_size": [70, 13],
                "cache_gating": True,
                "cache_gating_buffer_size": 1,
                "binary_diar_preds": True,
                "masked_asr": False,
                "single_speaker_mode": False,
                "diar_right_context": 0,
                "spkcache_len": None,
                "spkcache_update_period": 222,
                "fifo_len": 264,
                "log": False,
            }
        )
        streaming = self.asr.encoder.streaming_cfg
        self.nframes = streaming.valid_out_len + streaming.cache_drop_size
        configure_diar_streaming(
            self.diar,
            self.cfg,
            self.asr.encoder.subsampling_factor,
            self.nframes,
        )
        validate_feature_frame_strides(self.asr, self.diar)
        self.hop_samples = round(streaming.valid_out_len * self.asr.encoder.subsampling_factor * 160)
        cache_size = streaming.pre_encode_cache_size
        if isinstance(cache_size, (tuple, list)):
            cache_size = cache_size[-1]
        self.cache_samples = round(cache_size * 160)
        self._warmup()

    def _warmup(self) -> None:
        # The first decoder call initializes GPU kernels and can take seconds.
        # Exercise streaming and utterance-final decoding before readiness,
        # in a disposable session that cannot enroll a real participant.
        seconds = self.hop_samples / 16000
        session = self._session(
            _SpeakerConfig(
                silence_duration=2.5 * seconds,
                max_utterance_s=max(30.0, 4 * seconds),
            ),
            audio_origin_us=0,
        )
        session.selection.owner = 0
        session.selection.candidate = 0
        for second in range(ceil(3 * seconds)):
            session._feed(bytes(32000))

    def _session(self, cfg: _SpeakerConfig, *, audio_origin_us: int) -> _Session:
        return _Session(self, cfg, audio_origin_us=audio_origin_us)


class _Session:
    def __init__(self, models: _Models, cfg: _SpeakerConfig, *, audio_origin_us: int) -> None:
        import torch
        from nemo.collections.asr.parts.utils.multispk_transcribe_utils import SpeakerTaggedASR
        from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

        self.models = models
        self.selection = _Selection(cfg)
        self.adapter = SpeakerTaggedASR(models.cfg, models.asr, models.diar)
        # Only the diarization adapter is used. The multi-speaker driver would
        # allocate ASR caches for every preceding speaker index.
        self.adapter.instance_manager.diar_states = SimpleNamespace(
            streaming_state=models.diar.sortformer_modules.init_streaming_state(batch_size=1, device=models.device),
            diar_pred_out_stream=torch.zeros((1, 0, 8), device=models.device),
        )
        self.buffer = CacheAwareStreamingAudioBuffer(model=models.asr, online_normalization=False)
        self.pending = np.zeros(models.cache_samples, dtype=np.float32)
        self.step = 0
        self.audio_origin_us = audio_origin_us
        self.last_partial = ""
        self._reset_asr()

    def _reset_asr(self) -> None:
        self._asr = _AsrState(self.models.asr.encoder.get_initial_cache_state(batch_size=1))
        self.last_partial = ""

    def _decode(
        self,
        encoded,
        encoded_lengths,
        predictions,
        selected: int,
        drop: int,
        stream: _AsrState,
        finished: bool,
    ) -> str:
        model = self.models.asr
        if encoded.shape[1] != predictions.shape[1]:
            raise ValueError("speaker targets are not aligned with ASR encoder frames")
        # Cache and hypothesis history belong to one speaker; model weights and
        # pre-encoded features are shared across serial decoder calls.
        target = predictions[:, :, selected] > 0.5
        others = [i for i in range(8) if i != selected]
        interference = (predictions[:, :, others] > 0.5).any(dim=-1)
        model.set_speaker_targets(target.float(), interference.float())
        (
            stream.previous_pred,
            transcriptions,
            channel,
            time,
            length,
            stream.previous_hypotheses,
        ) = model.conformer_stream_step(
            processed_signal=encoded,
            processed_signal_length=encoded_lengths,
            cache_last_channel=stream.cache[0],
            cache_last_time=stream.cache[1],
            cache_last_channel_len=stream.cache[2],
            keep_all_outputs=finished,
            previous_hypotheses=stream.previous_hypotheses,
            previous_pred_out=stream.previous_pred,
            drop_extra_pre_encoded=drop,
            return_transcription=True,
            bypass_pre_encode=True,
        )
        stream.cache = (channel, time, length)
        text = transcriptions[0]
        if not isinstance(text, str):
            text = text.text
        return text or ""

    def _feed(self, audio: bytes) -> list[dict]:
        import torch

        if len(audio) % 2:
            raise ValueError("audio must contain complete signed 16-bit PCM samples")
        samples = np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32768.0
        if samples.size > 16000:
            raise ValueError("an audio message must be at most one second")
        self.pending = np.concatenate((self.pending, samples))
        frame_samples = self.models.hop_samples + self.models.cache_samples
        events = []
        amp = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.models.device.type == "cuda" and self.models.precision == "bf16"
            else nullcontext()
        )
        with torch.inference_mode(), amp:
            while len(self.pending) >= frame_samples:
                frame = self.pending[:frame_samples].copy()
                self.pending = self.pending[self.models.hop_samples :]
                features, lengths = self.buffer.preprocess_audio(frame, device=self.models.device)
                features = features[:, :, : int(lengths[0])]
                # The first window also contains the normal pre-encoder cache
                # (padded with silence), so its extra frames must be dropped.
                drop = self.models.asr.encoder.streaming_cfg.drop_extra_pre_encoded
                states = self.adapter.instance_manager.diar_states
                states.streaming_state, all_preds = self.adapter._forward_diarization_streaming_step(
                    features,
                    lengths,
                    drop,
                )
                preds = all_preds[:, -self.models.nframes :]
                # Historical output tensors are not the identity cache. Retain
                # only the alignment window; the AOSC and FIFO stay untouched.
                states.diar_pred_out_stream = preds.detach().clone()
                recent = preds[:, -self.models.asr.encoder.streaming_cfg.valid_out_len :]
                active_mask = recent.amax(dim=1)[0] >= self.selection.cfg.activity_threshold
                active = set(active_mask.nonzero().flatten().tolist())
                chunk_pts = self.audio_origin_us + round(
                    self.step * self.models.hop_samples * 1_000_000 / 16000
                )
                selected, edges, started = self.selection._activity(
                    active,
                    self.models.hop_samples / 16000,
                    chunk_pts,
                )
                events.extend(edges)
                if started:
                    self._reset_asr()
                if selected is not None:
                    encoded, encoded_lengths = self.adapter.forward_pre_encoded(features, lengths, drop)
                    self.selection.text = self._decode(
                        encoded,
                        encoded_lengths,
                        preds,
                        selected,
                        drop,
                        self._asr,
                        self.selection._finished,
                    )
                    text = self.selection.text
                    if self.selection.owner is not None and text and text != self.last_partial:
                        events.append({"kind": "partial", "text": text})
                        self.last_partial = text
                if self.selection._finished:
                    # Forced truncation cannot enroll a voice from a prefix or
                    # release enrollment before the complete phrase is known.
                    text = self.selection.text
                    if self.selection.duration >= self.selection.cfg.max_utterance_s:
                        text = ""
                    events.extend(self.selection._finish(text))
                    self._reset_asr()
                self.step += 1
        return events
