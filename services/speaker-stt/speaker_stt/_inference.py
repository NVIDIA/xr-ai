# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo speaker-conditioned ASR with optional display-only diagnostic streams."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from math import ceil
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
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


@dataclass
class _DiagnosticStream:
    asr: _AsrState
    pts_us: int
    duration: float = 0.0
    silence: float = 0.0
    text: str = ""


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
        self._diagnostic_streams: dict[int, _DiagnosticStream] = {}
        self._diagnostic_active: dict[int, str] = {}
        self._reset_asr()

    def _new_asr(self) -> _AsrState:
        return _AsrState(self.models.asr.encoder.get_initial_cache_state(batch_size=1))

    def _reset_asr(self) -> None:
        if not self.selection.cfg.diagnostics:
            self._asr = self._new_asr()
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

    def _diagnostic_step(
        self, features, lengths, predictions, drop: int, active: set[int], pts_us: int, selected: int | None,
    ) -> tuple[dict[int, str], list[tuple[int, _DiagnosticStream]]]:
        cfg = self.selection.cfg
        texts: dict[int, str] = {}
        completed: list[tuple[int, _DiagnosticStream]] = []
        speakers = active | self._diagnostic_streams.keys()
        if not speakers:
            return texts, completed
        # Encode once; each serial conditioned decoder keeps its own history.
        try:
            encoded, encoded_lengths = self.adapter.forward_pre_encoded(features, lengths, drop)
        except Exception:
            if selected is not None:
                raise
            self._diagnostic_streams.clear()
            logger.exception("speaker diagnostic encoding failed")
            return texts, completed
        seconds = self.models.hop_samples / 16000
        # Allocate and decode the selected speaker first; optional caches must
        # not take the capacity needed to produce the primary transcript.
        for speaker in sorted(speakers, key=lambda speaker: (speaker != selected, speaker)):
            if speaker not in self._diagnostic_streams:
                try:
                    self._diagnostic_streams[speaker] = _DiagnosticStream(self._new_asr(), pts_us)
                except Exception:
                    if speaker == selected:
                        raise
                    logger.exception("speaker diagnostic cache allocation failed speaker={}", speaker)
                    continue
            stream = self._diagnostic_streams[speaker]
            stream.duration += seconds
            stream.silence = 0.0 if speaker in active else stream.silence + seconds
            finished = stream.silence >= cfg.silence_duration or stream.duration >= cfg.max_utterance_s
            try:
                stream.text = self._decode(encoded, encoded_lengths, predictions, speaker, drop, stream.asr, finished)
            except Exception:
                if speaker == selected:
                    raise
                del self._diagnostic_streams[speaker]
                logger.exception("speaker diagnostic decoding failed speaker={}", speaker)
                continue
            texts[speaker] = stream.text
            if finished:
                completed.append((speaker, stream))
                del self._diagnostic_streams[speaker]
        return texts, completed

    def _diagnostic_events(
        self, active: set[int], completed: list[tuple[int, _DiagnosticStream]],
        owner_before: int | None, pts_us: int,
    ) -> list[dict]:
        owner = self.selection.owner
        statuses = {
            speaker: "enrolled" if speaker == owner else "ignored" if owner is not None else "waiting for enrollment"
            for speaker in active
        }
        events = [
            {"kind": "diagnostic", "speaker_id": speaker, "status": status, "pts_us": pts_us, "text": ""}
            for speaker, status in sorted(statuses.items())
            if self._diagnostic_active.get(speaker) != status
        ]
        # Enrollment completes in a silence tail; display the selected identity
        # even if the new owner is no longer acoustically active.
        if owner is not None and owner != owner_before and owner not in active:
            events.append({"kind": "diagnostic", "speaker_id": owner, "status": "enrolled",
                           "pts_us": pts_us, "text": ""})
        self._diagnostic_active = statuses
        for speaker, stream in completed:
            if speaker in {owner_before, owner} or not stream.text.strip():
                continue
            status = "ignored" if owner is not None else "ignored; waiting for enrollment"
            if stream.duration >= self.selection.cfg.max_utterance_s:
                status += "; truncated"
            events.append({"kind": "diagnostic", "speaker_id": speaker, "status": status,
                           "pts_us": stream.pts_us, "text": stream.text.strip()})
        return events

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
                    if self.selection.cfg.diagnostics and selected is not None:
                        # A newly selected candidate must recognize fresh audio;
                        # ignored speech from before release cannot enroll it.
                        self._diagnostic_streams.pop(selected, None)
                owner_before = self.selection.owner
                completed = []
                if self.selection.cfg.diagnostics:
                    texts, completed = self._diagnostic_step(
                        features, lengths, preds, drop, active, chunk_pts, selected,
                    )
                    if selected is not None:
                        self.selection.text = texts.get(selected, "")
                if selected is not None:
                    if not self.selection.cfg.diagnostics:
                        encoded, encoded_lengths = self.adapter.forward_pre_encoded(features, lengths, drop)
                        self.selection.text = self._decode(
                            encoded, encoded_lengths, preds, selected, drop, self._asr, self.selection._finished,
                        )
                    text = self.selection.text
                    if self.selection.owner is not None and text and text != self.last_partial:
                        events.append({"kind": "partial", "text": text})
                        self.last_partial = text
                if self.selection._finished:
                    # Forced truncation cannot enroll a voice from a prefix or
                    # release enrollment before the complete phrase is known.
                    # It remains a valid transcript boundary for the owner.
                    text = self.selection.text
                    truncated = self.selection.duration >= self.selection.cfg.max_utterance_s
                    events.extend(self.selection._finish(text, allow_control=not truncated))
                    self._reset_asr()
                if self.selection.cfg.diagnostics:
                    events.extend(self._diagnostic_events(active, completed, owner_before, chunk_pts))
                self.step += 1
        return events
