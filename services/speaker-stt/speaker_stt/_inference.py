# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo speaker-conditioned ASR with optional display-only diagnostic streams."""

from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from math import ceil
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from xr_ai_voicegate._speaker import _SpeakerConfig

from ._selection import _Selection

_FINAL_PADDING_SECONDS = 0.32


if TYPE_CHECKING:
    from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
    from torch import Tensor


@dataclass
class _AsrState:
    cache: tuple[Tensor, Tensor, Tensor]
    # Tokens of each recent step since the last utterance, newest last.
    recent: deque[list[int]]
    previous_hypotheses: list[Hypothesis] | None = None
    emitted: list[int] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)
    pending: list[int] = field(default_factory=list)

    def _lead_in(self) -> list[int]:
        """Tokens decoded in the quiet steps before this one."""
        return [token for tokens in list(self.recent)[:-1] for token in tokens]


@dataclass
class _DecodedStream:
    asr: _AsrState
    pts_us: int
    duration: float = 0.0
    silence: float = 0.0
    text: str = ""


@dataclass(frozen=True)
class _CompletedStream:
    pts_us: int
    duration: float
    text: str


class _Models:
    def __init__(self, cfg: dict) -> None:
        import torch
        from nemo.collections.asr.models import ASRModel, SortformerEncLabelModel
        from nemo.collections.asr.parts.submodules.rnnt_greedy_decoding import GreedyBatchedRNNTInfer
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
        decoder = self.asr.decoding.decoding
        if not isinstance(decoder, GreedyBatchedRNNTInfer) or not decoder.loop_labels:
            raise ValueError("speaker ASR requires greedy_batch label-loop decoding")
        self.diar = SortformerEncLabelModel.from_pretrained(cfg["diar_model"]).eval().to(self.device)
        self.asr.encoder.set_default_att_context_size([70, 1])
        self.cfg = OmegaConf.create(
            {
                "deploy_mode": True,
                "streaming_mode": True,
                "max_num_of_spks": 8,
                "batch_size": 1,
                "fix_prev_words_count": 5,
                "update_prev_words_sentence": 5,
                "ignored_initial_frame_steps": 0,
                "att_context_size": [70, 1],
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
        self.history_hops = ceil(streaming.last_channel_cache_size / self.nframes)
        # Bounded terminal quiet for this model's subword completion; it is
        # evaluated on a snapshot and never advances accepted capture time.
        self.final_padding_hops = ceil(_FINAL_PADDING_SECONDS * 16000 / self.hop_samples)
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
        self._speaker_states: dict[int, _AsrState] = {}
        self._history = deque(maxlen=models.history_hops)
        self._frame = None
        self._unavailable: set[int] = set()
        self._candidate_streams: dict[int, _DecodedStream] = {}
        self._candidate_completed: list[tuple[int, _CompletedStream]] = []
        self._truncated_episodes: dict[int, float] = {}
        self._diagnostic_streams: dict[int, _DecodedStream] = {}
        self._diagnostic_active: dict[int, str] = {}
        self._reset_asr()

    def _new_asr(self) -> _AsrState:
        # The decoder can emit a speaker's first word before diarization marks
        # them active. Words within one silence_duration of activity belong to
        # the utterance, mirroring the quiet that ends one.
        lead_in = ceil(self.selection.cfg.silence_duration * 16000 / self.models.hop_samples)
        return _AsrState(self.models.asr.encoder.get_initial_cache_state(batch_size=1), deque(maxlen=lead_in + 1))

    def _speaker_asr(self, speaker: int) -> _AsrState:
        if speaker not in self._speaker_states:
            state = self._new_asr()
            for encoded, lengths, predictions, drop in self._history:
                self._step_native(encoded, lengths, predictions, speaker, drop, state, False)
            state.emitted = []
            self._speaker_states[speaker] = state
        return self._speaker_states[speaker]

    def _reset_asr(self) -> None:
        self._asr = self._speaker_asr(self.selection.owner) if self.selection.owner is not None else None
        if self._asr is not None:
            # A finished utterance's words cannot lead into the next one.
            if self.selection.candidate is None:
                self._asr.recent.clear()
            self._asr.tokens = self._asr._lead_in()
        self.last_partial = ""

    def _decode_streams(
        self,
        predictions,
        drop: int,
        active: set[int],
        pts_us: int,
        streams: dict[int, _DecodedStream],
        *,
        required: set[int],
    ) -> tuple[dict[int, str], list[tuple[int, _DecodedStream]]]:
        """Finalize attributed streams from this hop's native decoder output."""
        speakers = (active | streams.keys()) - self._unavailable
        if not speakers:
            return {}, []
        texts, completed = {}, []
        seconds = self.models.hop_samples / 16000
        for speaker in sorted(speakers, key=lambda speaker: (speaker not in required, speaker)):
            try:
                if speaker not in streams:
                    state = self._speaker_asr(speaker)
                    state.tokens = state._lead_in()
                    streams[speaker] = _DecodedStream(state, pts_us)
                stream = streams[speaker]
                stream.duration += seconds
                stream.silence = 0.0 if speaker in active else stream.silence + seconds
                finished = stream.silence >= self.selection.cfg.silence_duration or (
                    stream.duration >= self.selection.cfg.max_utterance_s
                )
                stream.text = self._decode(
                    predictions, speaker, drop, stream.asr,
                    stream.silence >= self.selection.cfg.silence_duration,
                )
            except Exception:
                if speaker in required:
                    raise
                streams.pop(speaker, None)
                logger.exception("optional speaker stream failed speaker={}", speaker)
                continue
            texts[speaker] = stream.text
            if finished:
                completed.append((speaker, stream))
                stream.asr.recent.clear()
                del streams[speaker]
        return texts, completed

    def _candidate_step(
        self, predictions, drop: int, active: set[int], pts_us: int,
    ) -> list[dict]:
        """Decode attributed candidates before resolving this step's enrollment."""
        self._candidate_completed = []
        cfg = self.selection.cfg
        seconds = self.models.hop_samples / 16000
        for speaker, quiet in list(self._truncated_episodes.items()):
            quiet = 0.0 if speaker in active else quiet + seconds
            if quiet >= cfg.silence_duration:
                del self._truncated_episodes[speaker]
            else:
                self._truncated_episodes[speaker] = quiet
        speakers = (active | self._candidate_streams.keys()) - self._truncated_episodes.keys()
        if not speakers:
            return []
        # All pre-enrollment streams are required: losing one could turn a
        # simultaneous phrase tie into a false unique nomination.
        _, finished = self._decode_streams(
            predictions,
            drop,
            active - self._truncated_episodes.keys(),
            pts_us,
            self._candidate_streams,
            required=speakers,
        )
        completed = []
        for speaker, stream in finished:
            truncated = stream.duration >= cfg.max_utterance_s
            completed.append((speaker, stream.text, stream.pts_us, truncated))
            self._candidate_completed.append((speaker, _CompletedStream(stream.pts_us, stream.duration, stream.text)))
            if truncated:
                self._truncated_episodes[speaker] = 0.0
        events = self.selection._enroll_completed(
            completed,
            completed_at_us=pts_us + round(self.models.hop_samples * 1_000_000 / 16000),
        )
        if self.selection.owner is not None:
            self._candidate_streams.clear()
            self._truncated_episodes.clear()
            if not cfg.diagnostics:
                self._speaker_states = {self.selection.owner: self._speaker_states[self.selection.owner]}
        return events

    def _step_native(self, encoded, lengths, predictions, selected, drop, state, finished):
        from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

        model = self.models.asr
        if encoded.shape[1] != predictions.shape[1]:
            raise ValueError("speaker targets are not aligned with ASR encoder frames")
        others = [i for i in range(8) if i != selected]
        model.set_speaker_targets(
            (predictions[:, :, selected] > 0.5).float(), (predictions[:, :, others] > 0.5).any(dim=-1).float()
        )
        (_, _, channel, time, length, hypotheses) = model.conformer_stream_step(
            processed_signal=encoded,
            processed_signal_length=lengths,
            cache_last_channel=state.cache[0],
            cache_last_time=state.cache[1],
            cache_last_channel_len=state.cache[2],
            keep_all_outputs=finished,
            previous_hypotheses=state.previous_hypotheses,
            previous_pred_out=None,
            drop_extra_pre_encoded=drop,
            return_transcription=True,
            bypass_pre_encode=True,
        )
        state.cache = (channel, time, length)
        hypothesis = hypotheses[0]
        tokens = hypothesis.y_sequence.tolist()
        # Native continuation consumes the opaque predictor state, not the
        # accumulated transcript. Keep output metadata bounded to this step.
        state.previous_hypotheses = [
            Hypothesis(
                score=0.0,
                y_sequence=[],
                dec_state=hypothesis.dec_state,
                last_token=hypothesis.last_token,
                biasing_cfg=hypothesis.biasing_cfg,
            )
        ]
        # The padded suffix is a guess. Real audio confirms it token by token;
        # once real audio decodes differently, its tokens are new words.
        while tokens and state.pending and tokens[0] == state.pending[0]:
            tokens.pop(0)
            state.pending.pop(0)
        if tokens:
            state.pending.clear()
        state.emitted = tokens
        state.recent.append(tokens)

    def _decode(self, predictions, selected, drop, stream, finished) -> str:
        stream.tokens.extend(stream.emitted)
        if finished:
            # Virtual terminal silence must not advance the live audio timeline
            # or its encoder cache; later real audio reconciles emitted tokens.
            snapshot = deepcopy(stream)
            frame = self._frame
            additional = []
            for _ in range(self.models.final_padding_hops):
                frame = np.concatenate(
                    (
                        frame[-self.models.cache_samples :] if self.models.cache_samples else frame[:0],
                        np.zeros(self.models.hop_samples, dtype=np.float32),
                    )
                )
                features, lengths = self.buffer.preprocess_audio(frame, device=self.models.device)
                features = features[:, :, : int(lengths[0])]
                padded, padded_lengths = self.adapter.forward_pre_encoded(features, lengths, drop)
                # The padding flushes the selected speaker's last subwords, so
                # it stays attributed to them; marking them silent drops those.
                targets = predictions.new_zeros((padded.shape[0], padded.shape[1], 8))
                targets[:, :, selected] = 1.0
                self._step_native(padded, padded_lengths, targets, selected, drop, snapshot, True)
                additional.extend(snapshot.emitted)
            stream.pending.extend(additional)
            stream.tokens.extend(additional)
        return self.models.asr.tokenizer.ids_to_text(stream.tokens)

    def _diagnostic_events(
        self,
        active: set[int],
        completed: list[tuple[int, _DecodedStream | _CompletedStream]],
        owner_before: int | None,
        pts_us: int,
    ) -> list[dict]:
        owner = self.selection.owner
        waiting = "waiting for enrollment" if self.selection.cfg.mode == "conversation" else "listening"
        statuses = {
            speaker: "enrolled" if speaker == owner else "ignored" if owner is not None else waiting
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
            events.append(
                {"kind": "diagnostic", "speaker_id": owner, "status": "enrolled", "pts_us": pts_us, "text": ""}
            )
        self._diagnostic_active = statuses
        for speaker, stream in completed:
            if speaker in {owner_before, owner} or not stream.text.strip():
                continue
            status = "ignored" if owner is not None else f"ignored; {waiting}"
            if stream.duration >= self.selection.cfg.max_utterance_s:
                status += "; truncated"
            events.append(
                {
                    "kind": "diagnostic",
                    "speaker_id": speaker,
                    "status": status,
                    "pts_us": stream.pts_us,
                    "text": stream.text.strip(),
                }
            )
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
                chunk_pts = self.audio_origin_us + round(self.step * self.models.hop_samples * 1_000_000 / 16000)
                self._frame = frame
                encoded_features = self.adapter.forward_pre_encoded(features, lengths, drop)
                needed = active if self.selection.owner is None or self.selection.cfg.diagnostics else set()
                if self.selection.owner is not None:
                    needed = needed | {self.selection.owner}
                # A completed candidate's padded suffix remains authoritative
                # until subsequent real audio reconciles it, even in silence.
                required = (
                    (active | self._candidate_streams.keys() | {
                        speaker for speaker, state in self._speaker_states.items() if state.pending
                    })
                    if self.selection.owner is None
                    else {self.selection.owner}
                )
                self._unavailable = set()
                for speaker in sorted(
                    needed | self._speaker_states.keys(), key=lambda value: (value not in required, value)
                ):
                    try:
                        state = self._speaker_asr(speaker)
                        self._step_native(*encoded_features, preds, speaker, drop, state, False)
                    except Exception:
                        if speaker in required:
                            raise
                        # isolation boundary: display-only inference cannot revoke the owner.
                        self._unavailable.add(speaker)
                        self._speaker_states.pop(speaker, None)
                        self._diagnostic_streams.pop(speaker, None)
                        logger.exception("optional speaker continuation failed speaker={}", speaker)
                if self.selection.owner is None:
                    candidate_events = self._candidate_step(
                        preds, drop, active, chunk_pts,
                    )
                    events.extend(candidate_events)
                    if self.selection.cfg.diagnostics:
                        accepted = {
                            event["speaker_id"]
                            for event in candidate_events
                            if event["kind"] in {"one_shot", "wake_ready"}
                        }
                        completed = [
                            (speaker, stream)
                            for speaker, stream in self._candidate_completed
                            if speaker not in accepted
                        ]
                        events.extend(self._diagnostic_events(active, completed, None, chunk_pts))
                    self._history.append((*encoded_features, preds.detach().clone(), drop))
                    self.step += 1
                    continue
                selected, edges, started = self.selection._activity(
                    active,
                    self.models.hop_samples / 16000,
                    chunk_pts,
                )
                events.extend(edges)
                if started or self._asr is None:
                    self._reset_asr()
                owner_before = self.selection.owner
                completed = []
                if selected is not None:
                    self.selection.text = self._decode(
                        preds,
                        selected,
                        drop,
                        self._asr,
                        self.selection._finished and self.selection.silence >= self.selection.cfg.silence_duration,
                    )
                    text = self.selection.text
                    if self.selection.owner is not None and text and text != self.last_partial:
                        events.append({"kind": "partial", "text": text})
                        self.last_partial = text
                if self.selection.cfg.diagnostics:
                    _, completed = self._decode_streams(
                        preds,
                        drop,
                        active - {owner_before},
                        chunk_pts,
                        self._diagnostic_streams,
                        required=set(),
                    )
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
                if self.selection.owner is None:
                    # Display-only history cannot seed the next enrollment.
                    self._diagnostic_streams.clear()
                    self._speaker_states = {owner_before: self._speaker_states[owner_before]}
                self._history.append((*encoded_features, preds.detach().clone(), drop))
                self.step += 1
        return events
