# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NAT commit surface for periodic transcript summaries."""

from typing import Any, Protocol

from nat.plugin_api import Builder, FunctionBaseConfig, FunctionInfo, register_function
from pydantic import BaseModel, ConfigDict, Field

from ..runtime.scope import current_invocation
from ..runtime.state import Session


class TranscriptSummaryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=500)


class TranscriptSummarizer(Protocol):
    async def commit_summary(
        self,
        session: Session,
        state: Any,
        turns: tuple[str, ...],
        summary: str,
    ) -> None: ...


class TranscriptSummaryConfig(FunctionBaseConfig, name="voice_application_transcript_summary_commit"):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    application: Any = Field(exclude=True, repr=False)


@register_function(config_type=TranscriptSummaryConfig)
async def transcript_summary_commit(config: TranscriptSummaryConfig, _builder: Builder):
    async def commit(request: TranscriptSummaryRequest) -> str:
        call = current_invocation()
        state = call.context["transcript.state"]
        turns = call.context["transcript.turns"]
        await config.application.commit_summary(call.session, state, turns, request.summary)
        return request.summary

    yield FunctionInfo.from_fn(
        commit,
        description="Persist one concise summary of the supplied transcript turns.",
    )


async def add_transcript_summary(builder: Builder, application: TranscriptSummarizer) -> None:
    await builder.add_function(
        "transcript__commit_summary",
        TranscriptSummaryConfig(application=application),
    )


__all__ = ["TranscriptSummaryRequest", "add_transcript_summary"]
