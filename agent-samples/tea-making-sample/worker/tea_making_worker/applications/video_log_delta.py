# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NAT commit surface for rolling visual deltas."""

import json
from typing import Any, Protocol

from nat.plugin_api import Builder, FunctionBaseConfig, FunctionInfo, register_function
from pydantic import BaseModel, ConfigDict, Field

from ..runtime.scope import current_invocation
from ..runtime.state import Session
from .video_log_store import VideoLogState


class VideoDeltaRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    delta: str = Field(min_length=1, max_length=400)


class VideoDeltaCommitter(Protocol):
    async def commit(
        self,
        session: Session,
        state: VideoLogState,
        caption: str,
        trace_id: str,
        delta: str,
    ) -> None: ...


class VideoDeltaConfig(FunctionBaseConfig, name="voice_desktop_video_log_commit"):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    application: Any = Field(exclude=True, repr=False)


@register_function(config_type=VideoDeltaConfig)
async def video_log_commit(config: VideoDeltaConfig, _builder: Builder):
    async def commit(request: VideoDeltaRequest) -> str:
        call = current_invocation()
        await config.application.commit(
            call.session,
            call.context["video_log.state"],
            str(call.context["video_log.caption"]),
            call.trace_id,
            request.delta.strip(),
        )
        return json.dumps({"recorded": True}, separators=(",", ":"))

    yield FunctionInfo.from_fn(
        commit,
        description="Append the current broad caption and its unique visible delta to the video log.",
    )


async def add_video_log_commit(builder: Builder, application: VideoDeltaCommitter) -> None:
    await builder.add_function(
        "video_log__commit",
        VideoDeltaConfig(application=application),
    )


__all__ = ["VideoDeltaRequest", "add_video_log_commit"]
