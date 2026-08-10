# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NAT commit surface for prompt-driven visual change decisions."""

import json
from typing import Any, Protocol

from nat.plugin_api import Builder, FunctionBaseConfig, FunctionInfo, register_function
from pydantic import BaseModel, ConfigDict, Field

from ..runtime.scope import current_invocation
from ..runtime.state import Session


class ChangeCommitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    important: bool
    summary: str = Field(default="", max_length=300)


class ChangeCommitter(Protocol):
    async def commit(
        self,
        session: Session,
        caption: str,
        request: ChangeCommitRequest,
    ) -> bool: ...


class ChangeCommitConfig(FunctionBaseConfig, name="voice_application_change_watch_commit"):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    application: Any = Field(exclude=True, repr=False)


@register_function(config_type=ChangeCommitConfig)
async def change_watch_commit(config: ChangeCommitConfig, _builder: Builder):
    async def commit(request: ChangeCommitRequest) -> str:
        call = current_invocation()
        caption = str(call.context["change_watch.caption"])
        notified = await config.application.commit(call.session, caption, request)
        return json.dumps(
            {"important": request.important, "notified": notified},
            separators=(",", ":"),
        )

    yield FunctionInfo.from_fn(
        commit,
        description="Record the current caption and notify only when the visible change is important.",
    )


async def add_change_commit(builder: Builder, application: ChangeCommitter) -> None:
    await builder.add_function(
        "change_watch__commit",
        ChangeCommitConfig(application=application),
    )


__all__ = ["ChangeCommitRequest", "add_change_commit"]
