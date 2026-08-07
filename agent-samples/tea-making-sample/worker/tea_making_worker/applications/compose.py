# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose the sample applications from generic routed NAT functions."""

from __future__ import annotations

from dataclasses import dataclass

from nat.builder.function import Function
from nat.builder.workflow_builder import WorkflowBuilder
from nat.plugin_api import LLMRef

from ..agents import AgentRegistry
from ..desktop.functions import add_desktop_functions, desktop_status_function
from ..desktop.registry import Desktop
from ..desktop.runtime import DesktopRuntime
from ..desktop.spec import DesktopSpec
from ..desktop.types import FunctionEffect, RoutedFunction
from .background import BackgroundRegistry, Notice, TextOutput
from .change_watch import ChangeWatchApplication
from .controls import background_function_specs
from .transcript import TranscriptApplication
from .video_log import VideoLogApplication


@dataclass(frozen=True, slots=True)
class HostedApplications:
    desktop: Desktop
    backgrounds: BackgroundRegistry
    change_watch: ChangeWatchApplication
    transcript: TranscriptApplication
    video_log: VideoLogApplication


async def build_applications(
    builder: WorkflowBuilder,
    *,
    llm_ref: LLMRef,
    spec: DesktopSpec,
    runtime: DesktopRuntime,
    tea: AgentRegistry,
    current_view: Function,
    notice: Notice,
    text_output: TextOutput,
) -> HostedApplications:
    desktop = Desktop(spec, runtime)
    backgrounds = BackgroundRegistry()
    change_watch = ChangeWatchApplication(spec.application("change_watch"), runtime, text_output)
    transcript = TranscriptApplication(spec.application("transcript"), runtime, text_output)
    video_log = VideoLogApplication(spec.application("video_log"), runtime)
    await change_watch.build(builder, llm_ref, current_view)
    await transcript.build(builder, llm_ref)
    await video_log.build(builder, llm_ref, current_view)
    await add_desktop_functions(builder, runtime)
    root_functions = root_function_specs(spec)
    desktop.register_foreground("tea", tea)
    await desktop.build(builder, llm_ref, root_functions)
    backgrounds.register(change_watch)
    backgrounds.register(transcript)
    backgrounds.register(video_log)
    return HostedApplications(desktop, backgrounds, change_watch, transcript, video_log)


def root_function_specs(spec: DesktopSpec) -> tuple[RoutedFunction, ...]:
    background = tuple(
        function
        for app in spec.applications.values()
        if app.mode == "background"
        for function in background_function_specs(app)
    )
    return (
        RoutedFunction("current_view", spec.capabilities["current_view"]),
        RoutedFunction("rag_lookup", spec.capabilities["rag_lookup"]),
        RoutedFunction(
            "workflow__start",
            spec.application("tea").route,
            effect=FunctionEffect.FOREGROUND,
            return_direct=True,
        ),
        desktop_status_function(),
        *background,
    )


__all__ = ["HostedApplications", "build_applications", "root_function_specs"]
