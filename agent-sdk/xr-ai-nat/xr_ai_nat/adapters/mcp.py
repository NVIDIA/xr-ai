# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expose an explicit set of native NAT functions to MCP-only agents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, get_args, get_origin

from fastmcp import FastMCP
from fastmcp.exceptions import ValidationError as FastMCPValidationError
from fastmcp.tools import Tool
from fastmcp.tools.base import ToolResult
from nat.plugin_api import Function
from pydantic import PrivateAttr, TypeAdapter, ValidationError


def _wrapped_schema(schema: dict[str, Any]) -> dict[str, Any]:
    definitions = schema.pop("$defs", None)
    wrapped = {
        "type": "object",
        "properties": {"result": schema},
        "required": ["result"],
        "x-fastmcp-wrap-result": True,
    }
    if definitions:
        wrapped["$defs"] = definitions
    return wrapped


def _output_schema(function: Function, *, untyped: bool) -> dict[str, Any]:
    if untyped:
        output_type = function.single_output_type
        if get_origin(output_type) is list:
            item_type = get_args(output_type)[0]
            item_schema = TypeAdapter(item_type).json_schema(mode="serialization")
            if item_schema.get("type") == "object":
                item_schema = {"type": "object", "additionalProperties": True}
            return _wrapped_schema({"type": "array", "items": item_schema})
        return {"type": "object", "additionalProperties": True}

    schema = TypeAdapter(function.single_output_type).json_schema(mode="serialization")
    if schema.get("type") == "object":
        return schema
    return _wrapped_schema(schema)


class _NatFunctionTool(Tool):
    _function: Function = PrivateAttr()

    def __init__(
        self,
        function: Function,
        *,
        name: str | None = None,
        untyped_output: bool = False,
    ) -> None:
        if not function.has_single_output:
            raise TypeError("MCP exports require a NAT function with a single-response path")
        super().__init__(
            name=name or function.instance_name,
            description=function.description or "",
            parameters=function.input_schema.model_json_schema(mode="validation"),
            output_schema=_output_schema(function, untyped=untyped_output),
        )
        self._function = function

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            request = self._function.input_schema.model_validate(arguments)
        except ValidationError as exc:
            raise FastMCPValidationError(str(exc)) from exc
        result = await self._function.ainvoke(request)
        serialized = TypeAdapter(self._function.single_output_type).dump_python(
            result,
            mode="json",
        )
        return self.convert_result(serialized)


def create_mcp_server(
    name: str,
    functions: Sequence[Function],
    *,
    tool_names: Mapping[str, str] | None = None,
    untyped_outputs: set[str] | None = None,
) -> FastMCP:
    """Publish one ordinary MCP tool per explicitly selected NAT function."""

    aliases = dict(tool_names or {})
    function_names = {function.instance_name for function in functions}
    if unknown := set(aliases) - function_names:
        raise ValueError(f"MCP aliases reference unknown functions: {sorted(unknown)}")
    names = [aliases.get(function.instance_name, function.instance_name) for function in functions]
    if len(names) != len(set(names)):
        raise ValueError("MCP tool names must be unique")
    untyped = set(untyped_outputs or ())
    if unknown := untyped - function_names:
        raise ValueError(f"Untyped MCP outputs reference unknown functions: {sorted(unknown)}")

    server = FastMCP(name, strict_input_validation=True, mask_error_details=True)
    for function, tool_name in zip(functions, names, strict=True):
        server.add_tool(
            _NatFunctionTool(
                function,
                name=tool_name,
                untyped_output=function.instance_name in untyped,
            )
        )
    return server


__all__ = ["create_mcp_server"]
