# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
llama_nemotron_llm_server — OpenAI-compatible LLM HTTP server (Llama-3.1-Nemotron-Nano-8B-v1).

Loads nvidia/Llama-3.1-Nemotron-Nano-8B-v1 (or any HuggingFace causal LM with
a Llama-3.1-compatible chat template) in-process and serves an OpenAI-compatible
API:

    POST /v1/chat/completions   (text messages; tools=[...] supported via the
                                 native Llama-3.1 chat template; returns
                                 OpenAI-shaped tool_calls / finish_reason)
    GET  /v1/models

Tool calling
------------
When the request includes a non-empty ``tools`` array, the server:

    1. Passes ``tools`` to ``tokenizer.apply_chat_template(..., tools=tools)``
       so the Llama-3.1 chat template renders the tool schema into the prompt
       using the model's native format.
    2. Constrains decoding with ``lm-format-enforcer``: ``model.generate()``
       receives a ``prefix_allowed_tokens_fn`` built from a
       ``UnionParser([tool_call_grammar, free_text])``.  The tool_call grammar
       is the exact shape NVIDIA's vLLM parser expects —
       ``<TOOLCALL>[{"name": "<tool>", "arguments": {...}}]</TOOLCALL>`` —
       with ``arguments`` matching each tool's JSON Schema.  Invalid JSON
       tokens (stray quotes, missing commas, schema echoes, etc.) are
       filtered from the model's vocabulary at every step, so the output is
       always either a syntactically valid tool call or free assistant text.
    3. Parses the generated text for tool-call JSON (still supports legacy
       wrapper variants for robustness when grammar constraints are bypassed).
    4. Returns parsed calls in OpenAI format:
       ``choices[0].message.tool_calls = [{id, type: "function",
        function: {name, arguments}}]`` with ``finish_reason: "tool_calls"``.

Conversation history containing prior tool calls is also supported: assistant
messages with a ``tool_calls`` field and ``tool`` role messages with a
``tool_call_id`` field flow through to the chat template unchanged, which is
what tool-calling agents (e.g. LangChain ``ChatOpenAI.bind_tools()``, NAT's
``tool_calling_agent``) require for multi-turn tool loops.

The reasoning toggle is a pure system-prompt feature of this model: include
"detailed thinking on" or "detailed thinking off" anywhere in a user or
system message. See https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1

Accepts --config <path>.yaml (auto-passed by xr-ai-launcher).

Config keys
-----------
    model:          str    HuggingFace model ID (required)
    port:           int    HTTP port (default: 8106)
    host:           str    Bind address (default: "0.0.0.0")
    hf_token:       str    HuggingFace token for gated models
    system_prompt:  str    Optional system prompt prepended to every request
                           (use "detailed thinking off" to force fast replies)
    model_cache:    str    HuggingFace weight cache path.  Resolved relative to
                           this YAML file's directory.  Default: ../models
    max_new_tokens: int    Max tokens to generate (default: 1024)
    dtype:          str    Torch dtype (default: bfloat16)
    stop:           list   Default stop sequences applied if request omits stop
                           (default: [] — rely on tokenizer EOS handling)
"""
import argparse
import ast
import asyncio
import json
import os
import re
import sys
import threading
import uuid
from pathlib import Path

import yaml

_DEFAULT_PORT = 8106
_DEFAULT_TOKENS = 1024
_DEFAULT_STOP: list[str] = []


def _shim_transformers_for_lmfe() -> None:
    """Make ``lm-format-enforcer==0.11.x`` importable on transformers 5.x.

    LMFE's ``integrations/transformers.py`` does
    ``from transformers.tokenization_utils import PreTrainedTokenizerBase``
    at module load.  In transformers 5.x that class moved to
    ``transformers.tokenization_utils_base``, and the old module no longer
    re-exports it — which makes the entire integrations shim
    ``ImportError`` on load.  Republishing the attribute at the old path
    is enough: LMFE only uses the class as a type annotation and a single
    ``isinstance`` check, both of which are satisfied by the real class.
    """
    try:
        import transformers.tokenization_utils as _tu
        if not hasattr(_tu, "PreTrainedTokenizerBase"):
            from transformers.tokenization_utils_base import PreTrainedTokenizerBase
            _tu.PreTrainedTokenizerBase = PreTrainedTokenizerBase
    except ImportError:
        # transformers not installed yet — _ensure_loaded will import it later;
        # just skip the shim, LMFE imports will happen after transformers loads.
        pass


def _resolve_model_cache(cfg: dict, yaml_dir: Path) -> Path:
    raw = cfg.get("model_cache", "../models")
    p = Path(raw)
    if not p.is_absolute():
        p = (yaml_dir / p).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _get_torch_dtype(dtype_str: str):
    import torch
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping.get(dtype_str, torch.bfloat16)


class _StopStringCriteria:
    """StoppingCriteria that checks decoded text for stop strings."""

    def __init__(self, tokenizer, stop_strings: list[str], prompt_length: int):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        generated_ids = input_ids[0, self.prompt_length:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        return any(stop in text for stop in self.stop_strings)


# ── Llama-3.1 tool-call output parser ─────────────────────────────────────────

# Llama-3.1-Nemotron-Nano-8B-v1's canonical tool-call wrapper is
# ``<TOOLCALL>[...]</TOOLCALL>`` (cf. NVIDIA's own vLLM parser for this family,
# ``llama_nemotron_nano_toolcall_parser.py``).  In practice an 8B model also
# drifts into a handful of near-neighbour variants — ``<TOOL_CALL>``,
# ``<ANGLED_TOOLS>``, and occasionally an open tag with no close — so we
# accept any of those wrappers.  We then validate the payload to avoid
# mistaking a regurgitated tool *schema* for a tool *call*.
_TOOLCALL_BLOCK_RE = re.compile(
    r"<(?:TOOLCALL|TOOL_CALL|ANGLED_TOOLS)>\s*"
    r"(\[.*?\]|\{.*?\})"
    r"(?:\s*</(?:TOOLCALL|TOOL_CALL|ANGLED_TOOLS)>)?",
    re.DOTALL,
)
_PYTHON_TAG_RE = re.compile(r"<\|python_tag\|>\s*(\{.*\})", re.DOTALL)

# Safety-net pattern for the case where the model emits a tool invocation
# as plain Python-style text instead of the canonical <TOOLCALL> block.
# e.g. ``ask_camera(question="What is the color of the hair?")``.
# LMFE's free-text branch (``[^<]*``) syntactically allows this output
# because it never opens a ``<`` tag — the grammar only guarantees that
# either a ``<TOOLCALL>`` block is well-formed OR no ``<`` is emitted.
# 8B models occasionally take the free-text path for borderline tool
# decisions, so we recover it downstream.
_PSEUDO_CALL_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*\((.*)\)\s*\.?\s*$", re.DOTALL)


def _looks_like_json_schema(obj) -> bool:
    """True if *obj* is a JSON Schema (the tool definition), not a tool call.

    Small LLMs occasionally echo the tool catalogue they were shown instead
    of emitting a real tool call.  Those entries carry ``{"type": "object",
    "properties": {...}}`` or ``{"properties": {...}}`` under their
    parameters/arguments field — no real tool-call argument set looks like
    that, so we treat matches as "not a tool call".
    """
    if not isinstance(obj, dict):
        return False
    if obj.get("type") == "object" and isinstance(obj.get("properties"), dict):
        return True
    if isinstance(obj.get("properties"), dict) and isinstance(obj.get("required"), list):
        return True
    return False


def _coerce_tool_call_dict(obj) -> dict | None:
    """Return a normalised ``{"name": str, "arguments": <json-string>}`` dict or None."""
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    if not isinstance(name, str):
        return None

    # A regurgitated tool *definition* has a top-level ``description`` plus
    # a ``parameters`` or ``function.parameters`` that is itself a JSON
    # Schema (``type: object`` + ``properties``).  A real tool *call* carries
    # only a concrete argument dict.  Reject the former so we don't turn the
    # schema echo into a bogus tool invocation.
    if "description" in obj and (
        _looks_like_json_schema(obj.get("parameters"))
        or _looks_like_json_schema(obj.get("arguments"))
    ):
        return None

    # Llama-3.1 uses "arguments" (per the official chat template); some
    # templates use "parameters".  Accept either.
    args = obj.get("arguments", obj.get("parameters", {}))
    if _looks_like_json_schema(args):
        return None
    if isinstance(args, str):
        try:
            args_obj = json.loads(args)
        except json.JSONDecodeError:
            args_obj = {"_raw": args}
    else:
        args_obj = args
    return {"name": name, "arguments": json.dumps(args_obj)}


def _extract_json_candidates(text: str) -> list[dict]:
    """Extract zero or more top-level JSON objects/arrays from free text."""
    out: list[dict] = []

    m = _TOOLCALL_BLOCK_RE.search(text)
    if m:
        try:
            parsed = json.loads(m.group(1))
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            out.extend(x for x in parsed if isinstance(x, dict))
            return out
        if isinstance(parsed, dict):
            out.append(parsed)
            return out

    m = _PYTHON_TAG_RE.search(text)
    if m:
        try:
            parsed = json.loads(m.group(1))
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            out.append(parsed)
            return out
        if isinstance(parsed, list):
            out.extend(x for x in parsed if isinstance(x, dict))
            return out

    # Fall back: treat the whole string as JSON if it trimmed-parses cleanly.
    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            out.append(parsed)
        elif isinstance(parsed, list):
            out.extend(x for x in parsed if isinstance(x, dict))
    return out


def _parse_pseudo_call_args(args_str: str) -> dict | None:
    """Parse ``a=1, b="x"`` (the args portion of a Python-style call) into a dict.

    Uses ``ast`` so we only accept safe literals — no arbitrary code runs.
    Returns ``None`` if the argument list contains positional args,
    ``**kwargs`` expansion, or any value that isn't a Python literal.
    """
    try:
        tree = ast.parse(f"__f__({args_str})", mode="eval")
    except SyntaxError:
        return None
    if not isinstance(tree.body, ast.Call):
        return None
    if tree.body.args:
        # Positional arguments aren't usable here — tool schemas are keyword-only.
        return None
    kwargs: dict = {}
    for kw in tree.body.keywords:
        if kw.arg is None:  # ``**mapping`` unpacking
            return None
        try:
            kwargs[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, SyntaxError):
            return None
    return kwargs


def _try_parse_pseudo_call(text: str, tool_names: set[str]) -> dict | None:
    """Recognise ``tool_name(arg=value, ...)`` plain-text output from the model.

    Returns a ``{"name": str, "arguments": str-json}`` dict matching the
    output of :func:`_coerce_tool_call_dict`, or ``None`` if *text* isn't a
    standalone call to one of the advertised *tool_names*.
    """
    if not tool_names:
        return None
    m = _PSEUDO_CALL_RE.match(text.strip())
    if not m:
        return None
    name, args_str = m.group(1), m.group(2)
    if name not in tool_names:
        return None
    kwargs = _parse_pseudo_call_args(args_str)
    if kwargs is None:
        return None
    return {"name": name, "arguments": json.dumps(kwargs)}


def parse_llama_tool_calls(
    text: str,
    tool_names: set[str] | None = None,
) -> tuple[str | None, list[dict]]:
    """
    Split *text* into (assistant_content, tool_calls).

    Returns ``(text, [])`` unchanged when no tool call is detected, so a caller
    can treat a tool-less generation the same way as before.

    When *tool_names* is provided, also detects Python-style plain-text calls
    like ``ask_camera(question="...")`` — a known failure mode of Llama-Nemotron
    when the LMFE grammar's free-text branch is taken — and coerces them into
    a proper tool call.

    The returned ``tool_calls`` list is in OpenAI wire format:
        ``[{"id": "call_xxx", "type": "function",
            "function": {"name": str, "arguments": str-json}}]``
    """
    if not text:
        return text, []

    candidates = _extract_json_candidates(text)

    normalised: list[dict] = []
    for obj in candidates:
        n = _coerce_tool_call_dict(obj)
        if n is not None:
            normalised.append(n)

    if not normalised:
        # Safety net: catch ``ask_camera(question=...)``-style text before
        # falling back to a tool-less reply.  Only runs when the request
        # advertised tools; otherwise there's no name list to match against.
        if tool_names:
            pseudo = _try_parse_pseudo_call(text, tool_names)
            if pseudo is not None:
                tool_calls = [{
                    "id":       f"call_{uuid.uuid4().hex[:12]}",
                    "type":     "function",
                    "function": pseudo,
                }]
                return None, tool_calls
        return text, []

    # Strip the tool-call markup out of the text that accompanies the call.
    # Llama-3.1 sometimes emits the JSON alone (content becomes "") and
    # sometimes interleaves it with narration; we keep any narration around
    # the JSON block.  The tool-call JSON itself is exclusively delivered via
    # tool_calls to match OpenAI's contract.
    cleaned = _TOOLCALL_BLOCK_RE.sub("", text)
    cleaned = _PYTHON_TAG_RE.sub("", cleaned)
    for obj in candidates:
        cleaned = cleaned.replace(json.dumps(obj), "")
    cleaned = cleaned.strip()

    tool_calls = [
        {
            "id":       f"call_{uuid.uuid4().hex[:12]}",
            "type":     "function",
            "function": n,
        }
        for n in normalised
    ]
    return (cleaned or None), tool_calls


# ── grammar-constrained decoding ──────────────────────────────────────────────

def _build_tool_call_parser(tools: list[dict]):
    """Build the LMFE parser used to constrain ``model.generate()`` output.

    The grammar is ``UnionParser([tool_call_block, free_text])`` where:

      * ``tool_call_block`` = ``<TOOLCALL>[{...valid-JSON...}]</TOOLCALL>``
        and the JSON object inside must match the union of every advertised
        tool's schema — the ``name`` field is a ``const`` of one of the tool
        names, and ``arguments`` is a JSON Schema object with that tool's
        parameters.
      * ``free_text`` = any text that does not open a ``<TOOLCALL>`` tag, so
        the model stays free to answer conversationally (e.g. on "hello").

    Returns None if no valid tool schemas were provided — the caller skips
    constrained decoding in that case.
    """
    from lmformatenforcer import (
        SequenceParser, UnionParser, RegexParser, JsonSchemaParser,
    )

    function_schemas = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        # Accept both flat ``{name, parameters}`` and nested
        # ``{type: "function", function: {name, parameters}}`` (OpenAI).
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = fn.get("name")
        params = fn.get("parameters") or {"type": "object", "properties": {}}
        if not isinstance(name, str):
            continue
        function_schemas.append({
            "type":       "object",
            "properties": {
                "name":      {"const": name, "type": "string"},
                "arguments": params,
            },
            "required":             ["name", "arguments"],
            "additionalProperties": False,
        })

    if not function_schemas:
        return None

    # JSON Schema ``oneOf`` across all advertised tools — the generated object
    # must match exactly one tool's call shape.
    call_object_schema = (
        function_schemas[0]
        if len(function_schemas) == 1
        else {"oneOf": function_schemas}
    )

    tool_call_sequence = SequenceParser([
        RegexParser(r"<TOOLCALL>\["),
        JsonSchemaParser(call_object_schema),
        RegexParser(r"\]</TOOLCALL>"),
    ])
    # Free assistant text: any run of characters that doesn't start a TOOLCALL
    # tag.  (LMFE's RegexParser handles Python-regex-ish syntax.)
    free_text = RegexParser(r"[^<]*")
    return UnionParser([tool_call_sequence, free_text])


class _LlmBackend:
    """Thread-safe lazy loader for AutoModelForCausalLM models."""

    def __init__(
        self,
        model_id: str,
        system_prompt: str,
        model_cache: Path,
        dtype_str: str,
        default_stop: list[str],
    ) -> None:
        self._model_id = model_id
        self._system_prompt = system_prompt
        self._model_cache = model_cache
        self._dtype_str = dtype_str
        self._default_stop = default_stop
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()
        # LMFE's TokenEnforcerTokenizerData is expensive to build (vocab
        # walk); cache it across requests once the tokenizer is loaded.
        self._token_enforcer_data = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._model_cache.mkdir(parents=True, exist_ok=True)
            cache = str(self._model_cache)
            dtype = _get_torch_dtype(self._dtype_str)

            print(f"[llama_nemotron_llm_server] Loading {self._model_id}  cache={cache}  dtype={self._dtype_str}")
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self._model_id, cache_dir=cache, local_files_only=True,
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    self._model_id,
                    torch_dtype=dtype,
                    device_map="auto",
                    cache_dir=cache,
                    local_files_only=True,
                ).eval()
            except OSError:
                print(f"[llama_nemotron_llm_server] Downloading {self._model_id}…")
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self._model_id, cache_dir=cache,
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    self._model_id,
                    torch_dtype=dtype,
                    device_map="auto",
                    cache_dir=cache,
                ).eval()

            dev = next(self._model.parameters()).device
            print(f"[llama_nemotron_llm_server] Model ready on {dev}")

    def _build_prefix_allowed_tokens_fn(self, tools: list[dict]):
        """Return a ``prefix_allowed_tokens_fn`` constrained to *tools* JSON.

        Returns None if no valid tool schemas were extracted (caller then
        runs unconstrained decoding).  The expensive per-tokenizer LMFE
        state is cached on the backend so only the per-request parser is
        rebuilt each call.
        """
        parser = _build_tool_call_parser(tools)
        if parser is None:
            return None
        # Lazy, cached LMFE setup — needs transformers 5.x compatibility shim
        # first (see _shim_transformers_for_lmfe at module scope).
        _shim_transformers_for_lmfe()
        from lmformatenforcer.integrations.transformers import (
            build_token_enforcer_tokenizer_data,
            build_transformers_prefix_allowed_tokens_fn,
        )
        if self._token_enforcer_data is None:
            self._token_enforcer_data = build_token_enforcer_tokenizer_data(
                self._tokenizer,
            )
        return build_transformers_prefix_allowed_tokens_fn(
            self._token_enforcer_data, parser,
        )

    def infer(
        self,
        messages: list[dict],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None,
        tools: list[dict] | None = None,
    ) -> str:
        """Synchronous inference. Call from a thread pool."""
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        self._ensure_loaded()

        # Merge the server's system prompt into the message list without
        # producing two adjacent system messages (which Llama 3.1's chat
        # template rejects with "Conversation roles must alternate ...").
        # If the request already has a system message, prepend our content
        # to its content; otherwise prepend a new system message.
        full_messages = list(messages)
        if self._system_prompt:
            for i, m in enumerate(full_messages):
                if m.get("role") == "system":
                    existing = m.get("content", "")
                    full_messages[i] = {
                        "role": "system",
                        "content": f"{self._system_prompt}\n\n{existing}".strip(),
                    }
                    break
            else:
                full_messages = (
                    [{"role": "system", "content": self._system_prompt}]
                    + full_messages
                )

        chat_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if tools:
            chat_kwargs["tools"] = tools
        prompt = self._tokenizer.apply_chat_template(full_messages, **chat_kwargs)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        prompt_length = inputs.input_ids.shape[1]

        stop_strings = stop if stop else self._default_stop
        stopping_criteria = StoppingCriteriaList()

        class StringStopCriteria(StoppingCriteria):
            def __init__(inner_self, tokenizer, stops, plen):
                inner_self.tokenizer = tokenizer
                inner_self.stops = stops
                inner_self.plen = plen

            def __call__(inner_self, input_ids, scores, **kwargs) -> bool:
                gen_ids = input_ids[0, inner_self.plen:]
                text = inner_self.tokenizer.decode(gen_ids, skip_special_tokens=False)
                return any(s in text for s in inner_self.stops)

        if stop_strings:
            stopping_criteria.append(
                StringStopCriteria(self._tokenizer, stop_strings, prompt_length)
            )

        do_sample = temperature > 0
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self._tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        # Grammar-constrained decoding.  When the caller supplied a non-empty
        # ``tools`` array, force every generated token to keep the running
        # output on a valid path toward either (a) a well-formed
        # ``<TOOLCALL>[{...}]</TOOLCALL>`` block whose JSON matches the
        # advertised tool schemas, or (b) plain assistant text.  This blocks
        # the whole class of "stray character breaks JSON" failures that
        # 8B models occasionally exhibit when emitting tool-call JSON.
        if tools:
            prefix_fn = self._build_prefix_allowed_tokens_fn(tools)
            if prefix_fn is not None:
                gen_kwargs["prefix_allowed_tokens_fn"] = prefix_fn

        with torch.inference_mode():
            output_ids = self._model.generate(**gen_kwargs)

        generated_ids = output_ids[0, prompt_length:]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        for stop_str in stop_strings:
            if stop_str in text:
                text = text.split(stop_str)[0]

        return text.strip()


def _build_app(cfg: dict, model_cache: Path):
    from fastapi import FastAPI, Request

    model_id = cfg["model"]
    system_prompt = cfg.get("system_prompt", "").strip()
    max_new_tokens = int(cfg.get("max_new_tokens", _DEFAULT_TOKENS))
    dtype_str = cfg.get("dtype", "bfloat16")
    default_stop = cfg.get("stop", _DEFAULT_STOP)

    backend = _LlmBackend(model_id, system_prompt, model_cache, dtype_str, default_stop)

    app = FastAPI(title="LLM Server", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [{"id": model_id, "object": "model", "owned_by": "local"}],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        # Parse the raw JSON body directly. We deliberately avoid a strict
        # Pydantic schema here because OpenAI-compatible clients (LangChain,
        # openai-python, etc.) send many optional fields (n, frequency_penalty,
        # presence_penalty, seed, logprobs, reasoning_effort, service_tier,
        # etc.) that we don't care about. Pydantic v2 + FastAPI's schema
        # builder also has trouble with strict models defined inside a
        # function closure (ForwardRef "not fully defined" errors during
        # OpenAPI schema generation), which surfaced as 422 "req missing"
        # for every request — see issue history. This free-form approach
        # sidesteps both problems.
        try:
            body = await request.json()
        except Exception as exc:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail=f"invalid JSON body: {exc}")

        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail="messages must be a non-empty list")

        messages_dict = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = m.get("role", "user")
            content = m.get("content", "")
            if content is None:
                content = ""
            if not isinstance(content, str):
                # Content may be a list of blocks (text/image) in OpenAI spec;
                # this server is text-only, so concatenate any text blocks.
                if isinstance(content, list):
                    content = "".join(
                        b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                else:
                    content = str(content)

            # Preserve OpenAI tool-round fields so the chat template can
            # render a full tool-call history back to the model.
            msg: dict = {"role": role, "content": content}
            if role == "assistant" and isinstance(m.get("tool_calls"), list):
                msg["tool_calls"] = m["tool_calls"]
            if role == "tool" and isinstance(m.get("tool_call_id"), str):
                msg["tool_call_id"] = m["tool_call_id"]
            messages_dict.append(msg)

        max_tokens = int(body.get("max_tokens", max_new_tokens))
        temperature = float(body.get("temperature", 0.0))
        top_p = float(body.get("top_p", 1.0))
        stop = body.get("stop")
        if isinstance(stop, str):
            stop = [stop]

        tools = body.get("tools") if isinstance(body.get("tools"), list) else None
        if tools is not None and len(tools) == 0:
            tools = None

        # Diagnostic: log whether the request carried a tools array.  This is
        # the single biggest signal for debugging tool-calling agents — if the
        # agent never sends tools, the LLM has no way to know a tool exists.
        if tools:
            tool_names = []
            for t in tools:
                fn = t.get("function") if isinstance(t.get("function"), dict) else t
                n = fn.get("name") if isinstance(fn, dict) else None
                if isinstance(n, str):
                    tool_names.append(n)
            print(
                f"[llama_nemotron_llm_server] request with tools={tool_names}",
                flush=True,
            )
        else:
            print(
                "[llama_nemotron_llm_server] request with no tools",
                flush=True,
            )

        loop = asyncio.get_running_loop()
        answer = await loop.run_in_executor(
            None,
            backend.infer,
            messages_dict,
            max_tokens,
            temperature,
            top_p,
            stop,
            tools,
        )

        # Parse tool calls out of the assistant text only when the caller
        # supplied a tool schema.  This preserves the previous response shape
        # byte-identically for tool-less requests.  ``tool_names`` (already
        # collected above for logging) is passed through so the parser can
        # recover Python-style plain-text calls that LMFE's free-text branch
        # syntactically allows.
        message: dict
        finish_reason: str
        if tools:
            content_text, tool_calls = parse_llama_tool_calls(
                answer, set(tool_names),
            )
            if tool_calls:
                message = {
                    "role":       "assistant",
                    "content":    content_text,
                    "tool_calls": tool_calls,
                }
                finish_reason = "tool_calls"
            else:
                message = {"role": "assistant", "content": answer}
                finish_reason = "stop"
        else:
            message = {"role": "assistant", "content": answer}
            finish_reason = "stop"

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    return app, backend


async def _run(cfg: dict, yaml_dir: Path) -> None:
    import uvicorn

    if not cfg.get("model"):
        print("[llama_nemotron_llm_server] 'model' is required in config", file=sys.stderr)
        sys.exit(1)

    model_cache = _resolve_model_cache(cfg, yaml_dir)

    if hf_token := (cfg.get("hf_token") or os.environ.get("HF_TOKEN", "")):
        os.environ["HF_TOKEN"] = hf_token
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    port = int(cfg.get("port", _DEFAULT_PORT))
    host = cfg.get("host", "0.0.0.0")

    app, backend = _build_app(cfg, model_cache)

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, backend._ensure_loaded)

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    print(f"[llama_nemotron_llm_server] Ready  →  http://localhost:{port}/v1")
    await server.serve()
    print("[llama_nemotron_llm_server] Stopped.")


def run() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", type=Path, default=None)
    ns, _ = p.parse_known_args()

    cfg: dict = {}
    yaml_dir = Path.cwd()
    if ns.config and ns.config.exists():
        yaml_dir = ns.config.parent.resolve()
        with open(ns.config) as f:
            cfg = yaml.safe_load(f) or {}

    asyncio.run(_run(cfg, yaml_dir))


if __name__ == "__main__":
    run()
