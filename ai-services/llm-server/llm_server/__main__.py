"""
llm_server — OpenAI-compatible LLM HTTP server.

Loads nvidia/Mistral-NeMo-Minitron-8B-Instruct (or any HuggingFace causal LM)
in-process and serves an OpenAI-compatible API:

    POST /v1/chat/completions   (pure-text messages)
    GET  /v1/models

Accepts --config <path>.yaml (auto-passed by xr-ai-launcher).

Config keys
-----------
    model:          str    HuggingFace model ID (required)
    port:           int    HTTP port (default: 8101)
    host:           str    Bind address (default: "0.0.0.0")
    hf_token:       str    HuggingFace token for gated models
    system_prompt:  str    Optional system prompt prepended to every request
    model_cache:    str    HuggingFace weight cache path.  Resolved relative to
                           this YAML file's directory.  Default: ../models
    max_new_tokens: int    Max tokens to generate (default: 1024)
    dtype:          str    Torch dtype (default: bfloat16)
    stop:           list   Default stop sequences applied if request omits stop
"""
import argparse
import asyncio
import os
import sys
import threading
import uuid
from pathlib import Path

import yaml

_DEFAULT_PORT = 8101
_DEFAULT_TOKENS = 1024
_DEFAULT_STOP = ["<extra_id_1>", "<extra_id_0>"]


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

            print(f"[llm_server] Loading {self._model_id}  cache={cache}  dtype={self._dtype_str}")
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
                print(f"[llm_server] Downloading {self._model_id}…")
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
            print(f"[llm_server] Model ready on {dev}")

    def infer(
        self,
        messages: list[dict],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None,
    ) -> str:
        """Synchronous inference. Call from a thread pool."""
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        self._ensure_loaded()

        full_messages = list(messages)
        if self._system_prompt:
            full_messages = [{"role": "system", "content": self._system_prompt}] + full_messages

        prompt = self._tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=True,
        )
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
            messages_dict.append({"role": role, "content": content})

        max_tokens = int(body.get("max_tokens", max_new_tokens))
        temperature = float(body.get("temperature", 0.0))
        top_p = float(body.get("top_p", 1.0))
        stop = body.get("stop")
        if isinstance(stop, str):
            stop = [stop]

        loop = asyncio.get_running_loop()
        answer = await loop.run_in_executor(
            None,
            backend.infer,
            messages_dict,
            max_tokens,
            temperature,
            top_p,
            stop,
        )

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    return app, backend


async def _run(cfg: dict, yaml_dir: Path) -> None:
    import uvicorn

    if not cfg.get("model"):
        print("[llm_server] 'model' is required in config", file=sys.stderr)
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

    print(f"[llm_server] Ready  →  http://localhost:{port}/v1")
    await server.serve()
    print("[llm_server] Stopped.")


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
