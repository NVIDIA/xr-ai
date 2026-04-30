# mistral-minitron-llm-server

OpenAI-compatible LLM HTTP server using HuggingFace transformers.

Default model: `nvidia/Mistral-NeMo-Minitron-8B-Instruct` (~16 GB at BF16).

## Quickstart

```bash
cd ai-services/llm/mistral_minitron
uv sync
uv run mistral_minitron_llm_server --config mistral_minitron_llm_server.yaml
```

First run downloads weights (~16 GB) to the shared `models/` cache at the repo root.

## Endpoints

| Endpoint                  | Method | Description                        |
|---------------------------|--------|------------------------------------|
| `/health`                 | GET    | Health check (`{"status": "ok"}`)  |
| `/v1/models`              | GET    | List available models              |
| `/v1/chat/completions`    | POST   | Chat completion (OpenAI-compatible)|

## Config keys (`mistral_minitron_llm_server.yaml`)

| Key             | Type       | Default                                  | Description                                      |
|-----------------|------------|------------------------------------------|--------------------------------------------------|
| `model`         | str        | (required)                               | HuggingFace model ID                             |
| `port`          | int        | `8101`                                   | HTTP port                                        |
| `host`          | str        | `0.0.0.0`                                | Bind address                                     |
| `hf_token`      | str        | `""`                                     | HuggingFace token (for gated models)             |
| `system_prompt` | str        | `""`                                     | Optional system prompt prepended to all requests |
| `max_new_tokens`| int        | `1024`                                   | Max tokens to generate                           |
| `model_cache`   | str        | `../../../models`                        | Weight cache path (relative to YAML)             |
| `dtype`         | str        | `bfloat16`                               | Torch dtype (`bfloat16`, `float16`, `float32`)   |
| `stop`          | list[str]  | `["<extra_id_1>", "<extra_id_0>"]`       | Default stop sequences (if request omits stop)   |

## Example curl

```bash
# Health check
curl http://localhost:8101/health

# Chat completion
curl -X POST http://localhost:8101/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llm",
    "messages": [{"role": "user", "content": "Say OK"}],
    "max_tokens": 16
  }'
```

## Swap models

Edit `mistral_minitron_llm_server.yaml`:

```yaml
model: nvidia/Nemotron-H-4B-Instruct-128K
```

Any HuggingFace `AutoModelForCausalLM`-compatible model works. Adjust
`max_new_tokens`, `dtype`, and `stop` as needed for the new model.

For a tool-calling-capable alternative with a per-turn reasoning toggle, see
the sibling [../llama_nemotron/](../llama_nemotron/) (`nvidia/Llama-3.1-Nemotron-Nano-8B-v1`).

## Notes

- **No continuous batching.** FastAPI + raw transformers is simpler but slower
  under load than vLLM. For single-user voice agents, this is fine.
- **First-run download** goes to the shared `models/` cache.
- **Mistral-NeMo-Minitron-8B's chat template** emits `<extra_id_1>` as turn
  boundaries; the default `stop` list handles this so replies don't leak
  template tokens.
