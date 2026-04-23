# Dependency Map

> **AGENTS: This file is mandatory to maintain.**
> Any change to a `pyproject.toml`, a YAML config/example, a documented
> interface, or an architectural decision **must** be reflected here in the
> same commit. A change is not complete until this file is up to date.

---

## Internal packages

```
xr-ai-agent  (agent-sdk/)
    └── pyzmq >=26.0
    └── msgpack >=1.0

xr-ai-launcher  (launcher/)
    └── (stdlib only — zero runtime deps)

xr-media-hub  (server-runtime/)
    └── xr-ai-agent  [editable: ../agent-sdk]
    └── pyzmq >=26.0
    └── livekit >=0.17
    └── livekit-api >=0.7
    └── fastapi >=0.111
    └── uvicorn[standard] >=0.29
    └── httpx >=0.27
    └── websockets >=12.0
    └── numpy >=1.24
    └── pyyaml >=6.0
    └── cryptography >=42.0

cloudxr-runtime  (cloudxr-runtime/)
    └── isaacteleop[cloudxr]
    └── pyyaml
```

---

## Agent samples

### echo-agent  (agent-samples/echo-agent/)

| Sub-project | Package | Internal deps | External deps |
|---|---|---|---|
| Orchestrator | `echo-agent` | `xr-ai-launcher` | — |
| Worker | `echo-agent-worker` | `xr-ai-agent` | — |

### vlm-agent  (agent-samples/vlm-agent/)

| Sub-project | Package | Internal deps | External deps |
|---|---|---|---|
| Orchestrator | `vlm-agent` | `xr-ai-launcher` | — |
| Worker | `vlm-agent-worker` | `xr-ai-agent` | numpy >=1.24, Pillow >=10.0, torch >=2.2, torchvision >=0.17, transformers >=4.49, accelerate >=0.30, qwen-vl-utils >=0.0.8, hf-transfer >=0.1.4, pyyaml >=6.0 |

### cloudxr-agent  (agent-samples/cloudxr-agent/)

| Sub-project | Package | Internal deps | External deps |
|---|---|---|---|
| Orchestrator | `cloudxr-agent` | `xr-ai-launcher` | — |
| Worker | `cloudxr-agent-worker` | `xr-ai-agent` | — |

---

## Change impact map

When you change something in the left column, **all items on the right must be
updated in the same commit**.

| Component changed | Must also update |
|---|---|
| `agent-sdk/` API or types | `AGENTS.md` worker boilerplate, any sample worker that uses the changed API |
| `server-runtime/` config fields (`LiveKitConnectorConfig`) | `server-runtime/xr_media_hub.yaml` (reference copy), each sample's `xr_media_hub.yaml`, `AGENTS.md` Config section |
| `launcher/` `Process` / `run_stack` API | `AGENTS.md` orchestrator boilerplate and process model section |
| vlm-agent model class or supported architectures | `agent-samples/vlm-agent/vlm_agent_worker.yaml.example` comments |
| vlm-agent YAML config keys (`model`, `hf_token`, `system_prompt`, …) | `agent-samples/vlm-agent/vlm_agent_worker.yaml.example`, worker `__main__.py` docstring |
| cloudxr-runtime YAML config keys | `agent-samples/cloudxr-agent/cloudxr_runtime.yaml`, `AGENTS.md` CloudXR section |
| Any `pyproject.toml` dependency | `DEPENDENCIES.md` (this file) |
| Any new sample added | `DEPENDENCIES.md`, `AGENTS.md`, `README.md` |
| Any new shared component added (peer of `server-runtime/`) | `AGENTS.md` Architecture section, `DEPENDENCIES.md` |

---

## Dependency rules (enforced)

- `launcher/` — zero runtime dependencies. Stdlib only.
- `agent-sdk/` — only `pyzmq` + `msgpack`. No server-side packages.
- Agent workers — `xr-ai-agent` + task-specific libs (numpy, torch, etc.).
  Must never import from `xr-media-hub` or `xr-ai-launcher`.
- New external deps require a note here explaining why they were added.
