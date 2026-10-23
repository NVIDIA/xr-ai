# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private to xr_ai_model_server_stack: compose configs and delegate lifecycle."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import yaml
from xr_ai_launcher import Process, detect_gpu_config, require_credentials, run_stack
from xr_ai_logging import setup_logging
from xr_ai_vllm import stop_persistent_servers

from ._reconcile import model_state

_SERVICES_DIR = Path(__file__).resolve().parents[3] / "services"
# The explicit order leaves Omni's compilation ahead of Cosmos initialization.
_SERVICES = {
    "speaker-stt": ("speaker-stt", "speaker_stt", "speaker_stt"),
    "stt": ("stt-server", "stt_server", "stt_server"),
    "tts": ("pocket-tts", "pocket_tts_server", "pocket_tts_server"),
    "omni": ("nemotron-omni-llm", "nemotron_omni_llm_server", "nemotron_omni_llm_server"),
    "vlm": ("vlm-server", "vlm_server", "vlm_server"),
    "clef": ("clef-server", "clef_server", "clef_server"),
    "embedding": ("embedding-server", "embedding_server", "embedding_server"),
}


def _read_yaml(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"{path}: could not read YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return data


def _configs(base: Path, services: tuple[str, ...], profile: str | None) -> dict[str, dict]:
    overrides_path = base / "yaml" / f"{profile}.yaml"
    overrides = {}
    if profile is not None:
        profiles = {path.stem for path in (base / "yaml").glob("*.yaml")}
        if profile not in profiles:
            raise ValueError(f"unknown GPU profile {profile!r}; available: {', '.join(sorted(profiles))}")
        overrides = _read_yaml(overrides_path)
        if set(overrides) != set(services):
            raise ValueError(f"{overrides_path}: must contain exactly these servers: {', '.join(services)}")
    configs = {}
    for service in services:
        if service not in _SERVICES:
            raise ValueError(f"unknown model service {service!r}")
        project, _, basename = _SERVICES[service]
        common_path = _SERVICES_DIR / project / f"{basename}.yaml"
        config = _read_yaml(common_path)
        override = overrides.get(service, {})
        if not isinstance(override, dict):
            raise ValueError(f"{overrides_path}: {service} must be a mapping")
        config.update(override)
        # Servers resolve relative cache paths against the supplied YAML. Resolve
        # before materializing in a temporary directory; overrides belong to the sample.
        cache_source = overrides_path if "model_cache" in override else common_path
        cache_value = config.get("model_cache")
        if not isinstance(cache_value, str) or not cache_value:
            raise ValueError(f"{service}: model_cache must be a non-empty path")
        cache = Path(cache_value)
        config["model_cache"] = str((cache_source.parent / cache).resolve())
        port = config.get("port")
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError(f"{service}: port must be an integer between 1 and 65535")
        configs[service] = config
    if len({config["port"] for config in configs.values()}) != len(configs):
        raise ValueError("model services must use distinct ports")
    return configs


def _processes(configs: dict[str, dict], directory: Path, root: Path) -> list[Process]:
    processes = []
    for service, (project, command, basename) in _SERVICES.items():
        if service not in configs:
            continue
        config = configs[service]
        path = directory / f"{basename}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        path.chmod(0o600)
        processes.append(Process(
            service, str(root / "services" / project), command,
            config=path, launch_mode="persist", port=config["port"],
        ))
    return processes


def run_model_stack(
    base: Path,
    services: tuple[str, ...],
    *,
    stop: bool = False,
    gpu_profile: str | None = None,
    state_file: Path | None = None,
) -> None:
    """Start or stop the configured local inference stack for a developer sample.

    ``base`` is the sample's model-server project directory, conventionally
    ``<repository>/agent-samples/<sample>/model-servers`` or
    ``<repository>/apps/<application>/model-servers``. Relative paths are
    resolved against the current working directory. Service projects are
    resolved under that repository's ``services/`` directory.

    ``services`` selects from ``speaker-stt``, ``stt``, ``tts``, ``omni``, ``vlm``,
    ``clef``, and ``embedding``. The selected ``base/yaml/<gpu_profile>.yaml`` must contain
    exactly those server mappings. Each mapping overrides the corresponding
    service's own YAML in ``services/``. Launch order is speaker ASR, batch STT, TTS,
    Omni, VLM, Clef, then embedding, omitting unselected services.

    ``gpu_profile`` selects a checked-in profile by filename stem; ``None``
    detects the hardware automatically for startup. ``stop=False`` requires ``HF_TOKEN``,
    waits for every server to become ready, and returns with servers running.
    ``stop=True`` needs neither GPU detection nor download credentials. With no
    explicit profile it uses the selected services' shared default ports; pass
    ``gpu_profile`` to honor profile-specific port overrides. It stops only recognized
    servers on those selected ports. It can stop servers used
    by another sample sharing those ports; no launch history is recorded.

    ``state_file`` opts into serialized model reconciliation for one deployment.
    Share the same file between sample model launchers. Its parent must be an
    owned private directory. Only ready services with matching resolved config,
    checkout revision and path, HF token, GPU visibility, and live instance
    identity are reused. Unused, changed, or unhealthy recorded models stop
    before new models start. Credentials are hashed, never saved in state.
    Untracked listeners are not adopted; stop them explicitly first. A partial
    launch must be stopped with the same state file before retrying. With
    ``stop=True``, recorded ports are used without GPU detection or credentials.
    Callers must stop the consuming agent before changing its model stack.
    Reconciliation supports native services and Docker-backed vLLM. Persistent
    pip vLLM can escape cancellation before binding its port, so it must use
    the explicit shutdown workflow without ``state_file``.

    Without ``state_file``, stop the current stack before switching samples,
    editing its server settings, or updating the checkout. This API does not read
    consumer endpoint profiles or launch independently deployed NIM endpoints.

    Raises:
        ValueError: The selected profile or server configuration is invalid.
        RuntimeError: GPU inventory, private deployment state, or server cleanup fails.
        SystemExit: Required credentials are absent or stack startup aborts.
    """
    if state_file is not None and not isinstance(state_file, Path):
        raise TypeError("state_file must be a Path or None")
    base = base.resolve()
    log_file = setup_logging("orchestrator", namespace=f"{base.parent.name}-models")
    if state_file is not None:
        if not stop:
            configs = _configs(base, services, gpu_profile or detect_gpu_config())
            if any(config.get("vllm_backend") == "pip" for config in configs.values()):
                raise ValueError("state-file reconciliation requires Docker for vLLM; use explicit shutdown for pip")
            require_credentials("HF_TOKEN")
        with model_state(state_file) as state:
            if stop:
                state.stop()
            else:
                state.prepare(configs, base.parents[2])
                _launch(base, configs, log_file)
                state.ready()
        return
    if stop:
        configs = _configs(base, services, gpu_profile)
        if not stop_persistent_servers([(name, config["port"]) for name, config in configs.items()]):
            raise RuntimeError("one or more configured model servers could not be stopped")
        return
    configs = _configs(base, services, gpu_profile or detect_gpu_config())
    require_credentials("HF_TOKEN")
    _launch(base, configs, log_file)


def _launch(base: Path, configs: dict, log_file: Path) -> None:
    # Configs stay available through child initialization and readiness. Services
    # read their config at startup; persistent servers do not reopen it afterward.
    directory = tempfile.mkdtemp(prefix="model-configs-", dir=log_file.parent)
    try:
        processes = _processes(configs, Path(directory), base.parents[2])
        run_stack(processes, base, exit_after_ready=True)
    except BaseException:
        # Keep failure-summary Config paths usable, without retaining inline HF credentials.
        for service, config in configs.items():
            path = Path(directory) / f"{_SERVICES[service][2]}.yaml"
            if path.exists() and config.get("hf_token"):
                path.write_text(yaml.safe_dump({**config, "hf_token": ""}, sort_keys=False), encoding="utf-8")
        raise
    else:
        shutil.rmtree(directory)
