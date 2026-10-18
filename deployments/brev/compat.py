# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility with pinned XR AI releases. Current runtimes use public deployment hooks."""

import ast
import ipaddress
import json
import re
import sys
import tomllib
from pathlib import Path


def model_project(root: Path, project: str = "") -> Path:
    candidates = [root / project / "model-servers"] if project else []
    candidates += [root / "model-server-samples/model-servers", root / "agent-samples/model-servers"]
    for path in candidates:
        if (path / "pyproject.toml").is_file():
            return path
    raise ValueError("no model-server project found")


def command(project: Path) -> str:
    scripts = tomllib.loads((project / "pyproject.toml").read_text())["project"]["scripts"]
    if len(scripts) != 1:
        raise ValueError(f"{project}: expected exactly one console command")
    name = next(iter(scripts))
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
        raise ValueError(f"{project}: invalid console command")
    return name


def structured_readiness(root: Path) -> bool:
    return (root / "utils/xr-ai-launcher/xr_ai_launcher/deployment.py").is_file()


def recording_flag(project: Path) -> str:
    # Read CLI declarations without importing samples or installing their dependencies.
    entry = project / "main.py"
    if not entry.is_file():
        return ""
    flags = {
        arg.value for node in ast.walk(ast.parse(entry.read_text()))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
        for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
    }
    return next((flag for flag in ("--session-recording", "--capture") if flag in flags), "")


def network_overrides(host: str, ports: list[str]) -> dict:
    external = ipaddress.ip_address(host).is_global
    return dict(
        zip(
            ("lk_port_ws", "lk_port_tcp", "lk_port_udp", "web_server_port", "token_server_port"),
            map(int, ports),
        )
    ) | {
        "lk_use_external_ip": external,
        "lk_skip_external_ip_validation": external,
        "web_server_extra_sans": [host],
        "cert_file": "",
        "key_file": "",
    }


def configure(root: Path, project: str, host: str, ports: list[str]) -> dict:
    overrides = network_overrides(host, ports)
    if (root / "services/device-io-hub/device_io_hub/deployment.py").is_file():
        return overrides
    overrides["lk_internal_url"] = f"ws://127.0.0.1:{overrides['lk_port_ws']}"
    # Old releases have no override hook. Limit rewriting to their flat hub YAML;
    # fail explicitly if a layout changes rather than appending ineffective keys.
    yaml = root / project / "yaml"
    hub = next((yaml / name for name in ("device_io_hub.yaml", "xr_media_hub.yaml") if (yaml / name).is_file()), None)
    if hub is None:
        raise ValueError(f"{yaml}: no supported hub configuration")
    text = hub.read_text()
    for name, value in overrides.items():
        if name == "web_server_extra_sans":
            pattern = r"^(?:#\s*)?web_server_extra_sans\s*:.*(?:\n(?:#\s*)?[ \t]+-.*)*"
            replacement = f"web_server_extra_sans:\n  - {host}"
        else:
            pattern = rf"^(?:#\s*)?{name}\s*:.*$"
            replacement = f"{name}: {json.dumps(value)}"
        if re.search(rf"^[ \t]+{name}\s*:", text, re.MULTILINE):
            raise ValueError(f"{hub}: nested {name} is unsupported by the release adapter")
        if re.search(pattern, text, re.MULTILINE):
            text = re.sub(pattern, lambda _: replacement, text, count=1, flags=re.MULTILINE)
        else:
            text += f"\n{replacement}\n"
    hub.write_text(text)
    return overrides


def main(args: list[str]) -> None:
    action, *values = args
    if action == "models":
        print(model_project(Path(values[0]), values[1]))
    elif action == "command":
        print(command(Path(values[0])))
    elif action == "readiness":
        print("structured" if structured_readiness(Path(values[0])) else "legacy")
    elif action == "configure":
        root, project, host, *ports = values
        print(json.dumps(configure(Path(root), project, host, ports)))
    else:
        raise ValueError(f"unknown compatibility action: {action}")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc
