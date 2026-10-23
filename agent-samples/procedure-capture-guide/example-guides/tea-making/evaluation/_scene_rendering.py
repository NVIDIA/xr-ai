# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render synthetic image fixtures shared by the visual and guide evaluators.

Draw label cards, colored rectangles, packets, temperature displays, and vessel
states from scene descriptions. Expected answers belong to the case manifest,
never to the renderer.
"""

from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont


def load_scenes(manifest_path: Path, manifest: dict) -> dict:
    scenes = {}
    if "scenes_file" in manifest:
        scenes.update(yaml.safe_load((manifest_path.parent / manifest["scenes_file"]).read_text()))
    scenes.update(manifest.get("scenes", {}))
    return scenes


def render_scene(scene: dict) -> Image.Image:
    if "kind" not in scene:
        image = Image.new("RGB", (640, 480), "white")
        draw = ImageDraw.Draw(image)
        if "rectangle" in scene:
            draw.rectangle((120, 100, 520, 380), fill=scene["rectangle"])
        if "text" in scene:
            draw.multiline_text((40, 40), scene["text"], fill="black", font_size=42, spacing=20)
        return image
    image = Image.new("RGB", (800, 600), "#eeeae1")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 500, 800, 600), fill="#c9b79c")

    def text(x, y, words, size=32, color="#19222c"):
        # Pillow's bundled font avoids dependence on a workstation font path.
        draw.multiline_text((x, y), words, font=ImageFont.load_default(size=size), fill=color, spacing=12)

    def bag(x, y):
        draw.rectangle((x, y, x + 80, y + 100), fill="#b99c6b", outline="#665433", width=3)
        for offset in range(10, 80, 10):
            draw.line((x + offset, y + 4, x + offset, y + 96), fill="#90774c")
        draw.line((x + 40, y, 530, 130, 585, 190), fill="#504639", width=3)
        draw.rectangle((570, 185, 620, 220), fill="#ece0b8", outline="#504639", width=2)

    kind = scene["kind"]
    if kind == "packet":
        draw.rectangle((170, 65, 620, 515), fill=scene.get("color", "#e5d191"), outline="#4d493f", width=5)
        draw.rectangle((190, 85, 600, 495), outline="#7e795c", width=2)
        if not scene.get("covered", False):
            text(220, 135, "TEA", 52)
            text(220, 265, scene["blend"].replace(" ", "\n", 1), 38)
        else:
            draw.rectangle((210, 125, 590, 425), fill="#bbb9b2")
    elif kind == "kettle":
        draw.arc((540, 160, 740, 420), 260, 100, fill="#343c45", width=28)
        draw.polygon(((215, 180), (105, 150), (150, 260), (225, 290)), fill="#b8c4cb", outline="#343c45")
        draw.rounded_rectangle((205, 125, 585, 480), radius=65, fill="#d9e3e8", outline="#343c45", width=6)
        draw.rounded_rectangle((240, 345, 550, 445), radius=20, fill="#a7cad6")
        draw.ellipse((245, 335, 545, 375), fill="#c1e2eb", outline="#6691a0", width=2)
        draw.rounded_rectangle((260, 105, 530, 150), radius=15, fill="#49505a")
        readings = scene["readings"]
        draw.rounded_rectangle((245, 190, 555, 330 if len(readings) == 1 else 395), radius=12,
                               fill="#182b2b", outline="#111e1e", width=3)
        for index, reading in enumerate(readings):
            top = 200 + index * 95
            text(265, top, reading["label"], 23, "#d5f5d6")
            text(265, top + 30, reading["value"], 43, "#d5f5d6")
        draw.rounded_rectangle((195, 475, 600, 515), radius=15, fill="#343c45")
    elif kind == "cup":
        state = scene["state"]
        if state not in {"empty", "dry_bag", "water", "above", "immersed", "floating", "obscured"}:
            raise ValueError(f"unknown cup state: {state}")
        draw.arc((520, 210, 690, 430), 260, 100, fill="#657982", width=12)
        draw.rounded_rectangle((230, 180, 560, 485), radius=30, fill="#f6f7f4", outline="#657982", width=6)
        if state not in {"empty", "dry_bag"}:
            draw.rectangle((238, 300, 552, 450), fill="#b5dbe7")
            draw.ellipse((238, 420, 552, 480), fill="#b5dbe7")
            draw.ellipse((238, 270, 552, 330), fill="#ceeaf1", outline="#598da1", width=3)
        if state in {"dry_bag", "immersed"}:
            bag(350, 365)
        elif state == "above":
            bag(340, 25)
        elif state == "floating":
            bag(350, 280)
        draw.ellipse((230, 155, 560, 215), outline="#657982", width=5)
        if state == "obscured":
            draw.rounded_rectangle((220, 220, 575, 490), radius=15, fill="#aa9d8c")
    else:
        raise ValueError(f"unknown scene kind: {kind}")
    return image
