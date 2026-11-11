# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

"""Inline SVG charts. Colors are CSS variables (``--s1``..``--s4``, ``--ink``, ``--ink2``, ``--grid``) so the page
themes them; hover text lives in ``data-tip`` attributes read by the page's tooltip script."""

from __future__ import annotations

import html
import math
from dataclasses import dataclass

W, H = 640, 300
ML, MR, MT, MB = 64, 18, 16, 46


@dataclass
class Series:
    """One line or bar series: ``points`` are (x, p50, p90 or None)."""

    name: str
    points: list[tuple[float | str, float, float | None]]
    slot: int = 1
    dashed: bool = False


def _fmt(v: float) -> str:
    if v >= 10_000:
        return f"{v / 1000:.0f}s"
    if v >= 1_000:
        return f"{v / 1000:.2f}".rstrip("0").rstrip(".") + "s"
    return f"{v:.0f}"


def _fmtx(v: float) -> str:
    """X-axis counts (tokens, bytes, characters): 1500 -> 1.5k."""
    if v >= 1_000:
        return f"{v / 1000:g}k"
    return f"{v:g}"


def _nice_max(v: float) -> float:
    if v <= 0:
        return 1.0
    mag = 10 ** math.floor(math.log10(v))
    for m in (1, 2, 2.5, 5, 10):
        if v <= m * mag:
            return m * mag
    return 10 * mag


def _legend(series: list[Series]) -> str:
    if len(series) < 2:
        return ""
    items = "".join(
        f'<span class="lg"><i class="sw s{s.slot}"></i>{html.escape(s.name)}</span>' for s in series)
    return f'<div class="legend">{items}</div>'


def _frame(title: str, sub: str, body: str, series: list[Series], table: str = "") -> str:
    return (f'<figure class="chart"><figcaption><b>{html.escape(title)}</b>'
            f'<span>{html.escape(sub)}</span></figcaption>{_legend(series)}'
            f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">{body}</svg>{table}</figure>')


def _axes(xmax_label: str, ymax: float, ylabel: str, xlabel: str) -> str:
    out = []
    for i in range(5):
        v = ymax * i / 4
        y = MT + (H - MT - MB) * (1 - i / 4)
        out.append(f'<line class="grid" x1="{ML}" x2="{W - MR}" y1="{y:.1f}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{ML - 8}" y="{y + 4:.1f}" text-anchor="end">{_fmt(v)}</text>')
    out.append(f'<text class="axl" x="14" y="{(H - MB + MT) / 2}" transform="rotate(-90 14 {(H - MB + MT) / 2})" '
               f'text-anchor="middle">{html.escape(ylabel)}</text>')
    out.append(f'<text class="axl" x="{(ML + W - MR) / 2}" y="{H - 6}" text-anchor="middle">{html.escape(xlabel)}</text>')
    return "".join(out)


def line_chart(title: str, sub: str, series: list[Series], xlabel: str, ylabel: str = "ms", *,
               log_x: bool = False, unit: str = "ms") -> str:
    """Lines through p50 with the p90 as a thin dashed companion; per-point hover."""
    xs = sorted({float(p[0]) for s in series for p in s.points})
    ymax = _nice_max(max((p[2] if p[2] is not None else p[1]) for s in series for p in s.points) * 1.05)
    lo, hi = (math.log10(max(xs[0], 1)), math.log10(xs[-1])) if log_x else (xs[0], xs[-1])

    def px(x: float) -> float:
        t = ((math.log10(max(x, 1)) if log_x else x) - lo) / ((hi - lo) or 1)
        return ML + (W - ML - MR) * t

    def py(y: float) -> float:
        return MT + (H - MT - MB) * (1 - y / ymax)

    body = [_axes("", ymax, ylabel, xlabel)]
    last_label_x = -1e9
    for x in xs:
        if px(x) - last_label_x < 42:
            continue
        last_label_x = px(x)
        body.append(f'<text class="tick" x="{px(x):.1f}" y="{H - MB + 16}" text-anchor="middle">{_fmtx(x)}</text>')
    for si, s in enumerate(series):
        pts = sorted(s.points, key=lambda p: float(p[0]))
        d50 = " ".join(f"{px(float(x)):.1f},{py(y):.1f}" for x, y, _ in pts)
        body.append(f'<polyline class="ln s{s.slot}" points="{d50}"/>')
        if all(p[2] is not None for p in pts):
            d90 = " ".join(f"{px(float(x)):.1f},{py(y90):.1f}" for x, _, y90 in pts)  # type: ignore[arg-type]
            body.append(f'<polyline class="ln p90 s{s.slot}" points="{d90}"/>')
        for x, y, y90 in pts:
            tip = f"{s.name} @ {_fmtx(float(x))}: p50 {y:.0f}{unit}" + (
                f", p90 {y90:.0f}{unit}" if y90 is not None else "")
            body.append(f'<circle class="pt s{s.slot}" cx="{px(float(x)):.1f}" cy="{py(y):.1f}" r="3.5" data-tip="{html.escape(tip)}"/>')
        lx, ly, _ = pts[-1]
        body.append(f'<text class="dl" x="{px(float(lx)) - 4:.1f}" y="{py(ly) - 8 - 12 * (len(series) - 1 - si):.1f}" text-anchor="end">{html.escape(s.name)}</text>')
    return _frame(title, sub, "".join(body), series)


def bar_chart(title: str, sub: str, categories: list[str], series: list[Series], ylabel: str = "ms",
              xlabel: str = "", *, unit: str = "ms") -> str:
    """Grouped vertical bars (p50) with a p90 tick; one group per category."""
    ymax = _nice_max(max((p[2] if p[2] is not None else p[1]) for s in series for p in s.points) * 1.05)
    gw = (W - ML - MR) / len(categories)
    bw = min(gw * 0.7 / len(series), 34)
    body = [_axes("", ymax, ylabel, xlabel)]
    base = H - MB
    for ci, cat in enumerate(categories):
        gx = ML + gw * ci + gw / 2
        body.append(f'<text class="tick" x="{gx:.1f}" y="{H - MB + 16}" text-anchor="middle">{html.escape(cat)}</text>')
        for si, s in enumerate(series):
            match = next((p for p in s.points if str(p[0]) == cat), None)
            if match is None:
                continue
            _, p50, p90 = match
            x = gx - bw * len(series) / 2 + bw * si + 1
            h = (H - MT - MB) * p50 / ymax
            tip = f"{s.name} / {cat}: p50 {p50:.0f}{unit}" + (f", p90 {p90:.0f}{unit}" if p90 is not None else "")
            body.append(f'<path class="bar s{s.slot}" d="M{x:.1f},{base} v{-max(h - 4, 0):.1f} q0,-4 4,-4 h{bw - 10:.1f} q4,0 4,4 v{max(h - 4, 0):.1f} z" data-tip="{html.escape(tip)}"/>')
            if p90 is not None:
                y90 = base - (H - MT - MB) * p90 / ymax
                body.append(f'<line class="whisk" x1="{x + bw / 2 - 1:.1f}" x2="{x + bw / 2 - 1:.1f}" y1="{base - h:.1f}" y2="{y90:.1f}"/>')
                body.append(f'<line class="whisk" x1="{x + bw / 2 - 4:.1f}" x2="{x + bw / 2 + 2:.1f}" y1="{y90:.1f}" y2="{y90:.1f}"/>')
            body.append(f'<text class="dl" x="{x + bw / 2 - 1:.1f}" y="{base - h - (6 if p90 is None else (base - h - y90 + 8)):.1f}" text-anchor="middle">{_fmt(p50)}</text>')
    return _frame(title, sub, "".join(body), series)


def hbar_stack(title: str, sub: str, rows: list[tuple[str, list[tuple[str, float, int]], float | None]],
               unit: str = "ms") -> str:
    """Horizontal stacked bars: each row is (label, [(segment name, ms, slot)], p90 total marker)."""
    left, row_h = 250, 26
    height = 30 + row_h * len(rows)
    xmax = _nice_max(max(max(sum(v for _, v, _ in segs), p90 or 0) for _, segs, p90 in rows) * 1.08)
    width = W + 120
    body = [f'<text class="tick" x="{left}" y="14">0</text>']
    for i in range(1, 5):
        x = left + (width - left - 20) * i / 4
        body.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="20" y2="{height - 6}"/>')
        body.append(f'<text class="tick" x="{x:.1f}" y="14" text-anchor="middle">{_fmt(xmax * i / 4)}</text>')
    legend_names: dict[str, int] = {}
    for ri, (label, segs, p90) in enumerate(rows):
        y = 26 + ri * row_h
        body.append(f'<text class="rowl" x="{left - 8}" y="{y + 13}" text-anchor="end">{html.escape(label)}</text>')
        x = float(left)
        total = 0.0
        for name, v, slot in segs:
            w = (width - left - 20) * v / xmax
            if slot == 0:  # invisible offset so a row can start later than zero (waterfall)
                x += w
                total += v
                continue
            legend_names[name] = slot
            if w > 0:
                tip = f"{label}: {name} {v:.0f}{unit}"
                body.append(f'<rect class="seg s{slot}" x="{x:.1f}" y="{y}" width="{max(w - 2, 0.5):.1f}" height="16" rx="3" data-tip="{html.escape(tip)}"/>')
            x += w
            total += v
        body.append(f'<text class="dl" x="{x + 6:.1f}" y="{y + 13}">{_fmt(total)}</text>')
        if p90 is not None:
            px90 = left + (width - left - 20) * p90 / xmax
            body.append(f'<line class="whisk" x1="{px90:.1f}" x2="{px90:.1f}" y1="{y - 1}" y2="{y + 17}" data-tip="{html.escape(f"{label}: p90 {p90:.0f}{unit}")}"/>')
    legend = "".join(f'<span class="lg"><i class="sw s{s}"></i>{html.escape(n)}</span>' for n, s in legend_names.items())
    return (f'<figure class="chart wide"><figcaption><b>{html.escape(title)}</b><span>{html.escape(sub)}</span></figcaption>'
            f'<div class="legend">{legend}<span class="lg"><i class="sw whiskl"></i>p90 (tick)</span></div>'
            f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">{"".join(body)}</svg></figure>')


def table(headers: list[str], rows: list[list[str]], caption: str = "") -> str:
    """Plain HTML table (the accessible companion to a chart)."""
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    cap = f"<caption>{html.escape(caption)}</caption>" if caption else ""
    return f'<table class="tbl">{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'
