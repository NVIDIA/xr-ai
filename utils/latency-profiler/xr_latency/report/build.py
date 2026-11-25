# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

"""Assemble ``report.html`` from a fitted ``model.json``: one self-contained page with inline SVG and a calculator."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .._machine import label as machine_label
from ..estimate import evaluate
from ..patterns import load_patterns, price
from ..validate import error_pct, first_audio_checks, independent, router_checks
from . import charts
from .charts import Series

HERE = Path(__file__).parent

CSS = """
:root{--bg:#fcfcfb;--panel:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--grid:#e6e5e1;--line:#d4d3ce;--accent:#2a78d6;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--good:#0a7c4a;--warn:#a86400;--bad:#b3261e;color-scheme:light}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){--bg:#1a1a19;--panel:#222221;--ink:#f0efec;--ink2:#c3c2b7;--grid:#34342f;--line:#43433d;--accent:#3987e5;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--good:#52c28a;--warn:#e0a13a;--bad:#ef7a73;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#1a1a19;--panel:#222221;--ink:#f0efec;--ink2:#c3c2b7;--grid:#34342f;--line:#43433d;--accent:#3987e5;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--good:#52c28a;--warn:#e0a13a;--bad:#ef7a73;color-scheme:dark}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1040px;margin:0 auto;padding:24px 16px 80px}
h1{font-size:28px;margin:8px 0 4px}h2{font-size:21px;margin:44px 0 6px;padding-top:14px;border-top:1px solid var(--line)}h3{font-size:16px;margin:24px 0 4px}
p,li{color:var(--ink)}.sub,.note{color:var(--ink2)}.note{font-size:13px}
nav{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 0;z-index:5;display:flex;gap:14px;flex-wrap:wrap;font-size:13px}
nav a{color:var(--ink2);text-decoration:none}nav a:hover{color:var(--ink)}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr));gap:18px}
figure.chart{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 12px 8px}figure.wide{grid-column:1/-1}
figcaption b{display:block}figcaption span{color:var(--ink2);font-size:13px}svg{width:100%;height:auto;display:block}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12.5px;color:var(--ink2);margin:6px 0 2px}.lg{display:inline-flex;align-items:center;gap:6px}
.sw{width:10px;height:10px;border-radius:3px;display:inline-block}.sw.whiskl{width:2px;height:12px;background:var(--ink2);border-radius:0}
.s0{--c:transparent}.s1{--c:var(--s1)}.s2{--c:var(--s2)}.s3{--c:var(--s3)}.s4{--c:var(--s4)}.sw{background:var(--c)}
.grid{stroke:var(--grid);stroke-width:1}.tick,.axl,.dl,.rowl{fill:var(--ink2);font-size:11px}.axl{font-size:11.5px}.dl{fill:var(--ink)}.rowl{font-size:12px;fill:var(--ink)}
.ln{fill:none;stroke:var(--c);stroke-width:2;stroke-linejoin:round}.ln.p90{stroke-width:1.2;stroke-dasharray:4 3;opacity:.8}
.pt{fill:var(--c);stroke:var(--panel);stroke-width:2;cursor:pointer}.bar,.seg{fill:var(--c)}.bar:hover,.seg:hover,.pt:hover{opacity:.8}.whisk{stroke:var(--ink2);stroke-width:1.4}
.tbl{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0}.tbl th,.tbl td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
.tbl th{color:var(--ink2);font-weight:600}.tbl caption{caption-side:top;text-align:left;color:var(--ink2);padding:4px 0}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.callout{border-left:3px solid var(--accent);background:var(--panel);padding:10px 14px;border-radius:0 8px 8px 0;margin:12px 0}
.callout.warn{border-color:var(--warn)}.callout.bad{border-color:var(--bad)}.callout b:first-child{display:block}
.pill{font-size:11px;padding:1px 7px;border-radius:99px;border:1px solid var(--line);color:var(--ink2);white-space:nowrap}
.pill.measured{color:var(--good);border-color:var(--good)}.pill.fitted{color:var(--accent);border-color:var(--accent)}.pill.code{color:var(--ink2);border-color:var(--ink2)}.pill.assumed{color:var(--warn);border-color:var(--warn)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:14px 0}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 12px}.kpi b{font-size:26px;display:block;line-height:1.1}.kpi span{font-size:12.5px;color:var(--ink2)}
.calc{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}.calc .row{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px 14px}
label{font-size:12.5px;color:var(--ink2);display:block}select,input{width:100%;font:inherit;padding:5px 7px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink)}
#tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--bg);font-size:12px;padding:4px 8px;border-radius:6px;display:none;max-width:320px;z-index:20}
button.theme{float:right;font:inherit;font-size:12px;border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:6px;padding:3px 9px;cursor:pointer}
code{background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:0 4px;font-size:13px}
@media(max-width:560px){h1{font-size:23px}}
"""

JS_TIP = """
const tip=document.getElementById('tip');
document.addEventListener('mousemove',e=>{const t=e.target.closest('[data-tip]');if(!t){tip.style.display='none';return}
tip.textContent=t.getAttribute('data-tip');tip.style.display='block';tip.style.left=Math.min(e.clientX+14,innerWidth-330)+'px';tip.style.top=(e.clientY+16)+'px'});
document.querySelector('button.theme').onclick=()=>{const r=document.documentElement,d=r.getAttribute('data-theme')==='dark'||(!r.getAttribute('data-theme')&&matchMedia('(prefers-color-scheme:dark)').matches);r.setAttribute('data-theme',d?'light':'dark')};
"""


def _ms(v: float | None) -> str:
    if v is None:
        return "–"
    return f"{v / 1000:.2f} s" if v >= 1000 else f"{v:.0f} ms"


def _row(table: list[dict[str, Any]], **keys: Any) -> dict[str, Any] | None:
    for r in table:
        if all(r.get(k) == v for k, v in keys.items()):
            return r
    return None


def _pts(table: list[dict[str, Any]], key: str, **where: Any) -> list[tuple[float | str, float, float | None]]:
    return [(r[key], r["p50"], r["p90"]) for r in table if all(r.get(k) == v for k, v in where.items())]


def _curve_series(name: str, curve: dict[str, Any] | None, slot: int) -> Series | None:
    if not curve:
        return None
    return Series(name, [(p["x"], p["p50"], p["p90"]) for p in curve["points"]], slot)


def _section(id_: str, title: str, intro: str, body: str) -> str:
    return f'<section id="{id_}"><h2>{html.escape(title)}</h2>{intro}{body}</section>'


def _safe(fn: Any, *args: Any, **kw: Any) -> str:
    try:
        return fn(*args, **kw)
    except (KeyError, IndexError, ValueError, TypeError, ZeroDivisionError) as exc:
        return f'<p class="note">Chart unavailable: {html.escape(type(exc).__name__)} {html.escape(str(exc))} (benchmark not run?)</p>'


# ---------------------------------------------------------------- components

def _llm(model: dict[str, Any]) -> str:
    llm = model["llm"]
    out = []
    cold, warm = _curve_series("cold prompt", llm["ttft_cold_vs_ctx"], 1), _curve_series("prefix cached", llm["ttft_warm_vs_ctx"], 3)
    out.append(charts.line_chart("Time to first token vs prompt size", "Omni LLM, thinking off, streamed. Solid p50, dashed p90.",
                                 [s for s in (cold, warm) if s], "prompt tokens", log_x=True))
    dec = _curve_series("decode (chat)", llm["decode_total_vs_tokens_chat"], 2)
    if dec:
        out.append(charts.line_chart("Total time vs output tokens", "Short prompt; slope is the decode cost per token.", [dec], "output tokens"))
    th = llm["thinking"]
    cats = ["off", "on"]
    first_vis = Series("first visible token", [("on" if r["thinking"] else "off", r["p50"], r["p90"]) for r in th["first_text"]], 1)
    first_reason = Series("first reasoning token", [("on" if r["thinking"] else "off", r["p50"], r["p90"]) for r in th["first_reasoning"]], 4)
    out.append(charts.bar_chart("Thinking: when the user first sees something", "2k-token prompt, trivial question", cats, [first_vis, first_reason]))
    cc = Series("TTFT", [(str(int(r["width"])), r["p50"], r["p90"]) for r in llm["concurrency_ttft"]], 1)
    ct = Series("total", [(str(int(r["width"])), r["p50"], r["p90"]) for r in llm["concurrency_total"]], 2)
    out.append(charts.bar_chart("Concurrent requests", "N simultaneous 2k-token requests (server max-num-seqs 8)", [str(int(r["width"])) for r in llm["concurrency_ttft"]], [cc, ct], xlabel="simultaneous requests"))
    con = llm.get("contention") or {}
    if con.get("first_text"):
        names = {"idle": "idle server", "tea_monitor": "tea-making monitor", "heavy_caption": "heavy caption loop"}
        cats = [names.get(r["load"], r["load"]) for r in con["first_text"]]
        first = Series("foreground TTFT", [(names.get(r["load"], r["load"]), r["p50"], r["p90"]) for r in con["first_text"]], 1)
        total = Series("foreground total", [(names.get(r["load"], r["load"]), r["p50"], r["p90"]) for r in con["total"]], 2)
        out.append(charts.bar_chart("A foreground turn while background work shares the server", "2k-token turn; background = image caption + judge call every 2 s on the same server", cats, [first, total]))
    return f'<div class="grid2">{"".join(out)}</div>'


def _decision(model: dict[str, Any]) -> str:
    d = model["decision"]
    series = [s for s in (_curve_series(f"{n} labels", d.get(f"total_vs_history_bytes_labels{n}"), i) for i, n in ((1, 2), (2, 4), (3, 8))) if s]
    parts = [charts.line_chart("Routing decision vs history it reads", "clef decision model; the router sends up to 12 KB of history", series, "history bytes in the prompt")]
    cats = sorted({int(r["width"]) for r in d["concurrency"]})
    rank_series = []
    for rank in range(max(r["rank"] for r in d["concurrency"]) + 1):
        pts = [(str(int(r["width"])), r["p50"], r["p90"]) for r in d["concurrency"] if r["rank"] == rank]
        if pts:
            rank_series.append(Series(f"finishes #{rank + 1}", pts, min(rank + 1, 4)))
    parts.append(charts.bar_chart("Concurrent decisions serialize", "N simultaneous decide calls: each finishes ~one call later than the last", [str(c) for c in cats], rank_series, xlabel="simultaneous decide calls"))
    return f'<div class="grid2">{"".join(parts)}</div>'


def _vlm(model: dict[str, Any]) -> str:
    vlm = model["vlm"]
    resolutions = sorted({r["res"] for r in vlm["chat_total"]}, key=lambda s: int(s.split("x")[0]) * int(s.split("x")[1]))
    series = [Series(f"{n} image{'s' if n > 1 else ''}", [(r["res"], r["p50"], r["p90"]) for r in vlm["chat_total"] if r["images"] == n], slot)
              for slot, n in ((1, 1), (2, 2), (3, 4))]
    parts = [charts.bar_chart("VLM answer time by image size and count", "Cosmos3-Nano, ~32 output tokens, fresh images (cache cold)", resolutions, series)]
    cached = [Series("cached repeat", [(r["res"], r["p50"], r["p90"]) for r in vlm["cached_total"] if r["images"] == 4], 3),
              Series("cold", [(r["res"], r["p50"], r["p90"]) for r in vlm["chat_total"] if r["images"] == 4], 1)]
    parts.append(charts.bar_chart("Asking twice about the same 4 images", "Prefix cache turns a repeat query cheap", resolutions, cached))
    fit = _curve_series("first text", vlm["first_text_vs_prompt_tokens"], 1)
    if fit:
        parts.append(charts.line_chart("Streamed first text vs image tokens", "Each point is one (resolution, count) cell", [fit], "prompt tokens (image tokens)"))
    return f'<div class="grid2">{"".join(parts)}</div>'


def _tts(model: dict[str, Any]) -> str:
    t = model["tts"]
    series = [Series("batch (whole WAV)", _pts(t["batch_table"], "chars"), 2), Series("streamed first chunk", _pts(t["stream_first_chunk"], "chars"), 3)]
    audio = Series("audio length", [(r["chars"], r["p50"] * 1000, r["p90"] * 1000) for r in t["audio_seconds"]], 1)
    return f'<div class="grid2">{charts.line_chart("Time to first audio vs text length", "pocket-tts: streamed first chunk is flat, batch grows with text", series, "characters")}{charts.line_chart("Speech duration vs text length", "How long the answer takes to play (the real floor on completion)", [audio], "characters")}</div>'


def _stt(model: dict[str, Any]) -> str:
    stt = model["stt"]
    rows = []
    for mode in ("always_on", "conversation"):
        for r in stt["end_to_final"]:
            if r["mode"] != mode:
                continue
            rtt = _row(stt["rtt_mean"], mode=mode, frame_ms=r["frame_ms"], utterance=r["utterance"])
            p99 = _row(stt["rtt_p99"], mode=mode, frame_ms=r["frame_ms"], utterance=r["utterance"])
            late = _row(stt["late_frames"], mode=mode, frame_ms=r["frame_ms"], utterance=r["utterance"])
            rows.append([mode, r["utterance"], f'{r["frame_ms"]} ms', f'<span class="n">{r["p50"]:.0f}</span>', f'{r["p90"]:.0f}', f'{rtt["p50"]:.1f}' if rtt else "–", f'{p99["p50"]:.0f}' if p99 else "–", f'{late["p50"]:.0f}' if late else "–"])
    empties = "".join(f'<li><code>{r["mode"]}</code>/{r["utterance"]}: {r["mean"] * 100:.0f}% of runs returned no text ({r["n"]} runs)</li>' for r in stt["empty_transcript_rate"])
    tbl = charts.table(["mode", "utterance", "frame", "end→final p50 ms", "p90", "RPC RTT mean ms", "RTT p99 ms", "late frames/utterance"], rows, "Speaker ASR, real-time pace")
    return f'{tbl}<h3>Empty results</h3><ul>{empties}</ul>'


def _vad(model: dict[str, Any]) -> str:
    vad = model["vad"]
    if not vad.get("fired_rate"):
        return '<p class="note">VAD benchmark not run.</p>'
    rows = []
    for r in vad["fired_rate"]:
        key = {"utterance": r["utterance"], "silence_duration": r["silence_duration"], "frame_ms": r["frame_ms"]}
        lag, fin, cpu = _row(vad["start_lag"], **key), _row(vad["end_to_final"], **key), _row(vad["cpu_ms_per_frame"], **key)
        rows.append([r["utterance"], f'{r["silence_duration"]:g}', f'{r["frame_ms"]}', f'{r["mean"] * 100:.0f}%', f'{lag["p50"] * 1000:.0f}' if lag else "–", f'{fin["p50"] * 1000:.0f}' if fin else "–", f'{cpu["p50"]:.2f}' if cpu else "–"])
    return charts.table(["utterance", "silence_duration s", "frame ms", "finalized", "speech-start lag ms", "end→final ms (audio time)", "CPU ms/frame"], rows, "Silero VadDetector (batch-STT path), audio time")


# ---------------------------------------------------------------- router atlas

PATH_LABELS = {
    "focused_fast_path": "Focused (skips routing)", "single_candidate": "Single candidate (skips routing)",
    "multi_route_accept": "Normal route, small history", "multi_route_accept_ctx12k": "Normal route, 12 KB history",
    "preamble_then_answer": "Normal route + destination preamble", "admit_slow_500ms": "Normal route, 500 ms admit",
    "all_ineligible_clarify": "Router asks to clarify (no answer)", "admission_clarify": "Destination asks to clarify (no answer)",
    "misroute_declined_once": "Misroute caught: decline → reroute", "release_with_text_then_reroute": "Release with text → reroute",
    "declines_exhausted_default": "Declines exhausted → default",
}


def _router(model: dict[str, Any]) -> str:
    r = model["router"]
    rows = []
    for name, label in PATH_LABELS.items():
        tot, dec, adm = _row(r["total"], path=name), _row(r["decide_ms"], path=name), _row(r["admit_ms"], path=name)
        if not tot:
            continue
        decide, admit = (dec["p50"] if dec else 0.0), (adm["p50"] if adm else 0.0)
        other = max(tot["p50"] - decide - admit, 0.0)
        segs = [("routing decisions", decide, 1), ("destination admit", admit, 2), ("other / control speech", other, 4)]
        rows.append((label, segs, tot["p90"]))
    atlas = charts.hbar_stack("Router path atlas: time before the destination starts answering", "Synthetic destinations over the real decision model; tick = p90 of the total", rows)
    ack_rows = []
    for started in r["ack_answer_started"]:
        emit, delay = started.get("emit_ms") or 0.0, started["answer_delay_ms"]
        e = _row(r["ack_first_output"], emit_ms=started.get("emit_ms"), answer_delay_ms=delay)
        chat = _row(r["ack_chat_ms"], emit_ms=started.get("emit_ms"), answer_delay_ms=delay)
        spoke = f'{e["n"]}/{started["n"]}' if e else f'0/{started["n"]}'
        ack_rows.append([f"{delay:.0f}", f"{emit:.0f}", spoke, _ms(e["p50"]) if e else "no ack", _ms(started["p50"]),
                         _ms(chat["p50"]) if chat and chat["p50"] else "–"])
    ack = charts.table(["destination TTFT ms", "ack delivery blocks ms", "turns with ack", "first output p50", "answer starts p50", "ack LLM call p50"], ack_rows, "Acknowledgment race (acknowledgment_delay_secs = 1.0)")
    return f'{atlas}<h3>Acknowledgment race</h3>{ack}'


# ---------------------------------------------------------------- streaming

def _streaming(model: dict[str, Any]) -> str:
    s = model["streaming"]
    parts = []
    sentences = sorted({int(r["sentences"]) for r in s["first_audio"]})
    cats = [str(n) for n in sentences]
    names = {"S1_chat_batch_tts": ("chat → batch TTS", 2), "S2_chat_stream_tts": ("chat → streamed TTS", 4), "S3_stream_sentence_tts": ("stream → sentence → streamed TTS", 3)}
    series = [Series(label, [(str(int(r["sentences"])), r["p50"], r["p90"]) for r in s["first_audio"] if r["strategy"] == key], slot) for key, (label, slot) in names.items()]
    parts.append(charts.bar_chart("Time to first audio by delivery strategy", "Answer length in sentences; same LLM, same TTS", cats, series, xlabel="sentences in the answer"))
    mode_names = {"chat": ("chat (non-streaming)", 2), "stream": ("stream()", 1), "stream_events": ("stream_events()", 3)}
    first = [Series(label, [(str(int(r["sentences"])), r["p50"], r["p90"]) for r in s["modes_first_text"] if r["mode"] == m], slot) for m, (label, slot) in mode_names.items() if m != "chat"]
    total = [Series(label, [(str(int(r["sentences"])), r["p50"], r["p90"]) for r in s["modes_total"] if r["mode"] == m], slot) for m, (label, slot) in mode_names.items()]
    parts.append(charts.bar_chart("Time to last token", "Streaming does not make the whole answer arrive sooner", cats, total, xlabel="sentences in the answer"))
    parts.append(charts.bar_chart("Time to first visible text (streaming)", "Non-streaming can only show text at the 'last token' time", cats, first, xlabel="sentences in the answer"))
    rows = []
    for r in s["short_total"]:
        f = _row(s["short_first"], max_tokens=r["max_tokens"], mode=r["mode"])
        rows.append([f'{int(r["max_tokens"])}', r["mode"], f'{r["p50"]:.0f}', f'{f["p50"]:.0f}' if f else "–"])
    short = charts.table(["max tokens", "mode", "total p50 ms", "first text p50 ms"], rows, "Very short outputs")
    tool_rows = [[r["mode"], f'{r["p50"]:.0f}', f'{r["p90"]:.0f}'] for r in s["tools_total"]]
    tools = charts.table(["mode", "time to tool call p50 ms", "p90"], tool_rows, "A turn that ends in a tool call")
    stall_rows = [[str(int(r["sentences"])), f'{r["mean"]:.0f}', f'{r["max"]:.0f}'] for r in s["stall"] if r["strategy"].startswith("S3")]
    stall = charts.table(["sentences", "mean audible stall ms", "worst stall ms"], stall_rows, "Gaps between sentences when streaming (0 = continuous speech)")
    return f'<div class="grid2">{"".join(parts)}</div><div class="grid2"><div>{short}</div><div>{tools}</div></div>{stall}'


# ---------------------------------------------------------------- calculator + waterfall

DEFAULT_DESIGN = [
    {"kind": "endpoint", "name": "End of speech → final transcript"},
    {"kind": "route", "name": "Route", "path": "multi_route", "history_bytes": 6000},
    {"kind": "speak", "name": "LLM → first audio", "prompt_tokens": 4000, "out_tokens": 50, "first_sentence_tokens": 20, "cached": True, "strategy": "sentence", "playback_ms": 140},
]


def waterfall(res: Any, title: str, sub: str) -> str:
    slots = {"measured": 3, "fitted": 1, "assumed": 4}
    rows = [(s.name, [("", s.start_p50, 0), (s.basis, s.end_p50 - s.start_p50, slots.get(s.basis, 1))], s.end_p90)
            for s in res.stages]
    return charts.hbar_stack(title, sub, rows)


def _validation(model: dict[str, Any]) -> str:
    rows = []
    for group_name, checks in (("first audio", first_audio_checks(model)), ("router path", router_checks(model))):
        for c in checks:
            err = error_pct(c)
            cls = "good" if abs(err) <= 15 else ("warn" if abs(err) <= 30 else "bad")
            rows.append([group_name, c["case"], _ms(c["predicted"]), _ms(c["measured"]), f'<span style="color:var(--{cls})">{err:+.0f}%</span>'])
    errs = [abs(error_pct(c)) for c in independent(first_audio_checks(model) + router_checks(model))]
    summary = f'<p>Median absolute error <b>{sorted(errs)[len(errs) // 2]:.0f}%</b> over {len(errs)} independent checks; worst {max(errs):.0f}%. The two calibration points are shown but not counted. The checks reuse the prose prompts the parameters were measured on, so they show the arithmetic composes, not that it generalizes to a different agent.</p>' if errs else ""
    return summary + charts.table(["kind", "case", "estimator", "measured", "error"], rows, "Estimator vs the measurements it composes")


def _patterns(model: dict[str, Any]) -> str:
    rows = []
    for name, pattern in load_patterns().items():
        try:
            res = price(model, pattern)
        except KeyError as exc:
            rows.append([f"<b>{html.escape(pattern.title)}</b>", f"unavailable: {html.escape(str(exc))}", "", "", "", ""])
            continue
        total = max(sum(s.end_p50 - s.start_p50 for s in res.stages), 1.0)
        solid = sum(s.end_p50 - s.start_p50 for s in res.stages if s.basis in ("measured", "fitted")) / total
        first = _ms(res.first_audio_p50) if res.first_audio_p50 is not None else "n/a"
        p90 = _ms(res.first_audio_p90) if res.first_audio_p90 is not None else "–"
        biggest = max(res.stages, key=lambda s: s.end_p50 - s.start_p50)
        rows.append([f"<b>{html.escape(pattern.title)}</b><br><span class=note>{html.escape(pattern.description)}</span>",
                     f"<b>{first}</b>", p90, _ms(res.done_p50), f"{solid * 100:.0f}%",
                     f"{html.escape(biggest.name)} ({_ms(biggest.end_p50 - biggest.start_p50)})"])
    return charts.table(["pattern", "to first audio p50", "p90", "playback done p50", "measured or fitted", "largest stage"], rows,
                        "Common turn shapes priced with the fitted model, thinking off. Stages marked assumed (tool time, retrieval, frame fetch) have no benchmark behind them.")


def _calculator(model: dict[str, Any]) -> str:
    return """<div class="calc"><div class="row" id="calcin"></div><div id="calcout"></div></div>"""


def build(model: dict[str, Any], out: Path, findings_html: str = "", meta: dict[str, Any] | None = None) -> Path:
    """Write the report page to ``out`` and return the path."""
    meta = meta or {}
    res = evaluate(model, DEFAULT_DESIGN)
    sections = [
        _section("model", "A typical voice turn, end to end", '<p class="sub">Priced from the fitted model for a conversation with a routed LLM answer at 4k cached prompt tokens and 6 KB of router history. Bars show p50 duration of each stage; the tick is the stage p90.</p>'
                 + f'<div class="kpis"><div class="kpi"><b>{_ms(res.first_audio_p50)}</b><span>time to first audio (p50)</span></div><div class="kpi"><b>{_ms(res.first_audio_p90)}</b><span>conservative p90</span></div><div class="kpi"><b>{_ms(res.done_p50)}</b><span>until the answer finishes playing</span></div></div>'
                 + waterfall(res, "Where the time goes", "Each bar starts when the previous stage ends; tick = cumulative p90. Transport and device terms are assumptions."), ""),
        _section("findings", "Findings that should change designs", "", findings_html or '<p class="note">No findings written yet.</p>'),
        _section("llm", "LLM", "<p class=sub>How latency grows with context, how fast it decodes, what thinking costs, and what concurrency does.</p>", _safe(_llm, model)),
        _section("decision", "Decision model (router)", "<p class=sub>The router makes this call before every multi-destination turn; its cost is set by how much history it reads.</p>", _safe(_decision, model)),
        _section("router", "Router paths", "<p class=sub>Every path through <code>AgentRouter</code>, so you can price a misroute before building one.</p>", _safe(_router, model)),
        _section("streaming", "Streaming vs non-streaming", "<p class=sub>When streaming pays, and when it is pure overhead.</p>", _safe(_streaming, model)),
        _section("vlm", "VLM", "", _safe(_vlm, model)),
        _section("tts", "Text to speech", "", _safe(_tts, model)),
        _section("stt", "Speech recognition and VAD", "", _safe(_stt, model) + "<h3>Silero VAD (batch-STT path)</h3>" + _safe(_vad, model)),
        _section("patterns", "What common patterns cost", "<p class=sub>Generic turn shapes composed from the measured components. Edit or add a pattern in <code>patterns/</code>.</p>", _safe(_patterns, model)),
        _section("validation", "Can you trust the estimator?", "<p class=sub>Predicting composed results from the fitted components, then comparing with what was measured end to end.</p>", _safe(_validation, model)),
        _section("calc", "Calculator", "<p class=sub>Compose a turn and read off time to first audio. Same arithmetic as <code>xr-latency estimate</code>.</p>", _calculator(model)),
    ]
    nav = "".join(f'<a href="#{i}">{t}</a>' for i, t in (("model", "Typical turn"), ("findings", "Findings"), ("llm", "LLM"), ("decision", "Decision"), ("router", "Router paths"), ("streaming", "Streaming"), ("vlm", "VLM"), ("tts", "TTS"), ("stt", "STT/VAD"), ("patterns", "Patterns"), ("validation", "Validation"), ("calc", "Calculator")))
    calc_js = (HERE / "calc.js").read_text()
    page_js = (HERE / "page.js").read_text()
    doc = f"""<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>xr-ai latency profile</title><style>{CSS}</style></head><body><main>
<button class="theme">theme</button><h1>xr-ai latency profile</h1>
<p class="sub">{html.escape(meta.get("subtitle") or machine_label(model.get("machine")))}</p><nav>{nav}</nav>{"".join(sections)}
<p class="note">Numbers are medians (p50) and 90th percentiles from {model.get("sample_count", 0)} clean samples (samples taken while another client had requests in flight are dropped). Stage p90s are summed, which overstates a real turn's p90.</p></main>
<div id="tip"></div><script id="model-data" type="application/json">{json.dumps(model)}</script><script>{calc_js}</script><script>{JS_TIP}</script><script>{page_js}</script></body></html>"""
    out.write_text(doc)
    return out


def make_report(run_dirs: list[Path], out: Path, findings: Path | None = None, subtitle: str = "") -> Path:
    """Fit ``model.json`` from ``run_dirs`` into ``out`` and render ``report.html`` next to it."""
    from ..fit import fit_all

    out.mkdir(parents=True, exist_ok=True)
    model = fit_all(run_dirs)
    (out / "model.json").write_text(json.dumps(model, indent=2))
    return build(model, out / "report.html", findings.read_text() if findings else "", {"subtitle": subtitle})
