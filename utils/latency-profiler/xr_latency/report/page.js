// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
(function () {
  const model = JSON.parse(document.getElementById("model-data").textContent);
  const box = document.getElementById("calcin"), out = document.getElementById("calcout");
  if (!box) return;
  const vlmCells = (model.vlm && model.vlm.chat_total) || [];
  const resolutions = [...new Set(vlmCells.map((r) => r.res))];
  const fields = [
    ["asr", "Speech endpointing", "select", [["speaker", "Speaker ASR (measured)"], ["custom", "Custom ms"]], "speaker"],
    ["asr_ms", "Custom endpointing ms", "number", null, 1200],
    ["path", "Router path", "select", Object.keys(XRLatency.PATHS).map((k) => [k, k]), "multi_route"],
    ["history_bytes", "Router history bytes", "number", null, 6000],
    ["admit_ms", "Destination admit ms", "number", null, 0],
    ["use_vlm", "Vision call first", "select", [["no", "No"], ["yes", "Yes"]], "no"],
    ["res", "Image resolution", "select", resolutions.map((r) => [r, r]), resolutions[0] || ""],
    ["images", "Images", "select", [[1, "1"], [2, "2"], [4, "4"]], 1],
    ["hops", "Serial tool hops", "number", null, 0],
    ["tool_ms", "Time per tool ms", "number", null, 200],
    ["prompt_tokens", "Answer prompt tokens", "number", null, 4000],
    ["cached", "Prefix cached", "select", [["yes", "Yes (stable prefix)"], ["no", "No"]], "yes"],
    ["thinking", "Thinking", "select", [["no", "Off"], ["yes", "On"]], "no"],
    ["out_tokens", "Answer tokens", "number", null, 80],
    ["first_sentence_tokens", "Tokens in first sentence", "number", null, 20],
    ["strategy", "Delivery", "select", [["sentence", "Stream → sentence → streamed TTS"], ["stream_full", "Whole text → streamed TTS"], ["batch", "Whole text → batch TTS"]], "sentence"],
    ["playback_ms", "Transport + jitter buffer ms (assumed)", "number", null, 140],
  ];
  const state = {};
  for (const [k, label, type, opts, def] of fields) {
    state[k] = def;
    const wrap = document.createElement("div");
    const l = document.createElement("label");
    l.textContent = label;
    let el;
    if (type === "select") {
      el = document.createElement("select");
      for (const [v, t] of opts) { const o = document.createElement("option"); o.value = v; o.textContent = t; el.appendChild(o); }
    } else { el = document.createElement("input"); el.type = "number"; }
    el.value = def;
    el.addEventListener("input", () => { state[k] = type === "number" ? Number(el.value) : el.value; render(); });
    wrap.append(l, el);
    box.appendChild(wrap);
  }
  const fmt = (v) => (v == null ? "–" : v >= 1000 ? (v / 1000).toFixed(2) + " s" : Math.round(v) + " ms");
  function design() {
    const s = state, d = [];
    d.push(s.asr === "speaker" ? { kind: "endpoint", name: "End of speech → final transcript" } : { kind: "fixed", name: "Endpointing (custom)", p50: s.asr_ms });
    d.push({ kind: "route", name: "Route: " + s.path, path: s.path, history_bytes: s.history_bytes, admit_ms: s.admit_ms });
    if (s.use_vlm === "yes") d.push({ kind: "vlm", name: "Vision call", res: s.res, images: Number(s.images) });
    if (s.hops > 0) d.push({ kind: "tool_hop", name: "Tool hops", hops: s.hops, tool_ms: s.tool_ms, prompt_tokens: s.prompt_tokens, cached: s.cached === "yes" });
    d.push({ kind: "speak", name: "LLM → first audio", prompt_tokens: s.prompt_tokens, cached: s.cached === "yes", thinking: s.thinking === "yes", out_tokens: s.out_tokens, first_sentence_tokens: s.first_sentence_tokens, strategy: s.strategy, playback_ms: s.playback_ms });
    return d;
  }
  function render() {
    try {
      const r = XRLatency.evaluate(model, design());
      const max = r.stages.reduce((a, s) => Math.max(a, s.end_p90), 1);
      const rows = r.stages.map((s) => `<tr><td>${s.name}</td><td class="n">${fmt(s.end_p50 - s.start_p50)}</td><td class="n">${fmt(s.end_p90 - s.start_p90)}</td><td><span class="pill ${s.basis}">${s.basis}</span></td><td class="note">${s.note}</td><td style="width:30%"><div style="margin-left:${(100 * s.start_p50 / max).toFixed(1)}%;width:${Math.max(100 * (s.end_p50 - s.start_p50) / max, 0.5).toFixed(1)}%;height:10px;border-radius:3px;background:var(--accent)"></div></td></tr>`).join("");
      out.innerHTML = `<div class="kpis"><div class="kpi"><b>${fmt(r.first_audio_p50)}</b><span>time to first audio (p50)</span></div><div class="kpi"><b>${fmt(r.first_audio_p90)}</b><span>conservative p90</span></div><div class="kpi"><b>${fmt(r.done_p50)}</b><span>until the answer has finished playing</span></div></div><table class="tbl"><thead><tr><th>stage</th><th class="n">p50</th><th class="n">p90</th><th>basis</th><th>what it is</th><th>timeline</th></tr></thead><tbody>${rows}</tbody></table>`;
    } catch (e) { out.innerHTML = `<p class="note">${e.message}</p>`; }
  }
  render();
})();
