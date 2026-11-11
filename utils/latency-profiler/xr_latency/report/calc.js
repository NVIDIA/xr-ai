// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// JS port of xr_latency/estimate.py. Keep the two in lockstep: tests/parity check runs both on the same model.

(function (root) {
  const PATHS = {
    single_candidate: [0, 0, 0],
    focused: [0, 1, 0],
    multi_route: [1, 1, 0],
    misroute_declined_once: [2, 2, 0],
    release_with_text_then_reroute: [2, 2, 1],
    declines_exhausted_default: [2, 2, 0],
    clarify: [1, 0, 1],
  };

  function curve(c, x, stat) {
    if (!c) throw new Error("model has no data for this term; run the matching benchmark first");
    const f = c[stat];
    return Math.max(f.intercept + f.slope * x, 0);
  }
  const est = (c, x, basis = "fitted") => ({ p50: curve(c, x, "p50"), p90: curve(c, x, "p90"), basis });
  const lookup = (table, keys) => table.find((r) => Object.entries(keys).every(([k, v]) => r[k] === v)) || null;

  function llmTtft(m, promptTokens, cached) {
    const llm = m.llm;
    const e = cached ? est(llm.ttft_warm_vs_ctx, promptTokens) : est(llm.ttft_cold_vs_ctx, promptTokens);
    const floor = llm.ttft_floor;
    return floor ? { p50: Math.max(e.p50, floor.p50), p90: Math.max(e.p90, floor.p90), basis: e.basis } : e;
  }
  const charsPerToken = (m) => (m.streaming && m.streaming.chars_per_token) || 4.0;
  function decodePerToken(m) {
    const c = m.llm.decode_total_vs_tokens_chat;
    return { p50: Math.max(c.p50.slope, 0), p90: Math.max(c.p90.slope, 0), basis: "fitted" };
  }
  function thinkingTax(m) {
    const rows = m.llm.thinking.first_text;
    const on = lookup(rows, { thinking: true }), off = lookup(rows, { thinking: false });
    if (!on || !off) throw new Error("model has no thinking experiment");
    return { p50: Math.max(on.p50 - off.p50, 0), p90: Math.max(on.p90 - off.p90, 0), basis: "measured" };
  }
  function reasoningCost(m, s) {
    if (s.reasoning_tokens) {
      const dec = decodePerToken(m);
      return { p50: s.reasoning_tokens * dec.p50, p90: s.reasoning_tokens * dec.p90, basis: "fitted" };
    }
    return s.thinking ? thinkingTax(m) : { p50: 0, p90: 0, basis: "measured" };
  }
  function decideMs(m, historyBytes, labels = 4) {
    if (m.router && m.router.decide_curve) return est(m.router.decide_curve, historyBytes);
    const c = m.decision["total_vs_history_bytes_labels" + labels] || m.decision.total_vs_history_bytes_labels4;
    return est(c, historyBytes);
  }
  function ttsFirst(m, chars, stream) {
    const tts = m.tts;
    if (stream) {
      const rows = tts.stream_first_chunk;
      const mean = (k) => rows.reduce((a, r) => a + r[k], 0) / rows.length;
      return { p50: mean("p50"), p90: mean("p90"), basis: "measured" };
    }
    return est(tts.batch_total_vs_chars, chars);
  }
  function audioMsPerChar(m) {
    const rows = m.tts.audio_seconds;
    return rows.reduce((a, r) => a + (r.p50 * 1000) / r.chars, 0) / rows.length;
  }

  function endpoint(m, s) {
    const mode = s.mode || "conversation", utt = s.utterance || "medium", frame = s.frame_ms || 10;
    const r = m.stt.end_to_final.find((x) => x.mode === mode && x.utterance === utt && x.frame_ms === frame);
    if (!r) throw new Error("model has no matching speaker-ASR measurement");
    return [{ p50: r.p50, p90: r.p90, basis: "measured" }, "speaker ASR end of speech to final", 0];
  }
  function route(m, s) {
    const [decides, admits, controls] = PATHS[s.path || "multi_route"];
    const d = decideMs(m, s.history_bytes ?? 3000, s.labels || 4);
    const admit = s.admit_ms || 0;
    const ctrl = (s.control_speech_ms || 0) * controls;
    return [{ p50: decides * d.p50 + admits * admit + ctrl, p90: decides * d.p90 + admits * admit + ctrl, basis: "fitted" },
      `${decides} decide + ${admits} admit + ${controls} control speech`, 0];
  }
  function llm(m, s) {
    const ttft = llmTtft(m, s.prompt_tokens + m.llm.chat_overhead_tokens, !!s.cached);
    const dec = decodePerToken(m);
    const tax = reasoningCost(m, s);
    const n = s.out_tokens ?? 60;
    return [{ p50: ttft.p50 + tax.p50 + n * dec.p50, p90: ttft.p90 + tax.p90 + n * dec.p90, basis: "fitted" },
      `ttft + ${n} tokens decode` + (s.thinking ? " + thinking" : ""), 0];
  }
  function toolHop(m, s) {
    let p50 = 0, p90 = 0, prompt = s.prompt_tokens;
    const hops = s.hops ?? 1;
    for (let i = 0; i < hops; i++) {
      const [e] = llm(m, { ...s, prompt_tokens: prompt, out_tokens: s.hop_out_tokens ?? 40 });
      p50 += e.p50 + (s.tool_ms || 0);
      p90 += e.p90 + (s.tool_ms || 0);
      prompt += s.hop_growth_tokens ?? 300;
    }
    return [{ p50, p90, basis: "fitted" }, `${hops} serial tool hops (tool_ms is your assumption)`, 0];
  }
  function vlm(m, s) {
    const table = s.cached ? m.vlm.cached_total : m.vlm.chat_total;
    const r = lookup(table, { res: s.res, images: s.images });
    if (!r) throw new Error("no measured VLM cell for that resolution and image count");
    return [{ p50: r.p50, p90: r.p90, basis: "measured" }, `${s.images} image(s) at ${s.res}`, 0];
  }
  function speak(m, s) {
    const ttft = llmTtft(m, s.prompt_tokens + m.llm.chat_overhead_tokens, !!s.cached);
    const dec = decodePerToken(m);
    const tax = reasoningCost(m, s);
    const total = s.out_tokens ?? 60, first = s.first_sentence_tokens ?? 20, cpt = charsPerToken(m);
    const strategy = s.strategy || "sentence";
    let gen, sp;
    if (strategy === "sentence") {
      gen = { p50: ttft.p50 + tax.p50 + first * dec.p50, p90: ttft.p90 + tax.p90 + first * dec.p90 };
      sp = ttsFirst(m, first * cpt, true);
    } else {
      gen = { p50: ttft.p50 + tax.p50 + total * dec.p50, p90: ttft.p90 + tax.p90 + total * dec.p90 };
      sp = ttsFirst(m, total * cpt, strategy === "stream_full");
    }
    const play = s.playback_ms || 0;
    const tail = total * cpt * audioMsPerChar(m);
    return [{ p50: gen.p50 + sp.p50 + play, p90: gen.p90 + sp.p90 + play, basis: "fitted" }, `${strategy} delivery`, tail, true];
  }

  const KINDS = { endpoint, route, llm, tool_hop: toolHop, vlm, speak };

  function evaluate(m, design) {
    const out = { stages: [], first_audio_p50: null, first_audio_p90: null, done_p50: 0, done_p90: 0 };
    let t50 = 0, t90 = 0;
    for (const s of design) {
      let e, note, tail = 0, audio = false;
      if (s.kind === "fixed") {
        e = { p50: s.p50, p90: s.p90 ?? s.p50, basis: s.basis || "assumed" };
        note = s.note || "";
      } else if (KINDS[s.kind]) {
        [e, note, tail, audio] = KINDS[s.kind](m, s);
      } else throw new Error("unknown stage kind " + s.kind);
      out.stages.push({ name: s.name || s.kind, start_p50: t50, end_p50: t50 + e.p50, start_p90: t90, end_p90: t90 + e.p90, basis: e.basis, note });
      t50 += e.p50; t90 += e.p90;
      if (audio && out.first_audio_p50 === null) { out.first_audio_p50 = t50; out.first_audio_p90 = t90; }
      t50 += tail; t90 += tail;
    }
    out.done_p50 = t50; out.done_p90 = t90;
    return out;
  }

  const api = { evaluate, PATHS, decideMs, llmTtft, decodePerToken, thinkingTax, ttsFirst };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.XRLatency = api;
})(typeof window !== "undefined" ? window : globalThis);
