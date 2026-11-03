<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# RAG and text memory tools (agent-sdk/xr-ai-tools/xr_ai_tools/rag.py, text_memory.py, rpc.py)

## Role in a turn
`retrieve` (RAG) is a pre-answer lookup on the critical path (serial before the LLM answer). Text memory: `add_transcript` is background persistence per utterance; `query_transcripts`/`recall_conversation` are lookup tools on the critical path when called.

## Critical-path calls
### RAG (`rag.py`)
1. `Tool.invoke`/`execute` validation + Relay wrapper (tools-relay-overhead.md).
2. `RPCClient.call("retrieve", {query, top_k})` `:107-110`: msgpack pack -> ZMQ DEALER send under `_send_lock` -> wait on future with timeout -> receiver task resolves -> unpack (`rpc.py:75-132`). One IPC round trip. Server side (outside scope): embed query + vector search (+ any rerank).
3. `RetrieveResult.model_validate(result)` pydantic parse of up to `top_k` chunks `:108`.
4. `list_documents`, `get_health` (2 s timeout) are cheap calls.
### Text memory (`text_memory.py`)
* `add_transcript`: `await asyncio.to_thread(store.append)` `:284-289` -> under a threading `Lock`: `_path(create=True)` (loops over `.identity` and `.jsonl` existence checks, 2 `resolve()` per candidate, `:195-207`), open file in append mode, write one JSON line (no fsync).
* `query_transcripts`: `to_thread(store.query)` `:336-342`: `_path(create=False)` -> for non-canonical IDs scans ALL `*.identity` files, reading each (`:187-191`, O(#sources) file reads); then `read_text()` of the entire JSONL, `splitlines`, pydantic `model_validate_json` per line, filter, sort (`:216-240`).
* `recall_conversation` (`:352-370`): two SERIAL `to_thread(query)` calls (`participant:user`, then `participant:agent`), then merge-sort.
* `get_transcript_stats`: a full `query` (whole file) then aggregates (`:257-265`).
* `list_sources`: glob + read every identity file (`:242-255`).

## Fixed waits & timers
| Constant | Default | Where |
|---|---|---|
| RAG RPC timeout | 30.0 s | `rag.py:89` |
| Health call timeout | 2.0 s | `rag.py:123-125` |
| `top_k` | default 4, range 1..20 | `rag.py:24-29` |
| RPC client default | 10.0 s (tracking uses this); `RPCClient(endpoint, timeout_s=10.0)` | `rpc.py:48` |
| `_MAX_TIMESTAMP_US` | int64 max (default `end_us`) | `text_memory.py:21, :125` |
| ZMQ `LINGER` | 0 | `rpc.py:67` |
| Retries / polling | none | - |
| Text memory file IO | no fsync; sync IO in default thread pool | `:213-214` |

## Cost drivers
RAG: query embedding (server), index size (chunk_count), `top_k`, response size (chunk text) in tokens when returned to the LLM (up to 20 chunks x chunk length added to the next LLM prompt: increases next prefill). Text memory: transcript file size (query = O(lines) parse), number of sources (identity scan), number of calls.

## Estimate formula
RAG: `t_retrieve = t_validate + t_rpc_rtt(~0.2-0.5 ms localhost ipc/tcp, guess) + t_embed_query(server) + t_search(index) + t_unpack(chunks) + t_validate_result` -> plus downstream `delta_prefill = top_k * chunk_tokens / prefill_tps`. Server terms: needs measurement.
Text memory: `t_query = t_thread_hop(~0.1 ms) + t_scan_identities(n_sources * ~20 us) + t_read(file_bytes/1GB/s) + n_lines * ~15 us (pydantic)` e.g. 10 K lines ~ 150-250 ms (guess). `add_transcript` ~0.3-1 ms.

## Hidden latency / improvement candidates
* `query_transcripts` re-reads and re-parses the whole file every call: grows linearly with the session (10 K lines ~100-250 ms guess); an in-memory index or reverse-seek by timestamp would make it O(window). Also `recall_conversation` does it twice serially: `asyncio.gather` on the two reads halves wall time (saves 50 % of a possibly 100s-of-ms call).
* `get_transcript_stats` runs a full query then discards segments: single-pass count would save the pydantic parse.
* `add_transcript` per utterance uses a global lock + path resolve with filesystem `exists()` calls: tiny, background.
* RAG RPC has no streaming and no cache: repeated identical queries pay full cost; a query->result LRU would skip it (hit = ~0).
* `top_k=4` default and `le=20` bound prompt growth, a model that asks for 20 adds a large prefill (e.g. 20 x 200 tokens = 4 K tokens ~ 100s of ms).

## What to measure live
`RAGTools(endpoint).retrieve.execute(RetrieveRequest(query=..., top_k=4/20))` wall time vs `await rpc.call("retrieve", ...)` direct to separate tool wrapper cost; `get_health()` for index size. For text memory: pre-fill a source with 100/1 K/10 K/100 K lines then time `TextMemoryTools(dir).query_transcripts.execute(...)` and `recall_conversation.execute(...)`.

## Confidence
Control flow, constants: code-derived. Server retrieval time and file-scan numbers: needs measurement.
