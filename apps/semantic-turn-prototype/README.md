# Semantic-turn prototype

Private transcript-stream experiment using the existing Clef DecisionService.
Replay prints admission actions; the optional live worker connects the public
SDK admission hook to microphone input and spoken replies. It does not execute tools.

See the [prototype guide](https://nvidia.github.io/xr-ai/latest/guides/semantic-turn-prototype.html)
for behavior, event contracts, limitations, and qualification.

From this directory, exercise orchestration without downloading a model:

```bash
PYTHONPATH=worker uv --config-file ../../uv.toml run python -m semantic_turn_prototype eval/flows.jsonl --fixture --policy policy.json
uv --config-file ../../uv.toml run --group dev pytest
```

To classify the same stream using a running Clef service, remove `--fixture`:

```bash
PYTHONPATH=worker uv --config-file ../../uv.toml run python -m semantic_turn_prototype eval/flows.jsonl --policy policy.json --clef-url http://127.0.0.1:8120
```

The fixture backend is explicitly labeled `scripted-plumbing-only`. Its labels
are expected choices, not real inference. Commands do not start or download Clef.
Reuse a running conversation-example Clef server or start the repository's
existing [Clef service](../../services/clef-server/clef_server.yaml).

Score the independent labeled classification cases with real Clef:

```bash
PYTHONPATH=worker uv --config-file ../../uv.toml run python -m semantic_turn_prototype eval/classification.jsonl --classify --policy policy.json
```

This reports choices, confidence, probability maps, latency, and per-case results.
It rejects `--fixture`: scripted labels cannot establish model accuracy.

`policy.json` holds experimental confidence, stability, capacity, and follow-up
settings. For example, set `"independent": "interrupt"` to interrupt the current
reply for an unrelated query instead of queueing it.

## Live microphone worker

This worker uses the approved opt-in SDK hook and existing model services. It
has no tools or external side effects, so unspoken generation can be replaced.
Real model/acoustic performance is not yet qualified.

From this directory, first start the existing models with the desired profile:

```bash
uv --config-file ../../uv.toml run --project ../../agent-samples/conversation-example/model-servers conversation_example_model_servers --gpu-profile 96G_blackwell
```

Use `dual_48G_ada` or `spark` for the corresponding hardware. In a separate
terminal, from this same directory, start only the hub:

```bash
uv --config-file ../../uv.toml run --project ../../services/device-io-hub device_io_hub --config ../../agent-samples/conversation-example/yaml/device_io_hub.yaml
```

Then start this worker instead of the normal conversation worker:

```bash
PYTHONPATH=worker uv --config-file ../../uv.toml run python -m semantic_turn_prototype --live --models ../../agent-samples/conversation-example/yaml/models.json --voice-gate ../../agent-samples/conversation-example/yaml/voice_gate.yaml --policy policy.json
```

Connect the existing client using the hub's displayed URL and say the configured
start phrase: “Hey agent, let's start talking.” Do not run another conversation
worker against this hub during the experiment. The normal model-stack stop
command remains:

```bash
uv --config-file ../../uv.toml run --project ../../agent-samples/conversation-example/model-servers conversation_example_model_servers --stop
```
