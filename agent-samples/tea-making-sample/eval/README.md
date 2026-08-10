<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Guidance evals

`cases.yaml` covers each foreground action and every step policy with facts that do
not appear as worked examples in the prompts. `check.py` validates schema,
coverage, and prompt budgets without model servers. During a model-backed human
test, run the exact production foreground agents against the active model service:

```bash
uv run --directory agent-samples/tea-making-sample/worker \
  python ../eval/routes.py --models ../yaml/models.omni.json
```

Run the production change-detection, transcript-summary, and video-delta prompts separately:

```bash
uv run --directory agent-samples/tea-making-sample/worker \
  python ../eval/backgrounds.py --models ../yaml/models.omni.json
```

The command expands the compact state matrix and prints the lifecycle NAT tool
or direct-answer action for every case. Transition, skip, exit, restart,
status, and ambiguous questions run from every configured tea step; root tea
launch, background application launch, application status, and general queries run
separately. Skip cases also verify that the workflow
actually advances, while ordinary next commands against incomplete steps must
stay put. The command exits nonzero on a mismatch. For step probes, compare the
logged tool call or commit to the expected fields. Visual fixtures are ordinary
captions, including one negative water-visibility case that must fail the
deterministic evidence gate.
Voice cases exercise deterministic foreground selection: root input reaches
root and active tea input reaches the current tea-step variant. Bare “next” and
“next step” must reach advance, while questions containing transition words
remain direct answers. Appliance and adversarial timer commands remain in the
tea foreground; explicit guide reset and restart remain available. General tea
knowledge must use RAG, general scene questions must use current vision, and a
deictic tea question must inspect before retrieval while root owns foreground.
The background suite verifies a user-specified monitoring focus, an important
visual event, a viewpoint-only non-event, human-labeled UI outputs, one periodic
multi-utterance transcript summary, and one rolling visual delta without an
artifact outside its temporary directory.

Corner-case probes cover atomic identification readiness, irrelevant retrieval
results, absent or unitless temperature readings, unconfirmed immersion, a running timer, and
voice questions that require fresh vision or timer tools. These fixtures use
different labels and readings from the prompt rules so they remain behavioral
checks rather than worked-example recall. A repeated below-target reading also
guards against routine observation messages being spoken every cycle.
Identification probes include a visible variety paired with a different
retrieved variety; that mismatch must remain unready and commit no values.
The contract probe checks that prior completion status has a distinct name and
that field meanings and completion values remain visible to the small model.
The progress probe requires a brief message for a real non-completing state
change while the repeated-reading probe suppresses even attempted narration.
Retrieval chunk
settings are pinned to the compact original configuration.
