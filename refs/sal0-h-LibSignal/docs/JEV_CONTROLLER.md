# Jev controller through OpenRouter

`--agent jev` runs `typesafe/jev-1.13` using OpenRouter's native Decisions API.
It translates this repository's Traffic-R1 prompt into structured state and one
Choice question per intersection. It is an inference-only simulation controller.

## Run

From the repository root, activate your environment and install the two HTTP
integration dependencies if needed:

```bash
source .venv/bin/activate
uv pip install --python .venv/bin/python requests python-dotenv
python run.py --agent jev --world sumo --network sumo4x4 --seed 42 --ngpu -1 --prefix jev_l1
```

With a pip-equipped environment, `python -m pip install requests python-dotenv`
works too. The project requirements include both packages. No OpenAI or TypeSafe
SDK, local model weights, or inference GPU is needed for Jev.

Set `OPENROUTER_API_KEY` in the repository's ignored `.env`, or export it in your
shell. A nonempty environment value takes precedence. The client reads only this
key from `.env` without executing the file or loading unrelated settings into the
environment. Credentials never enter the YAML, decision logs, or request body.

The default YAML matches `traffic_r1.yml`: the grid4x4 hub-OD held-out route set,
1,800 simulation steps per route, and a decision every 10 steps. Use `sumo4x4`
with this config; its demand files belong to that network. Other networks need
matching demand settings and compatible four-movement phase mappings.

For a bounded integration check first, use the 60-step smoke configuration:

```bash
python run.py --agent jev_smoke --world sumo --network sumo4x4 --seed 42 --ngpu -1 --prefix jev_smoke
```

This runs six decision steps (96 requests for the 16 interior intersections),
using the same controller. Its output directory starts with `sumo_jev_smoke`.
The short rollout is not a traffic-performance benchmark.

## Code path

| File | Responsibility |
| --- | --- |
| `agent/jev.py` | Registered `JEVAgent`, SUMO actions, concurrent requests, decision logs. |
| `agent/jev_prompt.py` | Pure function translating the traffic observation to `state` and `questions`. |
| `agent/jev_backend.py` | OpenRouter HTTP request, bounded transport retries, typed-answer validation. |
| `configs/tsc/jev.yml` | Model and evaluation settings. |

`JEVAgent` inherits `LLMTSCAgent` to reuse its observation generators, lane
geometry, sensor-noise handling, and signal-to-green-index mapping. It overrides
the action path; inherited chat prompting, backends, and text parsing are never
called. The existing Traffic-R1 implementation is unchanged.

The main thread collects each junction's observations before starting HTTP
workers, because libsumo reads must not run concurrently. Each active junction
has a lazily created client and its own persistent connection pool. Requests run
concurrently up to `parallelism`, and their results retain agent order.
Single-phase fringe junctions keep action zero without an API call.

## Prompt translation

The source is `agent/llm_tsc_prompt.py`, which implements the Traffic-R1 Appendix
A.1 observation vocabulary. The translation preserves these inputs:

| Traffic-R1 content | Jev representation |
| --- | --- |
| Four controlled movements | Choice criteria: `ETWT`, `ELWL`, `NTST`, `NLSL`, with their lane descriptions. |
| Early queued counts on both approaches | `state.signals[signal].approaches[side].early_queued`. |
| Three approaching-vehicle segments | `segments`, ordered nearest to farthest, excluding early queued vehicles. |
| Counts summed across both approaches | `totals`, calculated in Python. |
| Select the next green signal | One `next_signal` question asking which signal should reduce waiting. |

The question makes the control objective explicit, including choosing a signal
when all counts are zero. It adds no hand-written queue weighting, new sensor
features, phase history, or current-phase preference. Segment assignment, stop
speed, observation distance, and noisy-count reconciliation remain inherited
from Traffic-R1. The spatial bins use lane length, not estimated arrival times.

Chat roles, internal-monologue requests, `<think>` tags, boxed-answer formatting,
output token limits, temperature, and text-repair retries are omitted. Right
turns are described as outside the four controlled choices rather than making
a promise about their simulator permissions. Jev receives a bounded decision
instead of a request to generate an explanation. This follows TypeSafe's
[Choice](https://docs.typesafe.ai/primitives/choice) and
[structured-state](https://docs.typesafe.ai/concepts/state) interfaces.

Jev 1.13 has documented weaknesses in arithmetic and numeric precision. Counts
and totals are therefore computed before the request; the model only selects a
signal. This avoids asking it to count vehicles, but does not establish that its
traffic judgments outperform Traffic-R1. See the
[model limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13).

## API and failure behavior

The client sends `model`, `state`, and `questions` to
`POST https://openrouter.ai/api/alpha/decisions`, authenticated with the OpenRouter
key. It consumes `answers.next_signal.choice` and verifies that it is an allowed
signal before mapping it to a SUMO green-phase index. The response's resolved
model must match the requested ID or a dated build of it. The request remains
`typesafe/jev-1.13`; it does not switch to `jev-latest`.

Confidence and probabilities are retained when supplied. OpenRouter's schema
makes these fields optional, so missing values are recorded as `null`. A valid
Choice controls the signal at every confidence level. There is no confidence
threshold, probability sampling, or fallback to MaxPressure. TypeSafe describes
confidence as a statistic of the returned probability distribution; it is not a
traffic-safety guarantee. See [confidence](https://docs.typesafe.ai/confidence).
Live OpenRouter responses round probabilities to two decimal places. Validation
allows that rounding error and logs the original values without renormalizing.
`--seed` seeds the simulation; the Decisions request has no sampling-seed field,
and repeated requests can return slightly different distributions.

Timeouts, connection failures, and temporary HTTP statuses (408, 429, 500, 502,
503, 504, 524, 529) receive at most `request_attempts` attempts, including the
initial request. Defaults are three attempts and a 30-second HTTP timeout.
Retries wait exponentially, or honor `Retry-After` seconds/date values. A server
delay over 30 seconds ends the run with an error. Authentication, credit,
configuration, redirect, malformed JSON, and invalid-answer errors stop
immediately. A failed batch produces no simulator step or heuristic action.
Retries after a timeout can incur additional billed requests.

The native endpoint is currently an alpha API. Its contract was checked against
the [OpenRouter API reference](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-questions-and-answers-request)
and [OpenAPI schema](https://openrouter.ai/openapi.json) on 2026-09-20.

## Logs and validation

`log_decisions: true` appends one JSON record per successful active-junction
decision to:

```text
data/output_data/tsc/sumo_jev/sumo4x4/<prefix>/logger/jev_decisions.jsonl
```

Each record contains the intersection, simulation time, signal, green index,
confidence, probabilities, resolved model, request ID, successful response usage
and cost (when supplied), elapsed request time, and attempt count. Standard
LibSignal travel-time and trip-metric outputs are unchanged. Use a new prefix
for a separate experiment; the decision file is append-only.
Records are written after the entire batch succeeds, so billed requests from a
failed batch or a timed-out attempt may not appear in these logs.

Offline regression checks:

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
python -m compileall -q agent world trainer task common utils generator dataset run.py environment.py
```

`tests/test_jev.py` mocks HTTP requests and covers observation preservation,
typed-answer validation, credential precedence, retries, invalid responses,
concurrent action ordering, fringe junctions, confidence logging, and matching
Traffic-R1 evaluation settings. It never needs the real API key.

Verified on 2026-09-20: all 49 repository tests and the compile check passed.
The 60-step live SUMO check logged 96 decisions across 16 interior intersections
with `typesafe/jev-1.13-20260917`, all on the first attempt. The successful requests
reported a combined cost of $0.003620736. Five distributions summed to 0.99 after
provider rounding, confirming the need for rounding tolerance. The full held-out
traffic-performance evaluation has not been run for Jev.

Additional primary sources consulted: TypeSafe's
[API contract](https://docs.typesafe.ai/api),
[workflow design](https://docs.typesafe.ai/concepts/how-to-build-with-system-one),
[structured questions](https://docs.typesafe.ai/primitives/advanced),
[composite scoring](https://docs.typesafe.ai/patterns/composite-scoring), and
[model reference](https://docs.typesafe.ai/models), plus the
[OpenRouter model listing](https://openrouter.ai/typesafe/jev-1.13).
