# TIDE — Temporal Inference of Dependencies in Agent Swarms
**Status:** Research prototype; current results are exploratory (9 November 2026).
**Track:** Primarily Track 4 — Multi-Agent Oversight and Control; related to Track 2 — The Science of Agent Networks.

> **Project summary:** TIDE explores whether potential dependencies between actions in a multi-agent system can be reconstructed from timestamped activity logs. It retrieves earlier candidate events, scores their possible contribution to later events, and compares private-context inference with public-summary inference and explicit references. These outputs are exploratory evidence, not proof of communication, collaboration, or causation.

---

## Current status

Implemented components include:

- AI Village slice construction, schema auditing, and provenance-preserving event loading.
- Pass 1 remote summaries with explicit `from_agent` and `to_agent` annotations.
- Private-context and public-summary Pass 2 embedding, candidate retrieval, and reranking.
- Event and agent graph analysis, including direct-mention predecessor proxies and community statistics.
- Public-score calibration against matched private scores, with sigmoid-only and sigmoid-plus-temporal-decay variants.
- Reference-relative network evaluation, event-level mention-proxy evaluation, and synthetic benchmarks.

Current real-data comparisons are exploratory: references are incomplete, Pass 2
candidate filters differ by context, and model scores are not ground-truth
probabilities. The latest snapshot is summarized below; generated reports,
metrics, and figures remain local outputs and are not committed. Results are
reference-relative, not proof that inferred edges are true dependencies.

### Latest evaluation snapshot (9 November 2026)

The comparison uses 35 registered authors from a 1,200-event cohort. Calibration
uses the first 200 targets; evaluation uses later jointly completed targets
(990/1,000 in this snapshot). Agent-pair scores use the maximum event-pair
score. Self-links are excluded; inferred edges use threshold 0.5. The public
calibration fits a ranking-only sigmoid and a sigmoid-plus-time-decay model on
957 matched training pairs, then evaluates both on 3,977 matched holdout pairs.

| Agent network | Edges | Density | Communities | Modularity |
|---|---:|---:|---:|---:|
| Direct chat mentions | 71 | 0.060 | 13 | 0.350 |
| Private-context inference | 104 | 0.087 | 8 | 0.294 |
| Public TIDE | 114 | 0.096 | 11 | 0.286 |
| Public TIDE (no temporal) | 152 | 0.128 | 7 | 0.248 |

| Reference comparison | Public variant | Precision | Recall | F1 | AP |
|---|---|---:|---:|---:|---:|
| Direct mentions | Public TIDE | 0.500 | 0.803 | 0.616 | 0.636 |
| Direct mentions | Public TIDE (no temporal) | 0.388 | 0.831 | 0.529 | 0.461 |
| Private inference | Public TIDE | 0.649 | 0.712 | 0.679 | 0.723 |
| Private inference | Public TIDE (no temporal) | 0.572 | 0.837 | 0.680 | 0.607 |

On matched event-pair holdout scores, sigmoid-plus-decay yields MSE 0.0857
and MAE 0.2249; sigmoid-only yields MSE 0.0906 and MAE 0.2384. At threshold
0.5 their precision/recall are 0.809/0.502 and 0.776/0.474, respectively.
This favors decay on this snapshot, not as a general conclusion: calibration
targets are limited and private scores are only a reference.

The event-level mention proxy resolves 234 of 242 target/mentioned-agent pairs;
8 mentions have no earlier cohort event. Of 990 holdout targets, 205 (20.7%)
have at least one proxy link. Event-level F1 is 0.105 and AP is 0.071 across
693,766 eligible earlier-event/target pairs; 105 of 234 reference links had a
TIDE candidate score. This proxy is based on the closest prior event by a
mentioned agent, not a verified information source.

| Generated figure (local, not committed) | Contents |
|---|---|
| `communities.png` / `.pdf` | Four agent networks in one shared 2x2 layout; community colors are maximum-overlap aligned. |
| `community-overlap.png` / `.pdf` | Three community membership contingency heatmaps. |
| `precision_recall.png` / `.pdf` | Both public variants against both references, with AP and prevalence baselines. |

Recreate the current analysis with `evaluate-networks`; use a new empty output
directory. Exact snapshot hashes, event-level metrics, threshold sweeps, and
community stability are emitted in `metrics.json` in that output directory.
Longer-term research directions are collected at the end under
[Future roadmap](#future-roadmap).

## Quickstart

```bash
uv run python -m tide.cli run-synthetic --output-dir outputs/synth --seed 7
uv run python -m unittest discover -s tests -p "test_*.py"
```

The synthetic command writes a reproducible offline example, including:

- `events.jsonl`
- `ground_truth_edges.jsonl`
- `retrieval_candidates.jsonl`
- `support_edges.jsonl`
- `event_graph.json`
- `agent_matrix.json`
- `summary.json`

## AI Village usage notes

This code supports schema auditing and conservative normalization for real traces when available, but does not attempt to bypass gated access.

Before using AI Village data:

- obtain access approval via Hugging Face;
- re-check current dataset docs/changelog;
- follow licensing and privacy constraints;
- treat generated summaries as convenience signals, not ground truth for computer-use actions.

## Pass 1: remote annotation

Pass 1 is implemented in `tide.pass1`. Its system prompt includes the canonical
agent display names and their model-string aliases, plus an example of an
anonymized summary and distinct recipient/source references. The model returns
only `summary`, `to_agent`, and `from_agent`; the pipeline attaches the source
`event_id` locally. References must be exact, unique names from `agents.jsonl`,
not UUIDs. Model output is validated structurally, but the factual correctness,
anonymity, and evidence behind references still require review.

The pipeline uses the official Adaption Python SDK and its Responses API.
Each event is an independent request with tools, agentic memory and response
storage disabled. The prompt requests JSON and the pipeline validates it; it does
not use an API-enforced JSON mode. Confirm your data policy and rate/concurrency
limits before running; provider-side batching is not implemented. Adaption's
documented Responses API does not expose a batch endpoint; `--concurrency` sends
separate requests concurrently. No model is run locally.

Pass 1 skips `CONSOLIDATE` events; their source records are unchanged. Only action
type, registered actor name, and `AGENT_TALK`/`USER_TALK` message
content are sent. Raw model outputs (including reasoning and tool calls), goals,
and other metadata are excluded. Public messages may themselves contain sensitive
information: this code does not automatically detect or redact it. Review the
data and provider's handling requirements before transmitting it. Events without
public chat will often have insufficient context for informative summaries or
references; missing information must not be inferred. Pass 2 is not implemented
by this command.

Set `ADAPTION_API_KEY` in a local `.env` file (which is gitignored), then run a
small pilot with the default model, `deepseek-v4-flash`:

```powershell
uv run --env-file .env python -m tide.cli pass1 --output-dir outputs/pass1 --limit 3
```

Use `--model` to select another model available to your account. Remove `--limit`
to process the full dataset, or change `--concurrency` only
after checking the provider's limits. The output directory
must be outside the master dataset; it contains `annotations.jsonl` and the
append-only checkpoint `pass1_checkpoint.jsonl` and `pass1_config.json`. Invalid
JSON and invalid annotation fields get up to two additional correction requests.
Events still failing after retries are recorded in `pass1_errors.jsonl` without
stopping other events. The error file contains event IDs, source indexes and
error messages, not model responses. The command exits nonzero when the run has
failures. Repeat the same command without `--overwrite` to retry missing events;
successful events are not requested again and resolved errors are removed. A
previously interrupted run also continues with its unattempted events. Use
`--retry-errors-only` to retry only recorded errors, without processing other
unfinished events; it cannot be combined with `--overwrite`. The checkpoint fingerprints
the input files, prompt, SDK version, endpoint, and model; changed configurations
require a new output directory, including runs made before consolidation filtering.
Alternatively, `--overwrite` starts a fresh run by deleting the existing Pass 1
configuration, checkpoint, errors, and annotations in the output directory; unrelated
files are left untouched. This discards saved progress and does not resume.
`--limit` selects the first N non-consolidation events, so
the pilot can be resumed without it. A forcibly terminated process may leave a
`pass1.lock`; remove it only after confirming the process is no longer running.
Checkpoints survive interruption. During processing, records are appended in
completion order; when a run finishes, both the checkpoint and annotations are
ordered by source event position. Deterministic temperature does not guarantee identical remote model
results. The progress bar shows attempted eligible events (including failed
ones), the failure count, processing time and an estimated remaining time; the
final `elapsed_seconds` includes the initial
dataset check. The initial dataset remains unchanged.

## Pass 2: temporal retrieval and reranking

Pass 2 reads the processed event slice and Pass 1 annotations without changing
either input. For each non-`CONSOLIDATE` event, it builds a private
representation from the agent's active goal, original event message/content,
and available thinking/reasoning trace. The anonymized Pass 1 summary is not
used. Pass 1 `from_agent`/`to_agent` values are used locally only for candidate
selection; the reference values are not included in embedding text or the
DeepSeek request. Pass 1 annotations also identify target events that were
successfully annotated.

Pass 2 creates separate query/document embeddings with
`jinaai/jina-embeddings-v5-text-nano` and retrieval prompts, then combines up to
three recent same-agent messages, five messages from annotated information
sources, five messages annotated as incoming to the target author, and ten
semantically similar prior events from other agents within 12 hours. Candidates
are deduplicated; timestamps must be strictly earlier than the target. DeepSeek
assesses the full representations and goal context; retrieval similarity and
the local retrieval-source labels are stored separately from dependency scores.
Consolidation events are excluded from targets, embeddings, and candidate
history. The model license is CC BY-NC 4.0; check usage rights and data-handling
requirements before downloading or transmitting private traces.

Install the optional local embedding dependencies. To cache and run each stage
independently, use the same output directory for all three commands:

```powershell
uv sync --extra pass2
uv run python -m tide.cli pass2-embed --dataset-dir data/processed/ai-village/main-village/2026-10-24_to_2026-10-31/v2 --output-dir outputs/pass2-deepseek
uv run python -m tide.cli pass2-retrieve --dataset-dir data/processed/ai-village/main-village/2026-10-24_to_2026-10-31/v2 --pass1-dir outputs/pass1-deepseek-filtered-v2 --output-dir outputs/pass2-deepseek
uv run --env-file .env python -m tide.cli pass2-rerank --dataset-dir data/processed/ai-village/main-village/2026-10-24_to_2026-10-31/v2 --output-dir outputs/pass2-deepseek --timeout 240 --retries 5 --concurrency 4 --candidate-group-size 4
```

The stages cache `event_embeddings.jsonl`, `retrieval_candidates.jsonl`, and
`dependencies.jsonl`, respectively, with separate checkpoints, errors, and
configuration fingerprints. Repeat a stage command to resume it. Retrieval
requires a complete embedding cache; reranking requires the retrieval cache.
You can instead run `pass2` once to execute all three stages in order using
these same caches. Use `--limit N` on `pass2-retrieve` to cache candidates for
only the first N annotated targets. If you change an upstream stage's inputs or
settings, rerun that stage with `--overwrite`, then rerun each downstream stage
with `--overwrite` so it consumes the refreshed cache.

Embedding precision defaults to `auto`: FP16 on CUDA and FP32 on CPU. Use
`--precision float32` to compare against full precision, or `--precision
float16` to require CUDA FP16 explicitly. Precision is recorded in the embedding
cache configuration; changing it requires `--overwrite`. To restart a partial
embedding run under CUDA FP16, for example:

```powershell
uv run python -m tide.cli pass2-embed --dataset-dir data/processed/ai-village/main-village/2026-10-24_to_2026-10-31/v2 --output-dir outputs/pass2-deepseek --batch-size 32 --precision auto --overwrite
```

The default embedding revision is pinned to a Hugging Face commit; use
`--embedding-revision` only when intentionally changing model revisions.
All artifacts are written only under the Pass 2 output directory. The API key
is needed only for reranking and uses `ADAPTION_API_KEY` by default.
Reranking concurrency defaults to one request; increase `--concurrency` only
within your provider's rate and concurrency limits. It can be adjusted when
resuming without invalidating the reranking cache. Each target is checkpointed
as it completes, and final dependencies are ordered by source event.
Use `--limit N` on either rerank command to process only the first N cached
targets in chronological order. Existing checkpointed targets are reused, even
when they are outside that limit; the dependencies output retains all completed
targets. Changing the limit does not require `--overwrite`.
By default, each target's candidates are sent in groups of four, with the target
included in each request; group scores are merged into one target result. Change
`--candidate-group-size` to tune prompt sizes. Changing it requires reranking
with `--overwrite`; this resets only rerank outputs and keeps the embedding and
retrieval caches.

Reranking also accepts `--candidate-filter proposed`. It filters, but does not
rewrite, the cached retrieval candidate lists: keep the same-agent history,
latest cached candidate per `from_agent` author, cached incoming messages after
the latest same-agent action, then up to five top semantic candidates not
already selected. The rerank summary reports candidate counts before and after
filtering. Switching filters requires reranking with `--overwrite`; the cached
`retrieval_candidates.jsonl` remains unchanged.

### Public-context experiment

The public-context stages use only the Pass 1 `summary` for event text. They do
not read agent goals, original event content, model output/thinking, or Pass 1
`from_agent`/`to_agent` values. Because those references are private, public
retrieval omits the identified-source and incoming-communication candidate
pools; it retains same-agent history and dense semantic candidates, the same
12-hour strict-previous-event window, deduplication, and scoring. This is the
necessary candidate-generation difference between the private and public
experiments.

Run the public stages into a separate directory; these commands do not touch
`outputs/pass2-deepseek`:

```powershell
uv run python -m tide.cli pass2-public-embed --dataset-dir data/processed/ai-village/main-village/2026-10-24_to_2026-10-31/v2 --pass1-dir outputs/pass1-deepseek-filtered-v2 --output-dir outputs/pass2-public --batch-size 16
uv run python -m tide.cli pass2-public-retrieve --dataset-dir data/processed/ai-village/main-village/2026-10-24_to_2026-10-31/v2 --pass1-dir outputs/pass1-deepseek-filtered-v2 --output-dir outputs/pass2-public
uv run --env-file .env python -m tide.cli pass2-public-rerank --dataset-dir data/processed/ai-village/main-village/2026-10-24_to_2026-10-31/v2 --pass1-dir outputs/pass1-deepseek-filtered-v2 --output-dir outputs/pass2-public --timeout 240 --retries 5 --concurrency 16 --candidate-group-size 4 --candidate-filter top10 --overwrite
```

`--candidate-filter top10` keeps at most ten candidates already in each target's
retrieval list; it never changes `retrieval_candidates.jsonl`. In public mode it
first preserves the cached same-agent history (up to three events), then fills
the remaining slots by descending similarity. In private mode it selects the
ten highest-similarity candidates. The rerank summary reports counts before and
after filtering. `--overwrite` resets only public rerank outputs, not the public
embedding or retrieval caches.

The public caches have separate configuration fingerprints and are marked
`context_mode: public`. Public and private stages refuse to share or overwrite
each other's output directory. Resume each public stage by repeating its
command; use `--overwrite` only to restart that public stage's own cache.

## Run tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Executive summary: extended synthetic validation

`python -m tide.cli run-extended-validation --output-dir outputs/extended_validation --report-path reports/extended_synthetic_validation.md --seeds 7,13,29 --runs-per-seed 3`

Current synthetic findings:

- Retrieval recall is high on this benchmark (near-or-at 1.0 for semantic variants), but this likely reflects that synthetic signals are still relatively easy.
- Full pipeline shows **moderate precision** and **high recall** (`precision ≈ 0.62`, `recall ≈ 0.94`, `F1 ≈ 0.74`, `AUPRC ≈ 0.80`).
- Negative control performance is mixed: shared-external non-ENV false positives are low, but independent-identical-task false positives remain non-trivial (`FPR ≈ 0.33`).
- Indirect chain handling improved (`incorrect_direct_A_to_C_rate = 0.0`), but delayed-scenario precision remains low and some redundant direct-edge attributions persist.

See `reports/extended_synthetic_validation.md` and `outputs/extended_validation/*.jsonl` for details and reproducible results.

## Hard-negative benchmark update

Running `python -m tide.cli run-fp-forensics --output-dir outputs/fp_forensics --seeds 7,13,29 --eval-seeds 101,103,107 --runs-per-seed 3` adds harder independent-agent and harder transfer controls.

Key held-out takeaway:

- At threshold 0.5, recall remains high but independent-agent false positives rise on harder controls.
- Raising threshold to 0.7 reduces independent FPR substantially, but loses broker sensitivity.
- No single current profile dominates across all objectives; this remains an open trade-off.

## Research vision and design reference

The following sections preserve the project's original research framing. They
describe hypotheses and possible methods; they are not claims that those
experiments or metrics are implemented. The current implementation and
observed results are described above and in the operational sections below.

### Research question

**Can we infer the probabilistic dependency structure between events in a multi-agent system from timestamped agent activity traces, and use it to identify cross-agent information flow and functional collective behaviour when private chain-of-thought is unavailable and some observations are hidden?**

The starting point is a collection of agents with individual goals, acting over time and interacting directly (e.g. messages) or indirectly (e.g. shared files, databases, memory or tools). We assume access to observable action records and, depending on the experimental condition, agent outputs and tool results. We do not require private chain-of-thought.

For each event, we record the agent, timestamp, goal/context, state where observable, action, tool, input, result and public output. For every new event, we search for a small set of relevant earlier events—both from the same agent and from other agents—and estimate whether each predecessor provides evidence supporting the new event.

The main output is a **probabilistic event-dependency graph**. An agent-level network is derived by aggregating event dependencies over time.

### Hypotheses

#### H1 — Recoverable event dependencies

A hybrid retrieval method combining semantic similarity, temporal information and resource/artifact relationships can retrieve supporting predecessor events more effectively than semantic similarity or temporal proximity alone.

#### H2 — Cross-agent dependencies beyond individual history

For some episodes, the history of other agents provides predictive information about an agent's subsequent actions beyond that agent's own history and observable shared task/environment context.

#### H3 — Observable information provenance supports network reconstruction

Combining event-level support estimation with explicit resource and communication evidence can recover meaningful information-flow paths, including multi-hop paths through intermediate agents.

#### H4 — The method can distinguish common-cause similarity from cross-agent dependency

When agents receive identical tasks but act independently, or respond to the same external event without communicating, the method should assign substantially weaker cross-agent support than in controlled information-transfer conditions.

#### H5 — Partial observability reduces but need not eliminate recoverability

Hiding message content, agent outputs, selected agents or selected events will reduce reconstruction quality. The rate and pattern of degradation can reveal which observable signals are most useful.

#### Exploratory hypothesis — Brokers and aggregators

Agents that receive information from several sources and subsequently relay, transform or combine it may be identifiable from time-respecting event paths and support relationships.

These hypotheses are provisional. The initial prototype should prioritise H1 and H4, then investigate H2/H3 if the trace quality permits.

### What TIDE measures (and what it does not)

The project separates four related concepts:

| Concept | Meaning | Candidate measurement |
|---|---|---|
| Event support | An earlier event contains information or a result that supports a later action | Pairwise support score / calibrated probability |
| Predictive dependence | One agent's past improves prediction of another agent's future beyond observed context | Conditional transfer entropy or held-out log-likelihood improvement |
| Collective synergy | Multiple agents jointly contain predictive information about a future joint/group outcome not available from any one agent alone | TDMI + partial information decomposition (PID) |
| Functional coordination | Agents make interdependent or complementary contributions to a related task/outcome | Evidence combining event provenance, predictive dependence, complementary roles and outcome relevance |

These should not be collapsed into a single unqualified “collaboration probability”. Similarity is not influence; temporal correlation is not communication; predictive dependence is not automatically causation; information transfer is not necessarily cooperation. An agent can mislead or obstruct another agent. The initial project therefore uses the cautious phrase **functional cross-agent coordination** rather than inferring intent.

### Event representation

Each observable event is represented with fields such as:

| Field | Description |
|---|---|
| `event_id` | Stable unique event identifier |
| `episode_id` | Village, session or task identifier |
| `agent_id` | Stable agent identity from the source |
| `timestamp` | Event timestamp (normalized to UTC, original retained) |
| `goal` | Agent goal and/or shared goal, where available |
| `subgoal` | Current subgoal, only if explicitly recorded or marked as inferred |
| `state_before` | Relevant pre-action state, where observable |
| `action_type` | Normalized action category |
| `action_summary` | Short factual description of the action |
| `tool` | Tool or resource invoked |
| `input` | Arguments or relevant input |
| `result` | Tool result or observable action outcome |
| `agent_output` | Public/observable output; not private chain-of-thought |
| `artifact_refs` | File, URL, database key, shared memory or resource identifiers |
| `source_ref` | Source table and source-record identifier |
| `observability_flags` | Which fields are present in this observation condition |

The result of an action must be distinguished from the information available before that action. An action's result can support a later event, not retroactively explain the choice of the action that produced it.

Missing values must remain missing. TIDE must not invent unobserved goals, states, recipients, tool results or artifact links. The source records remain immutable; inferred predecessor relationships are stored separately.

### Method

#### Candidate predecessor retrieval

For each target event `e_k`, retrieve a bounded set of earlier events `C_k` with `timestamp(e_j) < timestamp(e_k)`. Retrieval is restricted to the same episode unless an explicit cross-episode relationship is justified.

Candidate generation combines:

1. **Same-agent history:** always retain a few recent events from the target agent, even if semantic similarity is low.
2. **Semantic retrieval:** use embeddings of factual event descriptions (goal, action, tool, input and available result/output).
3. **Temporal proximity:** use one or more configurable windows or temporal decay.
4. **Resource/artifact overlap:** prioritize exact links such as a file produced by one event and read by another, or a value returned by one tool and used in another call.
5. **Task context:** include agent goals, shared instructions and environment state as explanatory context, but do not let a shared goal alone create an edge.

A possible ranking function is:

\[
R(e_j,e_k)=\alpha s_{\mathrm{semantic}}+\beta s_{\mathrm{temporal}}+\gamma s_{\mathrm{resource}}+\delta s_{\mathrm{compatibility}}.
\]

This is a **retrieval score**, not a probability of influence. The initial candidate set should contain roughly 5–10 cross-agent events and 2–3 same-agent events, configurable by experiment. Query rewriting is a possible later ablation, not an MVP requirement.

#### Event-support estimation

For every candidate predecessor, estimate:

\[
p_{jk}=P(e_j\text{ supports }e_k\mid e_j,e_k,\mathcal{H}),
\]

where \(\mathcal{H}\) is the observable historical context. This is initially an inferred support relationship, not a proven causal probability.

Possible support categories:

- `direct`: explicit information in a message/output is used downstream;
- `artifact-mediated`: one event produces or modifies a resource another event reads/uses;
- `information-content`: a specific value/fact appears in a later input/action;
- `contextual`: an earlier event changes the task/environment context;
- `independent`: the action is adequately explained by the agent's own task/history;
- `ambiguous`: there is plausible but insufficient evidence;
- `none`: no meaningful support evidence.

Multiple earlier events may support one event. Candidate probabilities therefore do not need to sum to one. The method must permit abstention and provide concrete evidence for each inferred edge. A raw LLM-judge score must not be called a calibrated probability until assessed against labelled examples (e.g. Brier score and calibration curves).

#### Event graph

Construct a directed graph whose nodes are events and whose edges run from an earlier supporting event to a later supported event:

\[
G_E=(E,\mathcal{A}),\qquad e_j\rightarrow e_k\text{ if }t_j<t_k.
\]

Edges should record support score/probability, support type, time lag, retrieval score, evidence, alternative explanations, and source references. Preserve direct edges and multi-hop paths separately.

#### Agent-level dynamic network

Aggregate event dependencies into a time-varying directed matrix. Use the convention:

\[
W_{ij}(t)=\text{evidence that events by agent }j\text{ support events by agent }i\text{ in window }t.
\]

This convention means the source-to-recipient edge is `j → i`. The matrix need not be symmetric, and its rows should not automatically be normalized to sum to one: several prior events can support one target event.

Aggregation can use counts, sums, mean support per target event, unique predecessor counts or time-decayed sums. Report raw and normalized versions separately where useful.

#### Directed predictive dependence

As an optional statistical measure, estimate whether agent A's past improves prediction of agent B's next action after conditioning on B's own history and observable common context:

\[
TE_{A\rightarrow B\mid Z}
=I(A_{\mathrm{past}};B_{\mathrm{future}}\mid B_{\mathrm{past}},Z_{\mathrm{past}}),
\]

where \(Z\) may include common task instructions, shared goals, public feedback and observable environment state. For irregular event logs, a held-out predictive log-likelihood comparison may be more practical than a conventional time-series estimator.

A positive value indicates residual predictive dependence under the chosen conditioning set; it does not establish direct communication or causal influence when important common causes are unobserved.

#### TDMI and partial information decomposition

Riedl (2026) provides a complementary measure of higher-order temporal coordination. For two agents, let current states be \(X_{i,t},X_{j,t}\), and their joint future state be \(T_{ij,t+\ell}=(X_{i,t+\ell},X_{j,t+\ell})\). A two-source PID decomposes:

\[
I(\{X_{i,t},X_{j,t}\};T_{ij,t+\ell})
=UI_i+UI_j+Red_{ij}+Syn_{ij}.
\]

The synergy term represents predictive information about the joint future available only when considering both agents under the selected PID definition. It is symmetric in the two agents and is **not** a directed event-support probability.

Riedl also defines a macro-level criterion:

\[
S_{\mathrm{macro}}(\ell)
=I(V_t;V_{t+\ell})-\sum_i I(X_{i,t};V_{t+\ell}),
\]

where \(V_t\) is a macrostate or group-level signal. A positive value indicates macro self-predictability beyond the sum of individual predictive contributions under this criterion; redundancy can make the measure negative even if some synergy exists.

The triplet/coalition measure is:

\[
G_3=I((X_i,X_j,X_k)_t;V_{t+\ell})
-\max_{(a,b)\in\{(i,j),(i,k),(j,k)\}}I((X_a,X_b)_t;V_{t+\ell}).
\]

These statistics require a meaningful representation of synchronized agent states and a group-level target. The raw logs are asynchronous and high-dimensional, so PID should be treated as an optional extension using carefully defined time bins and low-dimensional/discretized features. It should not be forced into the MVP.

### Key scientific challenge: independent agents with the same task

Two agents can perform almost identical action sequences without communicating. Shared goals, prompts, tools and environmental events are common causes of behavioural similarity.

TIDE must include explicit controls:

#### Negative control 1 — identical task, independent execution

Give two or more agents the same model, prompt, task, tools and environment, but isolate them from one another. They may independently search for the same file, read the same content and produce similar outputs. The detector should not label this as strong cross-agent information transfer simply because the actions are similar.

#### Negative control 2 — shared external event

Agents do not communicate, but both react to the same external event or public environment change. The detector should condition on the observed common event/context and avoid inventing a direct agent-to-agent edge.

#### Positive control 1 — explicit information transfer

Agent A discovers a fact and sends it to B. The full-log condition contains the message; the message-blind condition hides its content from the detector.

#### Positive control 2 — shared resource channel

Agent A writes a fact to a shared file/database; B reads and uses it. No explicit message is necessary.

#### Positive control 3 — broker/aggregator

Agents A and B provide complementary facts to C. C combines them and produces a downstream result. Test whether C is identified as an aggregator or intermediary.

Additional conditions can include irrelevant messages, misleading information, two-hop relays, hidden agents and missing events.

### Candidate datasets

#### AI Village

Dataset: [AI Village on Hugging Face](https://huggingface.co/datasets/aidigestorg/ai-village)

Current dataset documentation describes a lightly processed export containing:

- approximately 233k structured events (`events.jsonl.gz`);
- approximately 123k chat messages (`chat_messages.jsonl.gz`);
- approximately 37k computer-use sessions (`computer_use_sessions.jsonl.gz`);
- approximately 1.14M computer-use turns (`computer_use_turns.jsonl.gz`);
- approximately 165k agent memories (`agent_memories.jsonl.gz`);
- 31 agent records in the main village (`agents.jsonl.gz`);
- agent goals, village goals, summaries and metadata.

These figures are approximate and should be rechecked at the point of access. The dataset includes several separate villages. Start with one coherent village/time slice; join by stable agent IDs, not names. Read the dataset changelog before interpreting changes over time because the surrounding scaffolding, prompts, tools, memory and model rosters changed.

**Access and use restrictions:** access is manually gated. The current research terms permit research/analysis but prohibit training or fine-tuning AI systems on the data without written permission. They also prohibit re-identification and require citation of AI Digest / AI Village and notification of resulting publications. Do not bypass the gate and do not train on this data without permission. Daily summaries are documented as secondary and potentially inaccurate because they do not inspect the computer-use sessions.

#### Synthetic controlled traces

A small synthetic generator is essential because real communication logs are weak ground truth: a message being sent does not mean it was read or used. Synthetic traces allow known support edges, independent agents and common causes to be specified exactly.

If AI Village access is not available in time, develop and evaluate the pipeline on synthetic traces, and state clearly that real-world validation remains future work.

#### Swarm Traces

[Swarm Traces](https://swarmtraces.org/) is relevant to the Hugging Face incident, but it should only be used for quantitative event-graph evaluation if the available records contain reliable timestamps, actor/agent attribution and defensible event linkage. Reconstructed payload relationships are not automatically proof of temporal or causal edges. If metadata is inadequate, use it only as a qualitative forensic case study.

### Proposed experiments and ablations

The experiment suite progressively removes information from the detector.

| Condition | What the detector sees | Purpose |
|---|---|---|
| E0: Full observable traces | Goals, observable state, actions, inputs, results, public outputs and available message contents; no private CoT | Main reconstruction condition |
| E1: Message-blind | Same action/result observations, but no inter-agent message content | Test whether dependencies survive without explicit message text |
| E1b: Metadata-only messages | Message content hidden, but sender/room/timestamp metadata retained if available | Separate content from interaction metadata |
| E2: Result/output hidden | Remove agent public outputs and/or tool results while retaining action/tool/input fields | Measure dependence on returned information |
| E3: Tool-call focused | Tool name, arguments and timestamp; remove natural-language output/result text | Action-only observability stress test |
| E4: Partial agent visibility | Hide selected agents | Test missing actors |
| E5: Partial event visibility | Remove a fraction of events or selected action types | Test incomplete logs |

No-CoT is the default in every condition, not an experimental treatment. The meaning of “action-only” must be specified exactly; tool results and arguments should not be silently mixed into that label.

Retrieval ablations:

1. semantic similarity only;
2. semantic + temporal score;
3. semantic + temporal + exact resource/artifact matching;
4. same-agent recent-history candidates forced into the candidate set;
5. query rewriting (later only).

Support-estimator ablations:

1. deterministic evidence rules;
2. pairwise LLM judge on retrieved candidates;
3. hybrid rules + LLM judge.

### Proposed measures and evaluation

#### Candidate retrieval

- Recall@K for labelled supporting predecessor events;
- MRR where one principal predecessor is known;
- retrieval latency and candidate-set size.

#### Event-support inference

- precision, recall, F1 and AUPRC for known support edges;
- Brier score and calibration/reliability plots if probabilities are reported;
- abstention coverage and selective accuracy;
- breakdown by direct-message, resource-mediated, information-content, same-agent, ambiguous and no-support cases.

#### Network reconstruction

- event-edge precision/recall/AUPRC against synthetic ground truth;
- direction and time-respecting path recovery;
- agent-edge recovery after aggregation;
- centrality/community analysis only as secondary descriptive analysis.

#### Predictive dependence and collective structure

- conditional transfer entropy or held-out predictive log-likelihood improvement;
- TDMI/PID synergy and optionally triplet `G3` if the time series is sufficiently long and low-dimensional;
- false-positive rate in identical-task independent agents and shared-external-event controls;
- sensitivity to window length, lag, discretization and missingness.

#### Broker/aggregator roles

Look for agents that receive information from multiple sources and subsequently pass on or combine that information. Time-respecting betweenness may be a useful structural feature, but high degree or betweenness alone is not sufficient to label an information broker.

### General limitations

- Observational traces cannot always distinguish direct influence from unobserved common causes.
- A message being sent does not establish that it was read or used.
- A file being read does not establish that its contents affected the next action.
- Similar action sequences may arise independently from identical prompts and tasks.
- Conditional transfer entropy is predictive dependence conditional on measured context, not causal proof.
- TDMI/PID measures temporal synergy and higher-order predictive structure; it does not provide directed event-level causal edges.
- Real-world logs may omit agents, tool results, hidden resources or external context.
- AI Village's undocumented scaffolding and temporal changes complicate behavioural interpretation.
- AI Village's current terms restrict model training/fine-tuning without written permission.
- The system can only infer information dependencies that leave observable consequences; it cannot reliably detect a covert channel that leaves no detectable trace.

### Original MVP milestones (historical)

#### Minimum viable result

1. Audit the repository and dataset access/schema.
2. Normalize a small subset of real traces if access is granted; otherwise use synthetic traces.
3. Build a synthetic generator with known dependencies and same-task/common-cause negative controls.
4. Implement hybrid predecessor retrieval with temporal filtering and same-agent candidates.
5. Implement a support scorer with evidence categories and abstention.
6. Build an event graph and aggregate agent graph/matrix.
7. Evaluate retrieval and edge recovery, with explicit false-positive reporting on independent same-task agents.
8. Run one useful ablation (full observable vs message-blind or result-hidden).
9. Produce at least one interpretable visualization and a concise report.

#### Stretch goals

- Conditional transfer entropy or held-out predictive comparison;
- TDMI/PID on compact synchronized states;
- broker/aggregator identification;
- partial-agent and partial-event robustness;
- Swarm Traces case study after metadata audit.

Do not attempt to process every row in AI Village, train a GNN, or estimate high-order PID over raw text during the weekend.

### Reproducibility principles

- Use stable event IDs, deterministic seeds and explicit configs.
- Stream large JSONL.GZ files; do not load all turns into memory unnecessarily.
- Cache embeddings and model-judge outputs.
- Record model name/version, prompt hash, embedding model, retrieval weights, temporal windows, data slice, missingness and costs.
- Keep raw/gated data out of Git. Add data/cache/secrets to `.gitignore`.
- Make the pipeline work offline on synthetic traces with a rule-based scorer.
- Keep ingestion, normalization, retrieval, support scoring, graph construction and evaluation as separate modules.
- Add tests for temporal filtering, multiple predecessors, missing fields, artifact matching, stable IDs and common-cause controls.

### Research references

- Riedl, C. (2026). [Emergent Coordination in Multi-Agent Language Models](https://arxiv.org/abs/2510.05174). ICLR 2026. See also the [official implementation](https://github.com/riedlc/AI-GBS). The paper uses TDMI/PID to measure higher-order temporal synergy and includes surrogate null tests; it is complementary to TIDE's directed event-support graph.
- AI Digest. [AI Village dataset](https://huggingface.co/datasets/aidigestorg/ai-village). Dataset includes event timelines, chat, computer-use sessions/turns, memories and goals. Check the current dataset card, schema, changelog and research terms before use.
- Apart Research. [Swarm Alignment Hackathon 2026 — resources and tracks](https://apartresearch.com/sprints/swarm-alignment-2026#resources). Includes Track 2 and Track 4 related work, project ideas and submission requirements.
- Schreiber, T. (2000). [Measuring Information Transfer](https://doi.org/10.1103/PhysRevLett.85.461). Physical Review Letters, 85, 461–464. Foundational reference for transfer entropy; use conditional variants carefully and do not interpret them as automatic causal proof.

## Pass 1 and Pass 2 graph analysis

Install the project dependencies, including igraph, with `uv sync`. The analysis
can snapshot a running rerank checkpoint without changing it. Choose thresholds
explicitly; the following values are examples, not calibrated cutoffs:

```powershell
uv run python -m tide.cli analyze-graphs --dataset-dir data/processed/ai-village/main-village/2026-10-24_to_2026-10-31/v2 --pass1-dir outputs/pass1-deepseek-filtered-v2 --pass2-dir outputs/pass2-deepseek --output-dir outputs/analysis-private --similarity-threshold 0.5 --dependency-threshold 0.5
uv run python -m tide.cli analyze-graphs --dataset-dir data/processed/ai-village/main-village/2026-10-24_to_2026-10-31/v2 --pass1-dir outputs/pass1-deepseek-filtered-v2 --pass2-dir outputs/pass2-public --output-dir outputs/analysis-public --similarity-threshold 0.5 --dependency-threshold 0.5
```

Each output directory must be empty and separate from the input directories.
Run again with a new directory to capture progress or try different thresholds.
The summary records input hashes, the checkpoint snapshot hash, thresholds,
candidate filter, seed, and completed and missing target counts. The analysis
verifies that dataset and annotation hashes match the retrieval cache. Only
complete checkpoint lines are read; incomplete targets are not treated as having
no dependencies.

To compare the same 1,200 events across private and public outputs, add
`--cohort-retrieval outputs/pass2-public/retrieval_candidates.jsonl --cohort-limit 1200`
to both commands, using new output directories. This selects the first 1,200
chronological public retrieval targets, matching the public rerank limit, and
keeps only edges whose endpoints are both in that cohort. Pass 1 reference
resolution still uses the full event history before boundary edges are removed;
it does not reconnect references to a different in-cohort event. The summary
records a cohort ID hash, excluded boundary edges, and coverage for each run.
Check that cohort hashes match and account for missing reranked targets and
different candidate filters before comparing results.

Each graph has event and agent edge JSONL files and statistics JSON files. All
non-consolidation events are included as event nodes, including isolates. Agent
nodes include all authors in those events. Edges point from the earlier source to
the later target. Pass 1 `from_agent` references link the nearest strictly
earlier event by that agent to the annotated event; `to_agent` links the
annotated event to the nearest strictly later event by that agent. Simultaneous
events are never linked; ties at the chosen earlier or later timestamp use
source order. References without a matching event or timestamp are counted.
Duplicate Pass 1 links are merged, retaining reference types.

The separate `pass1_direct_agent` graph uses the annotations directly: an
event by A with `from_agent: B` contributes B -> A, and `to_agent: B`
contributes A -> B. It does not choose a neighboring event. Its edge weight
counts references, separated by type, and its edge records include the
supporting event IDs. Agents referenced in the cohort are included even if
they authored no cohort event. `--min-direct-references` sets the inclusive
minimum count for an ordered agent pair (default 1); this is not a similarity
or dependency-score threshold. These annotations include intended recipients
and explicitly identified sources, and are not confirmed communication or
causal links.

`pass2_retrieval` contains all retrieved pairs at or above the similarity
threshold, with similarity stored as an attribute and no graph weight.
`pass2_retrieval_matched` applies the configured rerank candidate filter and
includes only completed rerank targets. `pass2_reranked` includes scored pairs
at or above the dependency threshold, weighted by dependency score; it is not
additionally filtered by similarity. Both cutoffs are inclusive. Even the
matched graphs can differ because their thresholds measure different quantities.
The public and private candidate filters may also differ, so their outputs need
not have the same candidate universe.

Agent edges aggregate event edges by ordered author pair, including self-links.
Their weights are edge counts for unweighted graphs and sums of dependency
scores for reranked graphs; each also records its event-edge count. Event edges
with an unknown author cannot be aggregated and are counted in the summary.
Statistics include per-node in/out degree, weighted strengths, weak and strong
components, degree distributions and Leiden communities. Leiden uses modularity
on an undirected projection, summing reciprocal weights; zero-weight edges are
excluded from that projection only. Agent communities use aggregated weights.
The seed is configurable with `--seed`, but results can vary across igraph
versions. Reference links, similarity, and model-scored dependencies are not
proof of causation or communication.

### Private score spot check

A qualitative review of eight private rerank examples near 0.25, 0.50, 0.75,
and 0.90 found plausible high-score examples alongside a possible false positive
involving two different sets of 13 checks and indirect interactions described
as direct. The examples were selected purposively and cannot establish score
accuracy. See the
[private Pass 2 qualitative report](reports/private_pass2_qualitative_examples.md)
and review private-data handling restrictions before sharing it.

## Public temporal decay calibration

Fit the public rerank scores against private rerank scores for matching
source-target event pairs, then evaluate on later targets:

```powershell
uv run python -m tide.cli calibrate-decay --dataset-dir data/processed/ai-village/main-village/2026-10-24_to_2026-10-31/v2 --public-dir outputs/pass2-public --private-dir outputs/pass2-deepseek --output-dir outputs/public-decay-calibration-200
```

The output directory must be empty and separate from the inputs. The command
snapshots checkpoints without modifying them, verifies input fingerprints, and
selects the first 1,200 chronological public retrieval targets. The first 200
targets are the training split and the remaining 1,000 are holdout; both
endpoints of each pair must be inside the cohort. Missing targets and pairs
not scored by both runs are excluded from fitting and comparison, not given
zero scores. Coverage and candidate-filter differences are reported separately.
Public-only and private-only pair counts are measured only for targets that
completed in both runs.

By default, the calibrated score is
`sigmoid(alpha * public_score + intercept) * exp(-lambda * hours)`. A bounded
SciPy least-squares optimizer minimizes pair-weighted training MSE against the
private score from four deterministic starting points. Alpha and intercept
are bounded to [-30, 30], and lambda to [0, 5] per hour by default; results
include optimizer convergence and active bounds. Multi-start optimization does
not guarantee a global optimum. Sigmoid calibration can assign a positive score
even when the original public score is zero.

As an ablation, the same run also fits a ranking-only sigmoid,
`sigmoid(alpha_rank * public_score + intercept_rank)`, with no elapsed-time
term. Both use the same matched training pairs and holdout pairs. Metrics report
raw scores, sigmoid-only ranking calibration, and sigmoid-plus-decay on both
splits; the exported public edge file contains both calibrated variants. The
network evaluation compares both public graphs against each reference at the
same threshold and reports both PR/AP results.

For comparison, a deterministic grid separately fits the simpler decay-only
score `public_score * exp(-lambda * hours)`. Defaults search 0 through 5 per
hour in 1,000 intervals; `--lambda-max` changes both models' upper lambda
bound, while `--grid-steps` changes the decay-only grid. The result flags an
upper-bound grid solution. `--model decay-only` exports the simpler model as
the calibrated score. `--train-targets` and `--cohort-limit` change the split.

`calibration.json` records snapshot hashes, coverage, each fit and raw,
ranking-only, decay-only and sigmoid-plus-decay MSE and MAE for both splits. It
also reports precision and recall at the fixed
`--threshold` (default 0.5), using private scores as labels **only on matched
pairs**. `matched_pairs.jsonl` contains the comparison scores and split;
`public_decayed_edges.jsonl` contains every completed public scored pair
within the cohort, with original, decay-only, ranking-only and sigmoid-plus-decay scores, before
thresholding. To produce a weighted graph from the calibrated scores, add
`--calibration-dir outputs/public-sigmoid-decay-calibration-200` to the public
`analyze-graphs` command together with the same cohort arguments used for
calibration. The graph uses `calibrated_public_score` and the inclusive
`--dependency-threshold`; it is written separately as `pass2_calibrated`.
The command requires the calibration cohort and public checkpoint snapshot to
match the analysis inputs, and checks that every public pair is present once.
The private checkpoint may be incomplete while running: use a new output directory
to recalibrate after it advances. These metrics measure agreement with private
scores, not accuracy against ground truth, and matched pairs can be biased by
the different candidate filters and missing targets.

## Compact agent-network evaluation

Compare direct raw-chat mentions, private-context inference, and calibrated
public TIDE on jointly completed held-out targets:

```powershell
uv run python -m tide.cli evaluate-networks --dataset-dir data/processed/ai-village/main-village/2026-10-24_to_2026-10-31/v2 --registry data/ai-village-extracted/main-village/agents.jsonl --public-dir outputs/pass2-public --private-dir outputs/pass2-deepseek --calibration-dir outputs/public-sigmoid-decay-calibration-200 --output-dir outputs/network-evaluation-authors-holdout
```

The node universe contains only registered authors of events in the full
calibration cohort; mentions of registered nonauthors are excluded. All three
networks use this same universe and exclude self-links. Raw chat @mentions are
directed from the message author to the mentioned agent and need not express
information flow. Private and public inferred event links point from source to
target and are aggregated by maximum score per agent pair. The public score
uses the fitted sigmoid and time decay; private scores have no separately fitted
temporal adjustment. Thresholds default to 0.5 and can be changed with
`--threshold` and `--private-threshold`; direct mentions require one mention.

The command emits `metrics.json`, `report.md`, and precision-recall and
community figures in PNG and PDF. The network figure has four panels:
explicit mentions, private inference, `Public TIDE`, and
`Public TIDE (no temporal)`, all on one shared layout with aligned community
colors. A separate heatmap figure shows community overlap. Community labels
are aligned by maximum-overlap Hungarian matching, but this is only label
matching, not evidence of equivalence. The PR figure compares both public
variants against each reference using sklearn average precision and shows
reference-prevalence baselines. The report also includes per-event connection
counts. The threshold sweep compares against each
reference separately. Average precision ranks all non-self author pairs,
assigning zero to unscored public pairs; reference absence is not a verified
negative. Leiden uses a weighted undirected projection, retaining isolates, and
the output reports active-only partition comparisons and seed sensitivity.
Only targets completed in both checkpoints are evaluated, and calibration
targets are excluded; candidate-filter differences and incompleteness remain
important limitations.

The evaluation also writes a separate `event_level_explicit_reference.json`
and `event_level_explicit_reference_edges.jsonl`. For each jointly completed
holdout target, it extracts unambiguous @mentions from the original
`AGENT_TALK` `data.content`, then links each non-self mentioned agent's latest
strictly earlier event in the full 1,200-event cohort to the target. Predecessor
selection uses time only, independently of TIDE candidates or scores. Mentions
without an earlier cohort event and self-mentions are counted as coverage or
exclusions, not false negatives. Event-level metrics use sigmoid-plus-decay
public scores and the same threshold; AP ranks strictly earlier cohort-event /
eligible-target pairs and assigns zero to unscored TIDE pairs. This is a
mention-based predecessor proxy, not a verified information-source label;
absent mentions are not verified negatives. The report compares it directly
with the agent-level mention metrics to show how aggregation changes overlap.

## Future roadmap

The following steps are proposed continuation work, not completed features.

1. **Finish and freeze the current comparison.** Resolve remaining Pass 2
	failures where practical, then save immutable public/private checkpoint
	snapshots and rerun calibration and evaluation from those exact snapshots.
	Keep the training/holdout split chronological and report coverage.
2. **Validate scores against reviewed evidence.** Sample event pairs across
	score ranges, including TIDE-only and reference-only cases. Have reviewers
	judge whether the earlier event plausibly contributed, separately recording
	direct evidence, contextual relevance, and insufficient evidence. Use these
	labels to assess calibration and score thresholds rather than treating
	private model scores or mentions as ground truth.
3. **Improve reference quality and comparability.** Audit whether explicit
	mentions identify an addressee, a cited agent, or an information source.
	Compare private and public inference on a common candidate set as well as
	under their current retrieval policies. Audit event-level predecessor
	proxies against actual message or artifact chains where observable.
4. **Stress-test common-cause errors.** Extend synthetic controls for identical
	tasks and shared external events, then measure how errors change under
	message-blind, result-hidden, and partial-observability conditions.
5. **Add higher-level network analyses only after edge validation.** Explore
	time-respecting paths and broker roles first; consider conditional transfer
	entropy or TDMI/PID only when a defensible, low-dimensional temporal state
	representation is available.

Throughout, keep dependency, communication, prediction, and causation claims
separate. Any use of private traces or publication of examples must continue to
follow the dataset's access, privacy, and research terms.


