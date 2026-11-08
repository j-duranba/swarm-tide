# TIDE — Temporal Inference of Dependencies in Agent Swarms
**Status:** Research prototype / hackathon project in progress (6–8 November 2026).  
**Track:** Primarily Track 4 — Multi-Agent Oversight and Control; related to Track 2 — The Science of Agent Networks.

> **Project summary:** TIDE investigates whether we can reconstruct dependencies between actions in a multi-agent system from observable activity logs. We represent agent activity as structured events, retrieve potentially relevant predecessor events, and estimate whether earlier events support later actions. The resulting event-level dependency graph can be aggregated into a time-varying agent network and analysed for information propagation, relays, brokers and collective behaviour. We explicitly test whether the method mistakes independent agents with identical tasks—or agents reacting to shared external events—for collaboration.

---

## 0. Current project status

This README is an initial research plan, not a claim that the experiments have already been completed. Update the status, dataset details, methods, figures and results as implementation proceeds. Clearly label planned, implemented and completed components.

Research prototype for reconstructing **probabilistic event dependencies** from multi-agent traces without relying on private chain-of-thought.

This repository implements an MVP pipeline that:

1. normalizes observable logs into immutable event records;
2. retrieves plausible earlier predecessor events for each target event;
3. estimates support/dependency with explicit evidence and abstention;
4. builds an event-level directed weighted dependency graph;
5. aggregates to an agent-level directed matrix using the convention  
   `W[target_agent, source_agent] = support evidence`.

The implementation keeps the key distinctions explicit:

- retrieval score vs support score;
- support/dependency vs causation;
- predictive dependence vs collaboration;
- similarity vs influence;
- observed links vs inferred links.

## Current MVP status

The repo currently includes:

- **Dataset audit tooling** for local files plus Hugging Face access-state checks (no gate bypass attempts).
- **Normalized schema** with source provenance and observability flags.
- **Synthetic controlled trace generator** with known ground-truth dependencies and negative controls.
- **Candidate retrieval** (semantic + temporal + artifact overlap + same-agent history).
- **Rule-based support estimator** with evidence categories, abstention, and alternative explanations.
- **Graph construction + aggregation** for event and agent-level outputs.
- **CLI pipeline** to run synthetic end-to-end and export JSON artifacts.

## Quickstart

```bash
python -m tide.cli run-synthetic --output-dir outputs/synth --seed 7
python -m tide.cli audit-dataset --dataset-dir data
```

Outputs include:

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
either input. It creates separate query/document embeddings with
`jinaai/jina-embeddings-v5-text-nano` and retrieval prompts, then combines
12-hour same-agent message history, annotated information sources, incoming
messages and dense semantic retrieval. Same-agent history is the explicit
exception to the cross-source same-agent exclusion. Each candidate set is
reranked against the full private content and available thinking/reasoning by
DeepSeek; retrieval similarity is retained separately from dependency scores.
Source A/B/C refer to the candidate-generation procedure: A's same-agent
messages are retained, while B-D exclude the target author. Events tied with the
target timestamp are not retrieved. The model license is CC BY-NC 4.0; check
usage rights and data-handling requirements before downloading or transmitting
private traces.

Install the optional local embedding dependencies, then run a small pilot:

```powershell
uv sync --extra pass2
uv run --env-file .env python -m tide.cli pass2 --dataset-dir data/processed/ai-village/main-village/2026-10-24_to_2026-10-31/v2 --pass1-dir outputs/pass1-deepseek-filtered-v2 --output-dir outputs/pass2-deepseek --limit 3
```

The default embedding revision is pinned to a Hugging Face commit; use
`--embedding-revision` only when intentionally changing model revisions.
`event_embeddings.jsonl`, `dependencies.jsonl`, append-only embedding
and reranking checkpoints, `pass2_errors.jsonl`, and `pass2_config.json` are
written only under the Pass 2 output directory. Repeating the same command
resumes missing work; changed inputs or settings require a new output directory,
or `--overwrite` to start fresh. The API key uses `ADAPTION_API_KEY` by default.

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

## 1. Research question

**Can we infer the probabilistic dependency structure between events in a multi-agent system from timestamped agent activity traces, and use it to identify cross-agent information flow and functional collective behaviour when private chain-of-thought is unavailable and some observations are hidden?**

The starting point is a collection of agents with individual goals, acting over time and interacting directly (e.g. messages) or indirectly (e.g. shared files, databases, memory or tools). We assume access to observable action records and, depending on the experimental condition, agent outputs and tool results. We do not require private chain-of-thought.

For each event, we record the agent, timestamp, goal/context, state where observable, action, tool, input, result and public output. For every new event, we search for a small set of relevant earlier events—both from the same agent and from other agents—and estimate whether each predecessor provides evidence supporting the new event.

The main output is a **probabilistic event-dependency graph**. An agent-level network is derived by aggregating event dependencies over time.

## 2. Hypotheses

### H1 — Recoverable event dependencies

A hybrid retrieval method combining semantic similarity, temporal information and resource/artifact relationships can retrieve supporting predecessor events more effectively than semantic similarity or temporal proximity alone.

### H2 — Cross-agent dependencies beyond individual history

For some episodes, the history of other agents provides predictive information about an agent's subsequent actions beyond that agent's own history and observable shared task/environment context.

### H3 — Observable information provenance supports network reconstruction

Combining event-level support estimation with explicit resource and communication evidence can recover meaningful information-flow paths, including multi-hop paths through intermediate agents.

### H4 — The method can distinguish common-cause similarity from cross-agent dependency

When agents receive identical tasks but act independently, or respond to the same external event without communicating, the method should assign substantially weaker cross-agent support than in controlled information-transfer conditions.

### H5 — Partial observability reduces but need not eliminate recoverability

Hiding message content, agent outputs, selected agents or selected events will reduce reconstruction quality. The rate and pattern of degradation can reveal which observable signals are most useful.

### Exploratory hypothesis — Brokers and aggregators

Agents that receive information from several sources and subsequently relay, transform or combine it may be identifiable from time-respecting event paths and support relationships.

These hypotheses are provisional. The initial prototype should prioritise H1 and H4, then investigate H2/H3 if the trace quality permits.

## 3. What TIDE measures (and what it does not)

The project separates four related concepts:

| Concept | Meaning | Candidate measurement |
|---|---|---|
| Event support | An earlier event contains information or a result that supports a later action | Pairwise support score / calibrated probability |
| Predictive dependence | One agent's past improves prediction of another agent's future beyond observed context | Conditional transfer entropy or held-out log-likelihood improvement |
| Collective synergy | Multiple agents jointly contain predictive information about a future joint/group outcome not available from any one agent alone | TDMI + partial information decomposition (PID) |
| Functional coordination | Agents make interdependent or complementary contributions to a related task/outcome | Evidence combining event provenance, predictive dependence, complementary roles and outcome relevance |

These should not be collapsed into a single unqualified “collaboration probability”. Similarity is not influence; temporal correlation is not communication; predictive dependence is not automatically causation; information transfer is not necessarily cooperation. An agent can mislead or obstruct another agent. The initial project therefore uses the cautious phrase **functional cross-agent coordination** rather than inferring intent.

## 4. Event representation

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

## 5. Method

### 5.1 Candidate predecessor retrieval

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

### 5.2 Event-support estimation

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

### 5.3 Event graph

Construct a directed graph whose nodes are events and whose edges run from an earlier supporting event to a later supported event:

\[
G_E=(E,\mathcal{A}),\qquad e_j\rightarrow e_k\text{ if }t_j<t_k.
\]

Edges should record support score/probability, support type, time lag, retrieval score, evidence, alternative explanations, and source references. Preserve direct edges and multi-hop paths separately.

### 5.4 Agent-level dynamic network

Aggregate event dependencies into a time-varying directed matrix. Use the convention:

\[
W_{ij}(t)=\text{evidence that events by agent }j\text{ support events by agent }i\text{ in window }t.
\]

This convention means the source-to-recipient edge is `j → i`. The matrix need not be symmetric, and its rows should not automatically be normalized to sum to one: several prior events can support one target event.

Aggregation can use counts, sums, mean support per target event, unique predecessor counts or time-decayed sums. Report raw and normalized versions separately where useful.

### 5.5 Directed predictive dependence

As an optional statistical measure, estimate whether agent A's past improves prediction of agent B's next action after conditioning on B's own history and observable common context:

\[
TE_{A\rightarrow B\mid Z}
=I(A_{\mathrm{past}};B_{\mathrm{future}}\mid B_{\mathrm{past}},Z_{\mathrm{past}}),
\]

where \(Z\) may include common task instructions, shared goals, public feedback and observable environment state. For irregular event logs, a held-out predictive log-likelihood comparison may be more practical than a conventional time-series estimator.

A positive value indicates residual predictive dependence under the chosen conditioning set; it does not establish direct communication or causal influence when important common causes are unobserved.

### 5.6 TDMI and partial information decomposition

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

## 6. Key scientific challenge: independent agents with the same task

Two agents can perform almost identical action sequences without communicating. Shared goals, prompts, tools and environmental events are common causes of behavioural similarity.

TIDE must include explicit controls:

### Negative control 1 — identical task, independent execution

Give two or more agents the same model, prompt, task, tools and environment, but isolate them from one another. They may independently search for the same file, read the same content and produce similar outputs. The detector should not label this as strong cross-agent information transfer simply because the actions are similar.

### Negative control 2 — shared external event

Agents do not communicate, but both react to the same external event or public environment change. The detector should condition on the observed common event/context and avoid inventing a direct agent-to-agent edge.

### Positive control 1 — explicit information transfer

Agent A discovers a fact and sends it to B. The full-log condition contains the message; the message-blind condition hides its content from the detector.

### Positive control 2 — shared resource channel

Agent A writes a fact to a shared file/database; B reads and uses it. No explicit message is necessary.

### Positive control 3 — broker/aggregator

Agents A and B provide complementary facts to C. C combines them and produces a downstream result. Test whether C is identified as an aggregator or intermediary.

Additional conditions can include irrelevant messages, misleading information, two-hop relays, hidden agents and missing events.

## 7. Datasets

### 7.1 AI Village — primary candidate

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

### 7.2 Synthetic controlled traces

A small synthetic generator is essential because real communication logs are weak ground truth: a message being sent does not mean it was read or used. Synthetic traces allow known support edges, independent agents and common causes to be specified exactly.

If AI Village access is not available in time, develop and evaluate the pipeline on synthetic traces, and state clearly that real-world validation remains future work.

### 7.3 Swarm Traces — possible follow-up

[Swarm Traces](https://swarmtraces.org/) is relevant to the Hugging Face incident, but it should only be used for quantitative event-graph evaluation if the available records contain reliable timestamps, actor/agent attribution and defensible event linkage. Reconstructed payload relationships are not automatically proof of temporal or causal edges. If metadata is inadequate, use it only as a qualitative forensic case study.

## 8. Experiments and ablations

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

## 9. Measures and evaluation

### Candidate retrieval

- Recall@K for labelled supporting predecessor events;
- MRR where one principal predecessor is known;
- retrieval latency and candidate-set size.

### Event-support inference

- precision, recall, F1 and AUPRC for known support edges;
- Brier score and calibration/reliability plots if probabilities are reported;
- abstention coverage and selective accuracy;
- breakdown by direct-message, resource-mediated, information-content, same-agent, ambiguous and no-support cases.

### Network reconstruction

- event-edge precision/recall/AUPRC against synthetic ground truth;
- direction and time-respecting path recovery;
- agent-edge recovery after aggregation;
- centrality/community analysis only as secondary descriptive analysis.

### Predictive dependence and collective structure

- conditional transfer entropy or held-out predictive log-likelihood improvement;
- TDMI/PID synergy and optionally triplet `G3` if the time series is sufficiently long and low-dimensional;
- false-positive rate in identical-task independent agents and shared-external-event controls;
- sensitivity to window length, lag, discretization and missingness.

### Broker/aggregator roles

Look for agents that receive information from multiple sources and subsequently pass on or combine that information. Time-respecting betweenness may be a useful structural feature, but high degree or betweenness alone is not sufficient to label an information broker.

## 10. Limitations and interpretation

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

## 11. Weekend scope and milestones

### Minimum viable result

1. Audit the repository and dataset access/schema.
2. Normalize a small subset of real traces if access is granted; otherwise use synthetic traces.
3. Build a synthetic generator with known dependencies and same-task/common-cause negative controls.
4. Implement hybrid predecessor retrieval with temporal filtering and same-agent candidates.
5. Implement a support scorer with evidence categories and abstention.
6. Build an event graph and aggregate agent graph/matrix.
7. Evaluate retrieval and edge recovery, with explicit false-positive reporting on independent same-task agents.
8. Run one useful ablation (full observable vs message-blind or result-hidden).
9. Produce at least one interpretable visualization and a concise report.

### Stretch goals

- Conditional transfer entropy or held-out predictive comparison;
- TDMI/PID on compact synchronized states;
- broker/aggregator identification;
- partial-agent and partial-event robustness;
- Swarm Traces case study after metadata audit.

Do not attempt to process every row in AI Village, train a GNN, or estimate high-order PID over raw text during the weekend.

## 12. Reproducibility and implementation notes

- Use stable event IDs, deterministic seeds and explicit configs.
- Stream large JSONL.GZ files; do not load all turns into memory unnecessarily.
- Cache embeddings and model-judge outputs.
- Record model name/version, prompt hash, embedding model, retrieval weights, temporal windows, data slice, missingness and costs.
- Keep raw/gated data out of Git. Add data/cache/secrets to `.gitignore`.
- Make the pipeline work offline on synthetic traces with a rule-based scorer.
- Keep ingestion, normalization, retrieval, support scoring, graph construction and evaluation as separate modules.
- Add tests for temporal filtering, multiple predecessors, missing fields, artifact matching, stable IDs and common-cause controls.

## 13. Initial references

- Riedl, C. (2026). [Emergent Coordination in Multi-Agent Language Models](https://arxiv.org/abs/2510.05174). ICLR 2026. See also the [official implementation](https://github.com/riedlc/AI-GBS). The paper uses TDMI/PID to measure higher-order temporal synergy and includes surrogate null tests; it is complementary to TIDE's directed event-support graph.
- AI Digest. [AI Village dataset](https://huggingface.co/datasets/aidigestorg/ai-village). Dataset includes event timelines, chat, computer-use sessions/turns, memories and goals. Check the current dataset card, schema, changelog and research terms before use.
- Apart Research. [Swarm Alignment Hackathon 2026 — resources and tracks](https://apartresearch.com/sprints/swarm-alignment-2026#resources). Includes Track 2 and Track 4 related work, project ideas and submission requirements.
- Schreiber, T. (2000). [Measuring Information Transfer](https://doi.org/10.1103/PhysRevLett.85.461). Physical Review Letters, 85, 461–464. Foundational reference for transfer entropy; use conditional variants carefully and do not interpret them as automatic causal proof.


