# TIDE — Temporal Inference of Dependencies in Agent Swarms

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
