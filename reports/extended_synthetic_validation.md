# TIDE Extended Synthetic Validation
## 1) Experimental setup
- Synthetic-only benchmark with **9 runs** (3 seeds × 3 run IDs).
- Ground truth includes direct edges, mediated-only paths, misleading transfers, and shared external causes.
- Retrieval and support scores were evaluated separately; support scores are **heuristic scores, not calibrated probabilities**.
- Repro command: `python -m tide.cli run-extended-validation --output-dir outputs/extended_validation --report-path reports/extended_synthetic_validation.md --seeds 7,13,29 --runs-per-seed 3`

## 2) Retrieval comparison
| Strategy | R@3 | R@5 | R@10 | R@20 | Same-agent R@10 | Cross-agent R@10 | Delayed R@10 | Artifact R@10 | Avg candidate size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.07 |
| recent_history_baseline | 0.167 | 0.167 | 0.167 | 0.167 | 1.000 | 0.000 | 0.000 | 0.000 | 0.26 |
| semantic_only | 0.967 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.25 |
| semantic_plus_temporal | 0.967 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.25 |
| semantic_temporal_artifact | 0.967 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.25 |

## 3) Full-pipeline event dependency results
| Metric | Value |
|---|---:|
| event_precision | 0.476 |
| event_recall | 0.938 |
| event_f1 | 0.632 |
| event_auprc | 0.786 |
| fpr_independent_identical_task | 0.559 |
| fpr_shared_external_non_env | 0.000 |
| precision_explicit_communication | 0.667 |
| precision_shared_resource | 0.667 |
| precision_broker_aggregator | 1.000 |
| precision_misleading_transfer | 1.000 |
| precision_delayed | 0.250 |
| broker_pattern_detected_rate | 1.000 |
| misleading_dependency_detected_rate | 1.000 |
| delayed_dependency_detected_rate | 1.000 |
| incorrect_direct_A_to_C_rate | 0.000 |

## 4) Observability ablations
| Ablation | F1 | Precision | Recall | AUPRC | Delayed detection rate |
|---|---:|---:|---:|---:|---:|
| increased_delay | 0.632 | 0.476 | 0.938 | 0.786 | 1.000 |
| missing_events_20pct | 0.624 | 0.472 | 0.926 | 0.797 | 0.778 |
| missing_selected_fields_30pct | 0.544 | 0.447 | 0.694 | 0.665 | 0.778 |
| no_action_arguments | 0.405 | 0.340 | 0.500 | 0.450 | 0.000 |
| no_action_results | 0.519 | 0.429 | 0.656 | 0.677 | 1.000 |
| no_artifact_ids | 0.645 | 0.492 | 0.938 | 0.825 | 1.000 |
| no_nl_outputs | 0.674 | 0.537 | 0.906 | 0.796 | 1.000 |
| only_action_and_time | 0.192 | 0.250 | 0.156 | 0.342 | 0.000 |
| timestamp_noise | 0.482 | 0.324 | 0.947 | 0.738 | 1.000 |

## 5) Agent-level directionality and brokerage checks
- Matrix convention used in all code/results: **W[target][source]**.
- Broker detection requires receiving from multiple upstream sources and relaying downstream, not centrality alone.
- Indirect chain check explicitly tracks A→B→C and records whether incorrect direct A→C edges are introduced.

## 6) Representative errors
### False positives (sample)
```json
[
  {
    "source_event_id": "e24",
    "target_event_id": "e26",
    "score": 0.95,
    "support_type": "direct",
    "scenario": "indirect_chain",
    "source_agent": "A",
    "target_agent": "B",
    "evidence": [
      "value reuse: ['token-1950']",
      "message content appears in target action"
    ]
  },
  {
    "source_event_id": "e11",
    "target_event_id": "e13",
    "score": 0.9,
    "support_type": "artifact-mediated",
    "scenario": "shared_resource",
    "source_agent": "A",
    "target_agent": "B",
    "evidence": [
      "artifact overlap: ['rid-1791']",
      "value reuse: ['rid-1791']"
    ]
  },
  {
    "source_event_id": "e70",
    "target_event_id": "e72",
    "score": 0.9,
    "support_type": "artifact-mediated",
    "scenario": "hardpos_indirect_artifact",
    "source_agent": "A",
    "target_agent": "C",
    "evidence": [
      "artifact overlap: ['aid-946']",
      "value reuse: ['aid-946']"
    ]
  }
]
```
### False negatives (sample)
```json
[
  {
    "source_event_id": "e30",
    "target_event_id": "e31",
    "scenario": "mixed_conditions",
    "reason": "not retrieved or below threshold"
  },
  {
    "source_event_id": "e63",
    "target_event_id": "e64",
    "scenario": "hardpos_identifier_transform",
    "reason": "not retrieved or below threshold"
  }
]
```

## 7) Bugs/risk fixes implemented
- Added strict temporal precedence checks to prevent retrieval/graph edges when ordering is ambiguous.
- Added explicit missing-field/provenance metadata on normalized events.
- Added tests that fail on direction reversal and temporal ambiguity.

## 8) Limitations and next steps
- Support estimator remains heuristic; scores are not calibrated probabilities.
- Synthetic benchmark cannot prove external validity for real-world traces.
- Next: run same protocol on real traces with access-compliant ingestion, then compare failure modes.
