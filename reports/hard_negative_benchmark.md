# Hard-Negative Benchmark and Held-out Method Comparison

- Baseline pre-hard-control results preserved at `outputs/extended_validation_baseline_v1/`.
- Train seeds: `[7, 13, 29]`
- Held-out seeds: `[101, 103, 107]`
- Runs per seed: `3`

## Held-out profile comparison

| Profile | Precision | Recall | F1 | AUPRC | Independent FPR | Shared-external FPR | Delayed detect | Broker detect | Indirect A->C error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_t05 | 0.476 | 0.938 | 0.632 | 0.786 | 0.559 | 0.000 | 1.000 | 1.000 | 0.000 |
| baseline_t07 | 0.727 | 0.750 | 0.738 | 0.786 | 0.059 | 0.000 | 1.000 | 0.000 | 0.000 |
| strict_cross_agent | 0.500 | 0.625 | 0.556 | 0.701 | 0.265 | 0.000 | 1.000 | 0.000 | 0.000 |
| context_aware_abstain | 0.500 | 0.625 | 0.556 | 0.701 | 0.265 | 0.000 | 1.000 | 0.000 | 0.000 |

## Updated synthetic benchmark summary

- Current full benchmark precision/recall/F1: 0.476 / 0.938 / 0.632
- Independent-scenario FPR: 0.559
- Indirect direct-edge error rate: 0.000

## Generalization note

Train and held-out scores are currently very close because the synthetic generator randomizes values more than structure. So this held-out split is still useful for leakage checks, but not a strong stress-test of structural generalization.

## First real-data protocol (no download in this step)

1. Run local schema audit first: `python -m tide.cli audit-dataset --dataset-dir <LOCAL_DATA_DIR> --output reports/real_data_audit.json`.
2. Normalize one small table slice (100-500 rows) while preserving raw records: `python -m tide.cli normalize-events --input <TABLE.jsonl|csv> --limit 300 --output outputs/normalized_slice.jsonl --raw-output outputs/normalized_slice_raw.jsonl`.
3. Manually inspect normalized events for timestamp/agent/provenance/missingness fields.
4. Run dependency inference only on that slice and inspect top predicted cross-agent edges manually before scaling.
5. Track observed missingness and unresolved ordering explicitly in the audit report.
