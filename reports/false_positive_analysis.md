# False Positive Forensic Analysis (Independent-Agent Controls)

- Total independent-scenario false positives inspected: **90**
- Classification categories:
  - shared_task_or_instruction_effect: 54
  - temporal_proximity_without_transfer: 36

- Scenario distribution:
  - hardneg_repeated_common_obs: 45
  - hardneg_timing_offset: 9
  - hardneg_paraphrased_independent: 9
  - hardneg_equivalent_sequences: 9
  - hardneg_verbosity: 9
  - hardneg_similar_diff_times: 9

## Representative cases

| Scenario | Source -> Target | Lag(s) | SemSim | Retrieval rank | Retrieval score | Support score/type | Classification |
|---|---|---:|---:|---:|---:|---|---|
| hardneg_timing_offset | e35(A) -> e36(B) | 16.0 | 0.422 | 2 | 0.419 | 0.880 (information-content) | shared_task_or_instruction_effect |
| hardneg_paraphrased_independent | e38(A) -> e39(B) | 1.0 | 0.535 | 1 | 0.470 | 0.880 (information-content) | shared_task_or_instruction_effect |
| hardneg_equivalent_sequences | e40(A) -> e41(B) | 1.0 | 0.889 | 1 | 0.630 | 0.550 (contextual) | shared_task_or_instruction_effect |
| hardneg_verbosity | e44(A) -> e45(B) | 1.0 | 0.316 | 1 | 0.372 | 0.550 (contextual) | shared_task_or_instruction_effect |
| hardneg_repeated_common_obs | e49(A) -> e50(B) | 1.0 | 0.923 | 1 | 0.645 | 0.550 (contextual) | shared_task_or_instruction_effect |
| hardneg_repeated_common_obs | e49(A) -> e51(ENV) | 2.0 | 0.277 | 2 | 0.335 | 0.550 (contextual) | temporal_proximity_without_transfer |
| hardneg_repeated_common_obs | e50(B) -> e51(ENV) | 1.0 | 0.185 | 3 | 0.293 | 0.550 (contextual) | temporal_proximity_without_transfer |
| hardneg_repeated_common_obs | e48(ENV) -> e52(A) | 4.0 | 0.277 | 4 | 0.335 | 0.550 (contextual) | temporal_proximity_without_transfer |
| hardneg_repeated_common_obs | e48(ENV) -> e53(B) | 5.0 | 0.185 | 5 | 0.293 | 0.550 (contextual) | temporal_proximity_without_transfer |
| hardneg_similar_diff_times | e54(A) -> e55(B) | 30.0 | 1.000 | 1 | 0.678 | 0.550 (contextual) | shared_task_or_instruction_effect |

## Failure mode conclusion

The current independent-agent FPR is driven primarily by systematic shared-task/parallel-execution patterns, not isolated random outliers. Cross-agent edges are still inferred when both agents independently surface the same fact.
