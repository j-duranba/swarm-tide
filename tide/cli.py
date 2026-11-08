from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from tide.audit import audit_dataset
from tide.benchmark import (
    analyze_independent_false_positives,
    default_retrieval_config,
    run_extended_validation,
    run_heldout_method_comparison,
    write_extended_report,
    write_false_positive_report,
    write_hard_negative_report,
)
from tide.ingest import load_table_rows, normalize_table
from tide.io_utils import write_json
from tide.io_utils import write_jsonl
from tide.pipeline import run_synthetic_pipeline
from tide.pass1 import run_pass1
from tide.pass2 import EMBEDDING_MODEL, EMBEDDING_REVISION
from tide.pass2_stages import run_pass2_all, run_pass2_embed, run_pass2_rerank, run_pass2_retrieve


def _cmd_run_synthetic(args: argparse.Namespace) -> int:
    outputs = run_synthetic_pipeline(output_dir=args.output_dir, seed=args.seed, min_edge_score=args.min_edge_score)
    print(json.dumps(outputs.summary, indent=2, sort_keys=True))
    return 0


def _cmd_audit_dataset(args: argparse.Namespace) -> int:
    report = audit_dataset(args.dataset_dir)
    payload = report.as_dict()
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(out_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_normalize_events(args: argparse.Namespace) -> int:
    events = normalize_table(path=args.input, source_table=args.source_table, limit=args.limit)
    rows = [e.as_dict() for e in events]
    raw_rows = []
    if args.raw_output:
        raw_source_table = args.source_table or Path(args.input).stem
        for record_id, raw in load_table_rows(args.input, limit=args.limit):
            raw_rows.append({"source_table": raw_source_table, "source_record_id": record_id, "raw_record": raw})
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(out_path, rows)
    if args.raw_output:
        raw_out = Path(args.raw_output)
        raw_out.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(raw_out, raw_rows)
    print(json.dumps({"normalized_events": len(rows), "output": args.output}, indent=2, sort_keys=True))
    return 0


def _cmd_pass1(args: argparse.Namespace) -> int:
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise ValueError(f"Set {args.api_key_env} before running Pass 1")
    result = run_pass1(
        args.dataset_dir, args.output_dir, args.model, api_key,
        concurrency=args.concurrency, limit=args.limit, timeout=args.timeout,
        retries=args.retries, overwrite=args.overwrite, retry_errors_only=args.retry_errors_only,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 1 if result["unresolved_total"] else 0


def _cmd_pass2(args: argparse.Namespace) -> int:
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise ValueError(f"Set {args.api_key_env} before running Pass 2")
    result = run_pass2_all(
        args.dataset_dir, args.pass1_dir, args.output_dir, api_key,
        model=args.model, embedding_model=args.embedding_model,
        embedding_revision=args.embedding_revision, precision=args.precision,
        batch_size=args.batch_size,
        limit=args.limit, timeout=args.timeout, retries=args.retries,
        concurrency=args.concurrency, candidate_group_size=args.candidate_group_size,
        candidate_filter=args.candidate_filter, overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 1 if result["failures"] else 0


def _cmd_pass2_embed(args: argparse.Namespace) -> int:
    result = run_pass2_embed(
        args.dataset_dir, args.output_dir, embedding_model=args.embedding_model,
        embedding_revision=args.embedding_revision, precision=args.precision,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 1 if result["failures"] else 0


def _cmd_pass2_retrieve(args: argparse.Namespace) -> int:
    result = run_pass2_retrieve(
        args.dataset_dir, args.pass1_dir, args.output_dir,
        limit=args.limit, overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _cmd_pass2_rerank(args: argparse.Namespace) -> int:
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise ValueError(f"Set {args.api_key_env} before running Pass 2 reranking")
    result = run_pass2_rerank(
        args.dataset_dir, args.output_dir, api_key, model=args.model,
        timeout=args.timeout, retries=args.retries, concurrency=args.concurrency,
        candidate_group_size=args.candidate_group_size, candidate_filter=args.candidate_filter,
        limit=args.limit, overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 1 if result["failures"] else 0


def _cmd_pass2_public_embed(args: argparse.Namespace) -> int:
    result = run_pass2_embed(
        args.dataset_dir, args.output_dir, embedding_model=args.embedding_model,
        embedding_revision=args.embedding_revision, precision=args.precision,
        batch_size=args.batch_size, context_mode="public", pass1_dir=args.pass1_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 1 if result["failures"] else 0


def _cmd_pass2_public_retrieve(args: argparse.Namespace) -> int:
    result = run_pass2_retrieve(
        args.dataset_dir, args.pass1_dir, args.output_dir,
        limit=args.limit, context_mode="public", overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _cmd_pass2_public_rerank(args: argparse.Namespace) -> int:
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise ValueError(f"Set {args.api_key_env} before running public Pass 2 reranking")
    result = run_pass2_rerank(
        args.dataset_dir, args.output_dir, api_key, model=args.model,
        timeout=args.timeout, retries=args.retries, concurrency=args.concurrency,
        candidate_group_size=args.candidate_group_size, candidate_filter=args.candidate_filter,
        context_mode="public",
        pass1_dir=args.pass1_dir, limit=args.limit, overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 1 if result["failures"] else 0


def _cmd_analyze_graphs(args: argparse.Namespace) -> int:
    from tide.graph_analysis import run_graph_analysis

    result = run_graph_analysis(
        args.dataset_dir, args.pass1_dir, args.pass2_dir, args.output_dir,
        similarity_threshold=args.similarity_threshold, dependency_threshold=args.dependency_threshold,
        seed=args.seed, cohort_retrieval=args.cohort_retrieval, cohort_limit=args.cohort_limit,
        min_direct_references=args.min_direct_references,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _cmd_extended_validation(args: argparse.Namespace) -> int:
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    results = run_extended_validation(output_dir=args.output_dir, seeds=seeds, runs_per_seed=args.runs_per_seed)
    write_extended_report(report_path=args.report_path, results=results)
    print(
        json.dumps(
            {
                "output_dir": args.output_dir,
                "report_path": args.report_path,
                "total_runs": results["summary"]["config"]["total_runs"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_fp_forensics(args: argparse.Namespace) -> int:
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    eval_seeds = [int(x.strip()) for x in args.eval_seeds.split(",") if x.strip()]
    results = run_extended_validation(output_dir=args.output_dir, seeds=seeds, runs_per_seed=args.runs_per_seed)
    write_extended_report(report_path=args.report_path, results=results)

    fp_rows = analyze_independent_false_positives(
        seeds=seeds,
        runs_per_seed=args.runs_per_seed,
        threshold=args.threshold,
        retrieval_cfg=default_retrieval_config(),
    )
    from tide.io_utils import write_jsonl, write_json

    write_jsonl(f"{args.output_dir}/independent_false_positives.jsonl", fp_rows)
    write_false_positive_report("reports/false_positive_analysis.md", fp_rows)

    method_results = run_heldout_method_comparison(
        train_seeds=seeds, eval_seeds=eval_seeds, runs_per_seed=args.runs_per_seed
    )
    write_json(f"{args.output_dir}/heldout_method_comparison.json", method_results)
    write_hard_negative_report(
        "reports/hard_negative_benchmark.md",
        method_results=method_results,
        summary_after=results["summary"],
    )
    print(
        json.dumps(
            {
                "output_dir": args.output_dir,
                "false_positive_report": "reports/false_positive_analysis.md",
                "hard_negative_report": "reports/hard_negative_benchmark.md",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TIDE MVP pipeline tools")
    sub = parser.add_subparsers(dest="command", required=True)

    p_synth = sub.add_parser("run-synthetic", help="Run synthetic end-to-end inference pipeline")
    p_synth.add_argument("--output-dir", default="outputs/synthetic", help="Output directory")
    p_synth.add_argument("--seed", type=int, default=7, help="Random seed for synthetic generator")
    p_synth.add_argument("--min-edge-score", type=float, default=0.4, help="Graph edge score threshold")
    p_synth.set_defaults(func=_cmd_run_synthetic)

    p_audit = sub.add_parser("audit-dataset", help="Audit local dataset schema and access state")
    p_audit.add_argument("--dataset-dir", default="data", help="Directory containing local dataset files")
    p_audit.add_argument("--output", default=None, help="Optional JSON output path")
    p_audit.set_defaults(func=_cmd_audit_dataset)

    p_norm = sub.add_parser("normalize-events", help="Normalize one source table into TIDE event schema")
    p_norm.add_argument("--input", required=True, help="Input .jsonl or .csv file")
    p_norm.add_argument("--source-table", default=None, help="Logical source table name")
    p_norm.add_argument("--limit", type=int, default=None, help="Max rows to normalize")
    p_norm.add_argument("--output", required=True, help="Output JSONL path")
    p_norm.add_argument("--raw-output", default=None, help="Optional JSONL path to preserve raw source records")
    p_norm.set_defaults(func=_cmd_normalize_events)

    p_pass1 = sub.add_parser("pass1", help="Annotate source events using Adaption")
    p_pass1.add_argument("--dataset-dir", default="data/processed/ai-village/main-village/2026-10-24_to_2026-10-31/v1")
    p_pass1.add_argument("--output-dir", required=True, help="Separate output directory for annotations and checkpoints")
    p_pass1.add_argument("--model", default="deepseek-v4-flash", help="Provider model identifier (default: deepseek-v4-flash)")
    p_pass1.add_argument("--api-key-env", default="ADAPTION_API_KEY", help="Environment variable containing the API key")
    p_pass1.add_argument("--concurrency", type=int, default=1, help="Maximum simultaneous individual requests")
    p_pass1.add_argument("--limit", type=int, default=None, help="Process only the first N non-consolidation events")
    p_pass1.add_argument("--timeout", type=float, default=120, help="Request timeout in seconds")
    p_pass1.add_argument("--retries", type=int, default=3, help="Retries for transient request failures")
    p_pass1.add_argument("--overwrite", action="store_true", help="Replace existing Pass 1 annotations and checkpoints")
    p_pass1.add_argument("--retry-errors-only", action="store_true", help="Retry only events recorded in pass1_errors.jsonl")
    p_pass1.set_defaults(func=_cmd_pass1)

    p_pass2 = sub.add_parser("pass2", help="Retrieve and rerank historical event dependencies")
    p_pass2.add_argument("--dataset-dir", required=True, help="Processed dataset directory containing events.jsonl and agents.jsonl")
    p_pass2.add_argument("--pass1-dir", required=True, help="Pass 1 output directory containing annotations or checkpoint")
    p_pass2.add_argument("--output-dir", required=True, help="Separate Pass 2 output directory")
    p_pass2.add_argument("--model", default="deepseek-v4-flash", help="DeepSeek model identifier")
    p_pass2.add_argument("--api-key-env", default="ADAPTION_API_KEY", help="Environment variable containing the API key")
    p_pass2.add_argument("--embedding-model", default=EMBEDDING_MODEL, help="Hugging Face embedding model")
    p_pass2.add_argument("--embedding-revision", default=EMBEDDING_REVISION, help="Pinned Hugging Face model revision")
    p_pass2.add_argument("--precision", choices=("auto", "float16", "float32"), default="auto", help="Embedding precision; auto uses FP16 on CUDA")
    p_pass2.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    p_pass2.add_argument("--limit", type=int, default=None, help="Rerank only the first N annotated target events")
    p_pass2.add_argument("--timeout", type=float, default=120, help="DeepSeek request timeout in seconds")
    p_pass2.add_argument("--retries", type=int, default=3, help="Retries for transient DeepSeek request failures")
    p_pass2.add_argument("--concurrency", type=int, default=1, help="Maximum simultaneous DeepSeek reranking requests")
    p_pass2.add_argument("--candidate-group-size", type=int, default=4, help="Maximum candidates sent per target request")
    p_pass2.add_argument("--candidate-filter", choices=("none", "top10", "proposed"), default="none", help="Filter the cached candidate list before reranking")
    p_pass2.add_argument("--overwrite", action="store_true", help="Replace existing Pass 2 outputs and checkpoints")
    p_pass2.set_defaults(func=_cmd_pass2)

    p_pass2_embed = sub.add_parser("pass2-embed", help="Run or resume Pass 2 event embedding")
    p_pass2_embed.add_argument("--dataset-dir", required=True, help="Processed dataset directory containing events.jsonl and agents.jsonl")
    p_pass2_embed.add_argument("--output-dir", required=True, help="Shared Pass 2 cache/output directory")
    p_pass2_embed.add_argument("--embedding-model", default=EMBEDDING_MODEL, help="Hugging Face embedding model")
    p_pass2_embed.add_argument("--embedding-revision", default=EMBEDDING_REVISION, help="Pinned Hugging Face model revision")
    p_pass2_embed.add_argument("--precision", choices=("auto", "float16", "float32"), default="auto", help="Embedding precision; auto uses FP16 on CUDA")
    p_pass2_embed.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    p_pass2_embed.add_argument("--overwrite", action="store_true", help="Reset the embedding cache and checkpoint")
    p_pass2_embed.set_defaults(func=_cmd_pass2_embed)

    p_pass2_retrieve = sub.add_parser("pass2-retrieve", help="Generate and cache Pass 2 historical candidates")
    p_pass2_retrieve.add_argument("--dataset-dir", required=True, help="Processed dataset directory containing events.jsonl and agents.jsonl")
    p_pass2_retrieve.add_argument("--pass1-dir", required=True, help="Pass 1 output directory containing annotations or checkpoint")
    p_pass2_retrieve.add_argument("--output-dir", required=True, help="Shared Pass 2 cache/output directory")
    p_pass2_retrieve.add_argument("--limit", type=int, default=None, help="Retrieve candidates for only the first N annotated targets")
    p_pass2_retrieve.add_argument("--overwrite", action="store_true", help="Reset the retrieval cache and checkpoint")
    p_pass2_retrieve.set_defaults(func=_cmd_pass2_retrieve)

    p_pass2_rerank = sub.add_parser("pass2-rerank", help="Run or resume Pass 2 LLM dependency reranking")
    p_pass2_rerank.add_argument("--dataset-dir", required=True, help="Processed dataset directory containing events.jsonl and agents.jsonl")
    p_pass2_rerank.add_argument("--output-dir", required=True, help="Shared Pass 2 cache/output directory")
    p_pass2_rerank.add_argument("--model", default="deepseek-v4-flash", help="DeepSeek model identifier")
    p_pass2_rerank.add_argument("--api-key-env", default="ADAPTION_API_KEY", help="Environment variable containing the API key")
    p_pass2_rerank.add_argument("--timeout", type=float, default=120, help="DeepSeek request timeout in seconds")
    p_pass2_rerank.add_argument("--retries", type=int, default=3, help="Retries for transient DeepSeek request failures")
    p_pass2_rerank.add_argument("--concurrency", type=int, default=1, help="Maximum simultaneous DeepSeek reranking requests")
    p_pass2_rerank.add_argument("--candidate-group-size", type=int, default=4, help="Maximum candidates sent per target request")
    p_pass2_rerank.add_argument("--candidate-filter", choices=("none", "top10", "proposed"), default="none", help="Filter the cached candidate list before reranking")
    p_pass2_rerank.add_argument("--limit", type=int, default=None, help="Rerank only the first N chronological targets")
    p_pass2_rerank.add_argument("--overwrite", action="store_true", help="Reset the reranking cache and checkpoint")
    p_pass2_rerank.set_defaults(func=_cmd_pass2_rerank)

    p_public_embed = sub.add_parser("pass2-public-embed", help="Embed Pass 1 public summaries only")
    p_public_embed.add_argument("--dataset-dir", required=True, help="Processed dataset directory")
    p_public_embed.add_argument("--pass1-dir", required=True, help="Pass 1 directory containing public summaries")
    p_public_embed.add_argument("--output-dir", required=True, help="Separate public-context Pass 2 output directory")
    p_public_embed.add_argument("--embedding-model", default=EMBEDDING_MODEL, help="Hugging Face embedding model")
    p_public_embed.add_argument("--embedding-revision", default=EMBEDDING_REVISION, help="Pinned Hugging Face model revision")
    p_public_embed.add_argument("--precision", choices=("auto", "float16", "float32"), default="auto", help="Embedding precision; auto uses FP16 on CUDA")
    p_public_embed.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    p_public_embed.add_argument("--overwrite", action="store_true", help="Reset only the public embedding cache")
    p_public_embed.set_defaults(func=_cmd_pass2_public_embed)

    p_public_retrieve = sub.add_parser("pass2-public-retrieve", help="Retrieve candidates using public-summary embeddings")
    p_public_retrieve.add_argument("--dataset-dir", required=True, help="Processed dataset directory")
    p_public_retrieve.add_argument("--pass1-dir", required=True, help="Pass 1 directory containing summaries and local references")
    p_public_retrieve.add_argument("--output-dir", required=True, help="Separate public-context Pass 2 output directory")
    p_public_retrieve.add_argument("--limit", type=int, default=None, help="Retrieve candidates for only the first N annotated targets")
    p_public_retrieve.add_argument("--overwrite", action="store_true", help="Reset only the public retrieval cache")
    p_public_retrieve.set_defaults(func=_cmd_pass2_public_retrieve)

    p_public_rerank = sub.add_parser("pass2-public-rerank", help="Rerank candidates using public summaries only")
    p_public_rerank.add_argument("--dataset-dir", required=True, help="Processed dataset directory")
    p_public_rerank.add_argument("--pass1-dir", required=True, help="Pass 1 directory containing public summaries")
    p_public_rerank.add_argument("--output-dir", required=True, help="Separate public-context Pass 2 output directory")
    p_public_rerank.add_argument("--model", default="deepseek-v4-flash", help="DeepSeek model identifier")
    p_public_rerank.add_argument("--api-key-env", default="ADAPTION_API_KEY", help="Environment variable containing the API key")
    p_public_rerank.add_argument("--timeout", type=float, default=120, help="DeepSeek request timeout in seconds")
    p_public_rerank.add_argument("--retries", type=int, default=3, help="Retries for transient DeepSeek request failures")
    p_public_rerank.add_argument("--concurrency", type=int, default=1, help="Maximum simultaneous DeepSeek requests")
    p_public_rerank.add_argument("--candidate-group-size", type=int, default=4, help="Maximum candidates sent per target request")
    p_public_rerank.add_argument("--candidate-filter", choices=("none", "top10", "proposed"), default="none", help="Filter the cached candidate list before reranking")
    p_public_rerank.add_argument("--limit", type=int, default=None, help="Rerank only the first N chronological targets")
    p_public_rerank.add_argument("--overwrite", action="store_true", help="Reset only the public rerank cache")
    p_public_rerank.set_defaults(func=_cmd_pass2_public_rerank)

    p_analysis = sub.add_parser("analyze-graphs", help="Analyze Pass 1 and Pass 2 event and agent graphs")
    p_analysis.add_argument("--dataset-dir", required=True, help="Processed dataset directory")
    p_analysis.add_argument("--pass1-dir", required=True, help="Pass 1 annotations directory")
    p_analysis.add_argument("--pass2-dir", required=True, help="Private or public staged Pass 2 directory")
    p_analysis.add_argument("--output-dir", required=True, help="Empty directory for analysis results")
    p_analysis.add_argument("--similarity-threshold", required=True, type=float, help="Inclusive retrieval similarity threshold")
    p_analysis.add_argument("--dependency-threshold", required=True, type=float, help="Inclusive reranked dependency threshold")
    p_analysis.add_argument("--seed", type=int, default=0, help="Leiden random seed")
    p_analysis.add_argument("--cohort-retrieval", default=None, help="Retrieval JSONL defining a common target cohort")
    p_analysis.add_argument("--cohort-limit", type=int, default=None, help="First N chronological targets in the cohort retrieval cache")
    p_analysis.add_argument("--min-direct-references", type=int, default=1, help="Minimum direct Pass 1 references per ordered agent pair")
    p_analysis.set_defaults(func=_cmd_analyze_graphs)

    p_ext = sub.add_parser("run-extended-validation", help="Run extended synthetic benchmark and write report")
    p_ext.add_argument("--output-dir", default="outputs/extended_validation", help="Output directory for machine-readable results")
    p_ext.add_argument("--report-path", default="reports/extended_synthetic_validation.md", help="Markdown report path")
    p_ext.add_argument("--seeds", default="7,13,29", help="Comma-separated integer seeds")
    p_ext.add_argument("--runs-per-seed", type=int, default=3, help="Independent synthetic runs per seed")
    p_ext.set_defaults(func=_cmd_extended_validation)

    p_fp = sub.add_parser("run-fp-forensics", help="Run false-positive forensics and held-out comparison")
    p_fp.add_argument("--output-dir", default="outputs/fp_forensics", help="Output directory")
    p_fp.add_argument("--report-path", default="reports/extended_synthetic_validation.md", help="Extended report path")
    p_fp.add_argument("--seeds", default="7,13,29", help="Train/development seeds")
    p_fp.add_argument("--eval-seeds", default="101,103,107", help="Held-out evaluation seeds")
    p_fp.add_argument("--runs-per-seed", type=int, default=3, help="Runs per seed")
    p_fp.add_argument("--threshold", type=float, default=0.5, help="False-positive analysis threshold")
    p_fp.set_defaults(func=_cmd_fp_forensics)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
