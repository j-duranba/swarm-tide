from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any

import igraph as ig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score, precision_recall_curve

from tide.calibration import _candidate_scores, _checked_rerank_inputs
from tide.graph_analysis import read_snapshot
from tide.io_utils import write_json
from tide.pass2 import file_sha256, load_agent_registry, load_events


def compare_edge_sets(reference: set[tuple[str, str]], predicted: set[tuple[str, str]]) -> dict[str, Any]:
    shared = len(reference & predicted)
    reference_only = len(reference - predicted)
    tide_only = len(predicted - reference)
    precision = shared / len(predicted) if predicted else None
    recall = shared / len(reference) if reference else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else (0.0 if precision is not None and recall is not None else None)
    union = len(reference | predicted)
    return {"reference_edges": len(reference), "tide_edges": len(predicted), "shared": shared,
            "reference_only": reference_only, "tide_only": tide_only,
            "precision": precision, "recall": recall, "f1": f1,
            "jaccard": shared / union if union else None}


def reference_mention_matches(content: str, agents: dict[str, dict[str, Any]]) -> list[tuple[str, str]]:
    aliases: dict[str, set[str]] = defaultdict(set)
    for agent_id, agent in agents.items():
        name = agent["name"]
        aliases[name.lower()].add(agent_id)
        aliases[re.sub(r"\s+", "-", name.lower())].add(agent_id)
        model = agent.get("model_string")
        if isinstance(model, str) and "://" not in model:
            aliases[model.rsplit("/", 1)[-1].lower()].add(agent_id)
    unique = {alias: next(iter(ids)) for alias, ids in aliases.items() if len(ids) == 1}
    if not unique:
        return []
    patterns = sorted(unique, key=len, reverse=True)
    expression = r"(?<![\w@])@(" + "|".join(re.escape(alias) for alias in patterns) + r")(?![\w-]|\.[0-9])"
    return [(unique[match.group(1).lower()], match.group(0))
            for match in re.finditer(expression, content, re.IGNORECASE)]


def reference_mentions(content: str, agents: dict[str, dict[str, Any]]) -> set[str]:
    return {agent_id for agent_id, _ in reference_mention_matches(content, agents)}


def aggregate_max_scores(rows: list[dict[str, Any]], authors: dict[str, str | None],
                         valid_agents: set[str], target_ids: set[str], score_key: str) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}
    for row in rows:
        target_id = row["target_event_id"]
        if target_id not in target_ids:
            continue
        source, target = authors.get(row["source_event_id"]), authors.get(target_id)
        if source not in valid_agents or target not in valid_agents or source == target:
            continue
        value = row[score_key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid event score")
        pair = (source, target)
        scores[pair] = max(scores.get(pair, 0.0), float(value))
    return scores


def precision_recall(reference: set[tuple[str, str]], scores: dict[tuple[str, str], float],
                     universe: set[tuple[str, str]] | None = None) -> dict[str, Any]:
    if not reference:
        return {"average_precision": None, "curve": [], "unscored_reference_edges": 0}
    universe = universe if universe is not None else reference | set(scores)
    if not reference <= universe or not set(scores) <= universe:
        raise ValueError("Ranking universe must include reference and scored pairs")
    pairs = sorted(universe)
    labels = [int(pair in reference) for pair in pairs]
    values = [scores.get(pair, 0.0) for pair in pairs]
    precision, recall, thresholds = precision_recall_curve(labels, values)
    curve = [{"threshold": float(threshold), "precision": float(point_precision), "recall": float(point_recall)}
             for threshold, point_precision, point_recall in zip(thresholds, precision[:-1], recall[:-1])]
    curve.append({"threshold": None, "precision": float(precision[-1]), "recall": float(recall[-1])})
    return {"average_precision": float(average_precision_score(labels, values)), "curve": curve,
            "prevalence": sum(labels) / len(labels), "positive_pairs": sum(labels),
            "unscored_reference_edges": len(reference - set(scores)), "ranking_universe": len(universe),
            "ap_implementation": "sklearn.metrics.average_precision_score"}


def community_comparison(left: list[int], right: list[int]) -> dict[str, float]:
    if len(left) != len(right) or not left:
        raise ValueError("Community partitions must have the same nonzero size")
    return {"ari": ig.compare_communities(left, right, method="adjusted_rand"),
            "nmi": ig.compare_communities(left, right, method="nmi")}


def community_contingency(reference: list[int], public: list[int]) -> dict[str, Any]:
    if len(reference) != len(public) or not reference:
        raise ValueError("Community partitions must have the same nonzero size")
    reference_labels = sorted(set(reference))
    public_labels = sorted(set(public))
    counts = [[sum(left == left_label and right == right_label for left, right in zip(reference, public))
               for right_label in public_labels] for left_label in reference_labels]
    rows, columns = linear_sum_assignment([[-count for count in row] for row in counts])
    matched = [{"reference_community": reference_labels[row], "public_community": public_labels[column],
                "overlap": counts[row][column]} for row, column in zip(rows, columns)]
    return {"reference_labels": reference_labels, "public_labels": public_labels, "counts": counts,
            "max_overlap_matches": matched}


def event_connection_summary(target_ids: set[str], connection_counts: dict[str, int]) -> dict[str, Any]:
    connected = [connection_counts.get(target, 0) for target in target_ids if connection_counts.get(target, 0)]
    return {"eligible_target_events": len(target_ids), "connected_target_events": len(connected),
            "target_events_with_no_connections": len(target_ids) - len(connected),
            "event_links": sum(connected),
            "mean_incoming_links_per_connected_target": sum(connected) / len(connected) if connected else 0.0,
            "median_incoming_links_per_connected_target": sorted(connected)[len(connected) // 2] if connected else 0}


def build_event_mention_reference(events: list[dict[str, Any]], agents: dict[str, dict[str, Any]],
                                 target_ids: set[str]) -> tuple[set[tuple[str, str]], dict[str, Any], dict[tuple[str, str], list[str]]]:
    by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event["author_id"] in agents and event["timestamp_dt"] is not None:
            by_agent[event["author_id"]].append(event)
    for rows in by_agent.values():
        rows.sort(key=lambda event: (event["timestamp_dt"], event["source_index"]))
    agent_times = {agent_id: [event["timestamp_dt"] for event in rows] for agent_id, rows in by_agent.items()}

    links: set[tuple[str, str]] = set()
    mention_tokens_by_link: dict[tuple[str, str], list[str]] = {}
    mention_pairs = 0
    distinct_mentioned: set[str] = set()
    mentions_with_predecessor = 0
    mentions_without_predecessor = 0
    self_mentions = 0
    chat_targets = 0
    targets_with_reference: set[str] = set()
    for target in events:
        if target["event_id"] not in target_ids or not isinstance(target.get("chat_content"), str):
            continue
        chat_targets += 1
        author = target["author_id"]
        matched_mentions = reference_mention_matches(target["chat_content"], agents)
        tokens_by_agent: dict[str, list[str]] = defaultdict(list)
        for agent_id, token in matched_mentions:
            if token not in tokens_by_agent[agent_id]:
                tokens_by_agent[agent_id].append(token)
        mentions = set(tokens_by_agent)
        mention_pairs += len(mentions)
        distinct_mentioned.update(mentions)
        for mentioned in mentions:
            if mentioned == author:
                self_mentions += 1
                continue
            predecessors = by_agent.get(mentioned, [])
            times = agent_times.get(mentioned, [])
            timestamp = target["timestamp_dt"]
            if timestamp is None:
                mentions_without_predecessor += 1
                continue
            position = bisect_left(times, timestamp) - 1
            if position < 0:
                mentions_without_predecessor += 1
                continue
            source_id = predecessors[position]["event_id"]
            if source_id == target["event_id"]:
                self_mentions += 1
                continue
            pair = (source_id, target["event_id"])
            links.add(pair)
            mention_tokens_by_link[pair] = tokens_by_agent[mentioned]
            mentions_with_predecessor += 1
            targets_with_reference.add(target["event_id"])
    return links, {
        "eligible_target_events": len(target_ids), "targets_with_chat_content": chat_targets,
        "explicit_mention_agent_pairs": mention_pairs, "distinct_mentioned_agents": len(distinct_mentioned),
        "mentions_with_eligible_predecessor": mentions_with_predecessor,
        "mentions_without_eligible_predecessor": mentions_without_predecessor,
        "self_mentions_excluded": self_mentions, "reference_event_links": len(links),
        "targets_with_at_least_one_reference_link": len(targets_with_reference),
        "target_proportion_with_reference_link": len(targets_with_reference) / len(target_ids) if target_ids else 0.0,
    }, mention_tokens_by_link


def evaluate_event_links(reference: set[tuple[str, str]], scores: dict[tuple[str, str], float],
                         events: list[dict[str, Any]], target_ids: set[str], threshold: float) -> dict[str, Any]:
    ordered = [event for event in events if event["event_id"] in target_ids]
    pair_count = 0
    positive = 0
    true_positive = 0
    predicted_count = 0
    reference_scored = 0
    label_values: list[int] = []
    score_values: list[float] = []
    for target in ordered:
        target_time = target["timestamp_dt"]
        for source in events:
            if source["event_id"] == target["event_id"] or target_time is None or source["timestamp_dt"] is None:
                continue
            if source["timestamp_dt"] >= target_time:
                continue
            pair = (source["event_id"], target["event_id"])
            pair_count += 1
            is_positive = pair in reference
            score = scores.get(pair, 0.0)
            label_values.append(int(is_positive))
            score_values.append(score)
            positive += is_positive
            predicted = score >= threshold
            predicted_count += predicted
            true_positive += predicted and is_positive
            reference_scored += is_positive and pair in scores
    false_positive = predicted_count - true_positive
    false_negative = positive - true_positive
    precision = true_positive / predicted_count if predicted_count else None
    recall = true_positive / positive if positive else None
    f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and precision + recall else (0.0 if precision is not None and recall is not None else None)
    union = positive + predicted_count - true_positive
    return {
        "evaluation_universe": "all strictly earlier cohort events -> jointly completed holdout targets",
        "eligible_event_pairs": pair_count, "reference_event_links": positive,
        "scored_event_pairs": len(scores), "reference_links_with_a_tide_score": reference_scored,
        "unscored_reference_links": positive - reference_scored,
        "predicted_event_links_at_threshold": predicted_count,
        "shared": true_positive, "reference_only": false_negative, "tide_only": false_positive,
        "precision": precision, "recall": recall, "f1": f1,
        "jaccard": true_positive / union if union else None,
        "average_precision": float(average_precision_score(label_values, score_values)) if positive else None,
        "positive_prevalence": positive / pair_count if pair_count else None,
        "average_precision_implementation": "sklearn.metrics.average_precision_score",
        "unobserved_pairs_are_verified_negatives": False,
    }


def cohort_agent_ids(events: list[dict[str, Any]], agents: dict[str, dict[str, Any]]) -> list[str]:
    return sorted({event["author_id"] for event in events if event["author_id"] in agents})


def build_community_graph(agent_ids: list[str], scores: dict[tuple[str, str], float], seed: int = 0) -> tuple[ig.Graph, list[int], float | None]:
    indices = {agent_id: index for index, agent_id in enumerate(agent_ids)}
    pairs = sorted(scores)
    graph = ig.Graph(n=len(agent_ids), edges=[(indices[source], indices[target]) for source, target in pairs], directed=True)
    graph.es["weight"] = [scores[pair] for pair in pairs]
    projected = graph.as_undirected(mode="collapse", combine_edges={"weight": "sum"})
    ig.set_random_number_generator(random.Random(seed))
    if not projected.ecount():
        return graph, list(range(len(agent_ids))), None
    partition = projected.community_leiden(objective_function="modularity", weights="weight", n_iterations=-1)
    return graph, partition.membership, projected.modularity(partition.membership, weights="weight")


def _read_edge_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _draw_precision_recall(results: dict[str, Any], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.7), layout="constrained")
    colors = {"Explicit mentions": "#b44b37", "Private inference": "#226f72"}
    variant_names = {"sigmoid_decay": "Public TIDE", "ranking_only": "Public TIDE (no temporal)"}
    for (reference, variant), result in results.items():
        if result["curve"]:
            curve = sorted(result["curve"], key=lambda point: point["recall"])
            style = "-" if variant == "sigmoid_decay" else "--"
            axis.step([point["recall"] for point in curve], [point["precision"] for point in curve],
                      where="post", color=colors[reference], linestyle=style, linewidth=2.0,
                        label=f"{reference}, {variant_names[variant]} "
                            f"(AP={result['average_precision']:.3f})")
    for reference, color in colors.items():
        prevalence = next(result["prevalence"] for (name, _), result in results.items() if name == reference)
        axis.axhline(prevalence, color=color, linewidth=0.8, linestyle=":", alpha=0.6)
    axis.set(xlabel="Recall relative to reference", ylabel="Precision relative to reference", xlim=(0, 1.02), ylim=(0, 1.02))
    axis.set_title("Precision-Recall Against Reference Networks", fontsize=16, weight="bold", pad=12)
    axis.grid(alpha=0.2)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8, frameon=False)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"precision_recall.{extension}", dpi=300)
    plt.close(fig)


def _draw_networks(agent_ids: list[str], graphs: dict[str, ig.Graph], memberships: dict[str, list[int]],
                   contingencies: dict[str, dict[str, Any]], output: Path) -> dict[str, dict[str, int]]:
    union = ig.Graph(n=len(agent_ids), directed=False)
    all_pairs = {tuple(sorted((edge.source, edge.target))) for graph in graphs.values() for edge in graph.es}
    union.add_edges(sorted(all_pairs))
    ig.set_random_number_generator(random.Random(0))
    active = [index for index, degree in enumerate(union.degree()) if degree]
    inactive = [index for index, degree in enumerate(union.degree()) if not degree]
    coordinates = [(0.0, 0.0)] * len(agent_ids)
    if active:
        layout = union.induced_subgraph(active).layout_fruchterman_reingold(niter=500)
        scale = max((max(abs(point[0]), abs(point[1])) for point in layout), default=1.0) or 1.0
        for index, point in zip(active, layout):
            coordinates[index] = (1.12 * point[0] / scale, 1.12 * point[1] / scale)
    for position, index in enumerate(inactive):
        angle = 2 * math.pi * position / len(inactive)
        coordinates[index] = (1.36 * math.cos(angle), 1.28 * math.sin(angle))
    primary_public = "Public TIDE"
    ranking_public = "Public TIDE (no temporal)"
    public_labels = sorted(set(memberships[primary_public]))
    color_maps: dict[str, dict[int, int]] = {primary_public: {label: index for index, label in enumerate(public_labels)}}
    next_color = len(public_labels)
    for label in ("Explicit mentions", "Private inference"):
        aligned = {item["reference_community"]: item["public_community"]
                   for item in contingencies[label]["max_overlap_matches"] if item["overlap"] > 0}
        color_maps[label] = {}
        for community in sorted(set(memberships[label])):
            if community in aligned:
                color_maps[label][community] = color_maps[primary_public][aligned[community]]
            else:
                color_maps[label][community] = next_color
                next_color += 1
    primary_colors = color_maps[primary_public]
    aligned_ranking = {item["reference_community"]: item["public_community"]
                       for item in contingencies[ranking_public]["max_overlap_matches"] if item["overlap"] > 0}
    color_maps[ranking_public] = {}
    for community in sorted(set(memberships[ranking_public])):
        if community in aligned_ranking:
            color_maps[ranking_public][community] = primary_colors[aligned_ranking[community]]
        else:
            color_maps[ranking_public][community] = next_color
            next_color += 1
    palette = plt.get_cmap("turbo", max(1, next_color))
    figure = plt.figure(figsize=(16, 4.8), layout="constrained")
    figure.suptitle("Graph Layout and Identified Communities", fontsize=18, weight="bold")
    grid = figure.add_gridspec(1, 4)
    axes = [figure.add_subplot(grid[0, index]) for index in range(4)]
    for axis, (label, graph) in zip(axes, graphs.items()):
        for edge in graph.es:
            source, target = coordinates[edge.source], coordinates[edge.target]
            axis.add_patch(FancyArrowPatch(source, target, arrowstyle="-|>", mutation_scale=5,
                                          linewidth=0.45, color="#56616a", alpha=0.27,
                                          shrinkA=4, shrinkB=4))
        community_sizes = {community: memberships[label].count(community) for community in set(memberships[label])}
        for index, (x, y) in enumerate(coordinates):
            community = memberships[label][index]
            color = "#d2d6d7" if community_sizes[community] == 1 else palette(color_maps[label][community])
            axis.scatter([x], [y], s=55, c=[color],
                         edgecolors="white", linewidths=0.4, zorder=3)
            axis.text(x, y, str(index + 1), fontsize=3.5, ha="center", va="center", zorder=4)
        axis.set_title(label, fontsize=11, weight="bold")
        axis.set_xlim(-1.43, 1.43)
        axis.set_ylim(-1.34, 1.34)
        axis.set_aspect("equal")
        axis.axis("off")
    for extension in ("png", "pdf"):
        figure.savefig(output / f"communities.{extension}", dpi=300)
    plt.close(figure)
    return color_maps


def _draw_community_heatmaps(contingencies: dict[str, dict[str, Any]], output: Path) -> None:
    panels = (("Explicit mentions", "Public TIDE"), ("Private inference", "Public TIDE"),
              ("Public TIDE (no temporal)", "Public TIDE"))
    figure, heatmap_axes = plt.subplots(1, 3, figsize=(14, 5), layout="constrained")
    figure.suptitle("Community Membership Overlap", fontsize=18, weight="bold")
    for axis, (label, public_label) in zip(heatmap_axes, panels):
        contingency = contingencies[label]
        matrix = contingency["counts"]
        image = axis.imshow(matrix, aspect="auto", cmap="Blues", interpolation="nearest")
        axis.set_title(f"{label} vs {public_label}", fontsize=10, weight="bold")
        axis.set_xlabel(f"{public_label} community")
        axis.set_ylabel("Reference community")
        axis.set_xticks(range(len(contingency["public_labels"])),
                        [f"{value + 1}" for value in contingency["public_labels"]], fontsize=6)
        axis.set_yticks(range(len(contingency["reference_labels"])),
                        [f"{value + 1}" for value in contingency["reference_labels"]], fontsize=6)
        if matrix and matrix[0]:
            axis.set_xticks([value - 0.5 for value in range(1, len(contingency["public_labels"]))], minor=True)
            axis.set_yticks([value - 0.5 for value in range(1, len(contingency["reference_labels"]))], minor=True)
            axis.grid(which="minor", color="white", linewidth=0.5)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Agents")
    for extension in ("png", "pdf"):
        figure.savefig(output / f"community-overlap.{extension}", dpi=300)
    plt.close(figure)


def _write_compact_report(output: Path, result: dict[str, Any]) -> None:
    def number(value: float | None) -> str:
        return f"{value:.3f}" if value is not None else "n/a"

    scope, structures, comparisons = result["scope"], result["structures"], result["comparisons"]
    lines = [
        "# TIDE agent-network evaluation", "",
        f"**Scope.** {scope['registered_agents']} registered agents who authored events in the {scope['cohort_targets']}-event cohort; {scope['joint_completed_holdout']} jointly scored "
        f"holdout targets from {scope['holdout_targets']} after excluding the first {scope['training_targets_excluded']} calibration targets. "
        "Only candidate source events within the original cohort are used. Incomplete targets are excluded, not treated as negatives.", "",
        "**Networks.** Explicit = literal unambiguous @mentions in original agent chat, author to mentioned agent; "
        "a mention can be an address, citation or other reference, not necessarily information transfer. "
        "Private = raw private-context dependency scores; Public TIDE = sigmoid-calibrated, exponentially lag-decayed scores; "
        "Public TIDE (no temporal) = sigmoid-only ranking calibration. "
        "No independently calibrated private decay exists. Inference is source to target; all self-links are excluded. "
        "One mention defines an explicit edge; inferred edges require a maximum event score >= 0.5 by default "
        f"(actual private {result['policies']['private_threshold']:.2f}, public {result['policies']['public_threshold']:.2f}).", "",
        "| Network | Nodes | Edges | Density | Communities | Modularity |", "|---|---:|---:|---:|---:|---:|",
    ]
    for name, stats in structures.items():
        lines.append(f"| {name} | {stats['nodes']} | {stats['edges']} | {number(stats['density'])} | {stats['communities']} | {number(stats['modularity'])} |")
    lines += ["", "| Reference | Public variant | Shared | Reference only | TIDE only | Precision | Recall | F1 | Jaccard | AP | ARI | NMI |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, variants in comparisons.items():
        for variant, entry in variants["variants"].items():
            edge, community = entry["at_threshold"], entry["community"]
            variant_label = "Public TIDE" if variant == "sigmoid_decay" else "Public TIDE (no temporal)"
            lines.append(f"| {name} | {variant_label} | {edge['shared']} | {edge['reference_only']} | {edge['tide_only']} | "
                         f"{number(edge['precision'])} | {number(edge['recall'])} | {number(edge['f1'])} | {number(edge['jaccard'])} | "
                         f"{number(entry['average_precision'])} | {number(community['ari'])} | {number(community['nmi'])} |")
    lines += ["", "**Event connections** (cross-agent links into jointly completed holdout targets):",
              "", "| Network | Eligible targets | Connected targets | Event links | Mean links per connected target |",
              "|---|---:|---:|---:|---:|"]
    for name, stats in result["event_connections"].items():
        lines.append(f"| {name} | {stats['eligible_target_events']} | {stats['connected_target_events']} | "
                     f"{stats['event_links']} | {number(stats['mean_incoming_links_per_connected_target'])} |")
    event_reference = result["event_level_explicit_reference"]
    event_counts = event_reference["reference_counts"]
    event_metrics = event_reference["metrics"]
    agent_metrics = event_reference["comparison_to_agent_level"]["agent_level_explicit_reference"]
    lines += ["", "**Explicit-reference evaluation by level** (sigmoid+decay public scores):",
              "", f"The event proxy resolves {event_counts['mentions_with_eligible_predecessor']} of "
              f"{event_counts['explicit_mention_agent_pairs']} distinct target/mentioned-agent pairs across "
              f"{event_counts['distinct_mentioned_agents']} mentioned agents; "
              f"{event_counts['mentions_without_eligible_predecessor']} have no earlier cohort event, and "
              f"{event_counts['self_mentions_excluded']} are self-mentions. "
              f"{event_counts['targets_with_at_least_one_reference_link']} of {event_counts['eligible_target_events']} targets "
              f"({100 * event_counts['target_proportion_with_reference_link']:.1f}%) have a reference link. "
              "Unresolvable mentions are coverage gaps, not TIDE false negatives.", "",
              "| Level | Reference links | Shared | Reference only | TIDE only | Precision | Recall | F1 | Jaccard | AP | Prevalence |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    lines.append(f"| Agent pairs | {agent_metrics['reference_edges']} | {agent_metrics['shared']} | {agent_metrics['reference_only']} | "
                 f"{agent_metrics['tide_only']} | {number(agent_metrics['precision'])} | {number(agent_metrics['recall'])} | "
                 f"{number(agent_metrics['f1'])} | {number(agent_metrics['jaccard'])} | "
                 f"{number(comparisons['Explicit mentions']['variants']['sigmoid_decay']['average_precision'])} | "
                 f"{number(comparisons['Explicit mentions']['positive_prevalence'])} |")
    lines.append(f"| Event predecessor proxy | {event_metrics['reference_event_links']} | {event_metrics['shared']} | {event_metrics['reference_only']} | "
                 f"{event_metrics['tide_only']} | {number(event_metrics['precision'])} | {number(event_metrics['recall'])} | "
                 f"{number(event_metrics['f1'])} | {number(event_metrics['jaccard'])} | "
                 f"{number(event_metrics['average_precision'])} | {event_metrics['positive_prevalence']:.3e} |")
    lines += ["", f"Event-pair AP ranks {event_metrics['eligible_event_pairs']:,} strictly earlier cohort-event/holdout-target pairs; "
              f"positive prevalence is {event_metrics['positive_prevalence']:.3e}. Unscored eligible pairs receive score zero for ranking, "
              f"but unobserved links are not verified negatives. TIDE scored {event_metrics['scored_event_pairs']:,} pairs and "
              f"scored {event_metrics['reference_links_with_a_tide_score']} of the {event_metrics['reference_event_links']} proxy links.", ""]
    explicit = comparisons["Explicit mentions"]["variants"]["sigmoid_decay"]
    private = comparisons["Private inference"]["variants"]["sigmoid_decay"]
    private_stability = structures["Private inference"]["leiden_seed_ari_range"]
    public_stability = structures["Public TIDE"]["leiden_seed_ari_range"]
    ranking_stability = structures["Public TIDE (no temporal)"]["leiden_seed_ari_range"]
    explicit_active = explicit["community"]["active_only"] or {}
    private_active = private["community"]["active_only"] or {}
    lines += ["", f"**Results.** At the stated thresholds, public TIDE has reference-relative F1 "
              f"{number(explicit['at_threshold']['f1'])} against mentions and {number(private['at_threshold']['f1'])} "
              f"against private inference. All-author NMI is {number(explicit['community']['nmi'])} and "
              f"{number(private['community']['nmi'])}; active-only NMI is "
              f"{number(explicit_active.get('nmi'))} and {number(private_active.get('nmi'))}. "
              f"Across four alternate seeds, private, Public TIDE and Public TIDE (no temporal) partition ARI versus the default ranges "
              f"{number(private_stability[0])}-{number(private_stability[1])} and "
              f"{number(public_stability[0])}-{number(public_stability[1])}, and "
              f"{number(ranking_stability[0])}-{number(ranking_stability[1])}; community interpretation is tentative.", "",
              "**Interpretation.** These are reference-relative recovery measures, not classification accuracy: "
              "a TIDE-only edge is not necessarily false, and neither reference is ground truth. "
              f"Explicit-reference agreement falls from agent-level F1 {number(agent_metrics['f1'])} to event-level F1 {number(event_metrics['f1'])}: "
              "agent aggregation merges distinct event links into one author-pair edge and can obscure predecessor-level misses. "
              "The explicit network records mention direction, not necessarily dependency direction. "
              "Private inference and public TIDE use different candidate filters; the public calibrator was fit "
              "against private scores on earlier targets, so they are not independent. "
              "Held-out targets can still involve the same agent pairs seen during calibration. "
            "Precision-recall ranks all ordered non-self agent pairs, assigning unscored public pairs zero for ranking; "
              "reference absences are not verified negatives. AP is sklearn average precision on the same labels and scores "
              f"as the curve. Positive prevalences are {number(comparisons['Explicit mentions']['positive_prevalence'])} for mentions "
              f"and {number(comparisons['Private inference']['positive_prevalence'])} for private inference, and are shown as dashed baselines.", "",
              "The calibration ablation compares Public TIDE with Public TIDE (no temporal) on the same fit/holdout split; "
              "the paired rows above show the impact on edge and community metrics. Leiden uses weighted undirected projections "
              "with reciprocal weights summed; singleton isolates remain. "
              "ARI and NMI include all cohort authors, including shared isolates; active-only comparisons and "
              "four alternate-seed stability checks are in the metrics file. One layout is computed from the union graph "
              "and reused in all panels. Reference community labels are aligned to public labels by maximum-overlap Hungarian "
              "matching solely for color consistency; contingency matrices report the actual overlap. This alignment does not "
              "imply equivalent communities. Gray nodes are singleton communities.", "",
              "![Precision-recall against reference networks](precision_recall.png)", "",
              "![Graph layout and identified communities](communities.png)", "",
              "![Community overlap heatmaps](community-overlap.png)", "",
              "Node numbers are consistent across panels and map to agent IDs in metrics.json. "
              "Threshold sweeps, coverage, snapshot hashes and exact policies are also in metrics.json. "
              "Event-level mention-proxy counts and construction policy are in event_level_explicit_reference.json; "
              "auditable reference links with matched mention tokens and TIDE scores/status are in "
              "event_level_explicit_reference_edges.jsonl; thresholded event TIDE predictions are in "
              "event_level_tide_event_edges.jsonl.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_network_evaluation(dataset_dir: str | Path, registry_path: str | Path, public_dir: str | Path,
                           private_dir: str | Path, calibration_dir: str | Path, output_dir: str | Path,
                           *, threshold: float = 0.5, private_threshold: float = 0.5) -> dict[str, Any]:
    if not all(math.isfinite(value) and 0 < value <= 1 for value in (threshold, private_threshold)):
        raise ValueError("Thresholds must be greater than zero and at most one")
    dataset, registry, public, private, calibration, output = (Path(value).resolve() for value in
        (dataset_dir, registry_path, public_dir, private_dir, calibration_dir, output_dir))
    if any(output == source or source in output.parents for source in (dataset, registry.parent, public, private, calibration)):
        raise ValueError("Evaluation output must be separate from input directories")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Evaluation output directory must be empty")
    agents, _ = load_agent_registry(registry)
    slice_agents, _ = load_agent_registry(dataset / "agents.jsonl")
    if any(agent_id not in agents or agents[agent_id]["name"] != agent["name"] for agent_id, agent in slice_agents.items()):
        raise ValueError("Full registry does not match the dataset's agent identities")
    public_retrieval, public_config, retrieval_path = _checked_rerank_inputs(public, "public", dataset)
    private_retrieval, private_config, _ = _checked_rerank_inputs(private, "private", dataset)
    if public_retrieval.get("pass1_sha256") != private_retrieval.get("pass1_sha256"):
        raise ValueError("Public and private retrieval use different annotations")
    calibration_info = json.loads((calibration / "calibration.json").read_text(encoding="utf-8"))
    hashes = calibration_info["input_hashes"]
    for key in ("events_sha256", "agents_sha256", "pass1_sha256"):
        if hashes.get(key) != public_retrieval.get(key):
            raise ValueError(f"Calibration does not match {key}")
    if hashes.get("public_retrieval_sha256") != file_sha256(retrieval_path):
        raise ValueError("Calibration does not match the public retrieval cache")
    retrieval, retrieval_hash = read_snapshot(retrieval_path)
    public_rows, public_hash = read_snapshot(public / "rerank_checkpoint.jsonl")
    private_rows, private_hash = read_snapshot(private / "rerank_checkpoint.jsonl")
    if public_hash != hashes.get("public_rerank_snapshot_sha256"):
        raise ValueError("Public checkpoint changed since calibration; recalibrate first")
    if private_hash != hashes.get("private_rerank_snapshot_sha256"):
        raise ValueError("Private checkpoint changed since calibration; recalibrate first")
    events, _ = load_events(dataset / "events.jsonl", slice_agents, {})
    by_id = {event["event_id"]: event for event in events}
    retrieval_ids = {row["event_id"] for row in retrieval}
    if any(row["event_id"] not in by_id or row.get("source_index") != by_id[row["event_id"]]["source_index"] for row in retrieval):
        raise ValueError("Retrieval targets do not match the dataset")
    cohort_limit = calibration_info["cohort"]["targets"]
    train_count = calibration_info["cohort"]["training_targets"]
    cohort = [event for event in events if event["event_id"] in retrieval_ids][:cohort_limit]
    cohort_ids = {event["event_id"] for event in cohort}
    cohort_hash = hashlib.sha256("\n".join(sorted(cohort_ids)).encode("utf-8")).hexdigest()
    if len(cohort) != cohort_limit or cohort_hash != calibration_info["cohort"]["event_ids_sha256"]:
        raise ValueError("Calibration cohort does not match dataset")
    public_by_id = {row["event_id"]: row for row in public_rows}
    private_by_id = {row["event_id"]: row for row in private_rows}
    training_pairs = [row for row in _read_edge_rows(calibration / "matched_pairs.jsonl") if row["split"] == "train"]
    train_ids = {event["event_id"] for event in cohort[:train_count]}
    checked_private = {}
    for pair in training_pairs:
        target_id = pair["target_event_id"]
        if target_id not in train_ids or target_id not in private_by_id:
            raise ValueError("Calibration training target is absent from the private checkpoint")
        if target_id not in checked_private:
            checked_private[target_id] = _candidate_scores(private_by_id[target_id], by_id, cohort_ids)
        source = checked_private[target_id].get(pair["source_event_id"])
        if source is None or source["score"] != pair["private_score"]:
            raise ValueError("Calibration training scores do not match the private checkpoint")
    holdout = cohort[train_count:]
    selected = [event for event in holdout if event["event_id"] in public_by_id and event["event_id"] in private_by_id]
    target_ids = {event["event_id"] for event in selected}
    if not target_ids:
        raise ValueError("No holdout targets were completed in both rerank checkpoints")
    authors = {event["event_id"]: event["author_id"] for event in cohort}
    agent_ids = cohort_agent_ids(cohort, agents)
    valid_agents = set(agent_ids)
    public_pairs = {}
    private_event_edges = []
    for target_id in target_ids:
        for source_id, value in _candidate_scores(public_by_id[target_id], by_id, cohort_ids).items():
            public_pairs[(source_id, target_id)] = value["score"]
        for source_id, value in _candidate_scores(private_by_id[target_id], by_id, cohort_ids).items():
            private_event_edges.append({"source_event_id": source_id, "target_event_id": target_id, "score": value["score"]})
    calibrated_rows = _read_edge_rows(calibration / "public_decayed_edges.jsonl")
    selected_rows = [row for row in calibrated_rows if row["target_event_id"] in target_ids]
    seen = set()
    for row in selected_rows:
        pair = (row["source_event_id"], row["target_event_id"])
        if pair not in public_pairs or pair in seen or row.get("public_score") != public_pairs[pair]:
            raise ValueError(f"Calibrated public scores do not match the rerank checkpoint: {pair}")
        seen.add(pair)
    if seen != set(public_pairs):
        raise ValueError("Calibrated scores do not cover the public cohort pairs")
    public_scores = aggregate_max_scores(selected_rows, authors, valid_agents, target_ids, "calibrated_public_score")
    ranking_scores = aggregate_max_scores(selected_rows, authors, valid_agents, target_ids, "ranking_calibrated_public_score")
    private_scores = aggregate_max_scores(private_event_edges, authors, valid_agents, target_ids, "score")
    explicit_counts: dict[tuple[str, str], float] = defaultdict(float)
    explicit_event_connection_counts: dict[str, int] = defaultdict(int)
    chat_content_by_id: dict[str, str] = {}
    explicit_events = 0
    with (dataset / "events.jsonl").open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            record = row.get("source_event", {}).get("source_record", {})
            target_id = record.get("id")
            if target_id not in target_ids:
                continue
            data = record.get("data", {})
            author = authors.get(target_id)
            if not isinstance(data.get("content"), str):
                continue
            if author not in valid_agents or data.get("actionType") != "AGENT_TALK":
                continue
            chat_content_by_id[target_id] = data["content"]
            explicit_events += 1
            for mentioned in reference_mentions(data["content"], agents):
                if author != mentioned and mentioned in valid_agents:
                    explicit_counts[(author, mentioned)] += 1
                    explicit_event_connection_counts[target_id] += 1
    event_reference_rows = [{**event, "chat_content": chat_content_by_id.get(event["event_id"])} for event in cohort]
    event_reference_edges, event_reference_counts, mention_tokens_by_link = build_event_mention_reference(
        event_reference_rows, agents, target_ids
    )
    event_tide_scores = {
        (row["source_event_id"], row["target_event_id"]): float(row["sigmoid_decay_public_score"])
        for row in selected_rows if row["target_event_id"] in target_ids
    }
    event_level_metrics = evaluate_event_links(event_reference_edges, event_tide_scores, cohort, target_ids, threshold)
    event_level_result = {"construction_policy": {
        "target_events": "jointly completed holdout targets only",
        "predecessor_pool": "all events in the original 1,200-event cohort",
        "mention_source": "literal unambiguous @name/@slug or model alias in original event data.content for AGENT_TALK targets, deduplicated per target and mentioned agent",
        "predecessor_selection": "latest strictly earlier event by mentioned agent; temporal proximity only; no TIDE scores used",
        "self_mentions": "excluded from event links and reported separately",
        "duplicate_links": "deduplicated by source event ID and target event ID",
        "unresolvable_mentions": "reported as coverage gaps; excluded from reference links and false negatives",
        "tide_score": "sigmoid-plus-decay calibrated public event-pair score",
        "self_loops": "event self-links excluded; predecessor is strictly earlier",
    }, "reference_counts": event_reference_counts, "metrics": event_level_metrics,
       "comparison_to_agent_level": {}}
    explicit = set(explicit_counts)
    private_edges = {pair for pair, score in private_scores.items() if score >= private_threshold}
    public_edges = {pair for pair, score in public_scores.items() if score >= threshold}
    ranking_edges = {pair for pair, score in ranking_scores.items() if score >= threshold}
    references = {"Explicit mentions": explicit, "Private inference": private_edges}
    graphs_data = {"Explicit mentions": dict(explicit_counts),
                   "Private inference": {pair: private_scores[pair] for pair in private_edges},
                   "Public TIDE": {pair: public_scores[pair] for pair in public_edges},
                   "Public TIDE (no temporal)": {pair: ranking_scores[pair] for pair in ranking_edges}}
    graphs, memberships, structures = {}, {}, {}
    for name, edges in graphs_data.items():
        graph, membership, modularity = build_community_graph(agent_ids, edges)
        graphs[name], memberships[name] = graph, membership
        structures[name] = {"nodes": graph.vcount(), "edges": graph.ecount(),
                            "density": graph.density(loops=False), "communities": len(set(membership)),
                            "modularity": modularity, "isolates": sum(degree == 0 for degree in graph.degree())}
        alternative = [build_community_graph(agent_ids, edges, seed=seed)[1] for seed in range(1, 5)]
        stability = [community_comparison(membership, other)["ari"] for other in alternative]
        structures[name]["leiden_seed_ari_range"] = [min(stability), max(stability)]
    comparisons = {}
    pr_results = {}
    universe = {(source, target) for source in agent_ids for target in agent_ids if source != target}
    contingencies = {}
    public_variants = {
        "sigmoid_decay": ("Public TIDE", public_scores),
        "ranking_only": ("Public TIDE (no temporal)", ranking_scores),
    }
    for name, reference in references.items():
        variant_results = {}
        for variant, (graph_name, scores) in public_variants.items():
            edge_set = public_edges if variant == "sigmoid_decay" else ranking_edges
            edge_metrics = compare_edge_sets(reference, edge_set)
            curve = precision_recall(reference, scores, universe)
            pr_results[(name, variant)] = curve
            active = [index for index in range(len(agent_ids)) if graphs[name].degree(index) or graphs[graph_name].degree(index)]
            community = community_comparison(memberships[name], memberships[graph_name])
            community["active_nodes"] = len(active)
            community["active_only"] = community_comparison([memberships[name][index] for index in active],
                                                             [memberships[graph_name][index] for index in active]) if active else None
            contingency = community_contingency(memberships[name], memberships[graph_name])
            if variant == "sigmoid_decay":
                contingencies[name] = contingency
            elif name == "Explicit mentions":
                contingencies[graph_name] = community_contingency(
                    memberships[graph_name], memberships["Public TIDE"]
                )
            variant_results[variant] = {"at_threshold": edge_metrics, "community": community,
                                        "community_contingency": contingency,
                                        "average_precision": curve["average_precision"],
                                        "positive_prevalence": curve["prevalence"],
                                        "average_precision_implementation": curve["ap_implementation"],
                                        "ranking_universe_pairs": curve["ranking_universe"],
                                        "unscored_reference_edges": curve["unscored_reference_edges"],
                                        "threshold_sweep": [{"threshold": round(value / 20, 2), **compare_edge_sets(
                                            reference, {pair for pair, score in scores.items() if score >= value / 20})}
                                            for value in range(1, 21)]}
        comparisons[name] = {"positive_prevalence": variant_results["sigmoid_decay"]["positive_prevalence"],
                             "variants": variant_results}
    agent_explicit = comparisons["Explicit mentions"]["variants"]["sigmoid_decay"]["at_threshold"]
    event_level_result["comparison_to_agent_level"] = {
        "agent_level_explicit_reference": agent_explicit,
        "event_level_explicit_predecessor_reference": {key: event_level_metrics[key]
                                                        for key in ("reference_event_links", "shared", "reference_only",
                                                                    "tide_only", "precision", "recall", "f1", "jaccard",
                                                                    "average_precision", "positive_prevalence")},
        "interpretation": "Agent aggregation merges event pairs into max-scored author pairs; counts and agreement can change. Event-level links are temporal proxies for mention references, not verified information sources.",
    }
    agent_explicit = comparisons["Explicit mentions"]["variants"]["sigmoid_decay"]["at_threshold"]
    event_level_result["comparison_to_agent_level"] = {
        "agent_level_explicit_reference": agent_explicit,
        "event_level_explicit_predecessor_reference": {key: event_level_metrics[key]
                                                        for key in ("reference_event_links", "shared", "reference_only",
                                                                    "tide_only", "precision", "recall", "f1", "jaccard",
                                                                    "average_precision", "positive_prevalence")},
        "interpretation": "Agent aggregation merges event pairs into max-scored author pairs; counts and agreement can change. Event-level links are temporal proxies for mention references, not verified information sources.",
    }
    private_connection_counts: dict[str, int] = defaultdict(int)
    for edge in private_event_edges:
        if edge["score"] >= private_threshold and authors.get(edge["source_event_id"]) != authors.get(edge["target_event_id"]):
            private_connection_counts[edge["target_event_id"]] += 1
    public_connection_counts: dict[str, int] = defaultdict(int)
    for row in selected_rows:
        if row["calibrated_public_score"] >= threshold and authors.get(row["source_event_id"]) != authors.get(row["target_event_id"]):
            public_connection_counts[row["target_event_id"]] += 1
    ranking_connection_counts: dict[str, int] = defaultdict(int)
    for row in selected_rows:
        if row["ranking_calibrated_public_score"] >= threshold and authors.get(row["source_event_id"]) != authors.get(row["target_event_id"]):
            ranking_connection_counts[row["target_event_id"]] += 1
    event_connections = {
        "Explicit mentions": event_connection_summary(target_ids, dict(explicit_event_connection_counts)),
        "Private inference": event_connection_summary(target_ids, private_connection_counts),
        "Public TIDE": event_connection_summary(target_ids, public_connection_counts),
        "Public TIDE (no temporal)": event_connection_summary(target_ids, ranking_connection_counts),
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "event_level_explicit_reference.json", event_level_result)
    with (output / "event_level_explicit_reference_edges.jsonl").open("w", encoding="utf-8", newline="\n") as edge_output:
        for source_id, target_id in sorted(event_reference_edges):
            pair = (source_id, target_id)
            source, target = by_id[source_id], by_id[target_id]
            score = event_tide_scores.get(pair)
            edge_output.write(json.dumps({
                "source_event_id": source_id, "target_event_id": target_id,
                "source_agent_id": source["author_id"], "source_agent_name": source["author_name"],
                "target_agent_id": target["author_id"], "target_agent_name": target["author_name"],
                "source_timestamp": source["timestamp"], "target_timestamp": target["timestamp"],
                "elapsed_seconds": (target["timestamp_dt"] - source["timestamp_dt"]).total_seconds(),
                "matched_mention_tokens": mention_tokens_by_link[pair], "tide_score": score,
                "tide_scored": score is not None,
                "tide_predicted_at_threshold": score >= threshold if score is not None else None,
                "status": ("matched" if score is not None and score >= threshold else
                           "reference_only" if score is not None else "unscored_reference"),
            }, ensure_ascii=False) + "\n")
    with (output / "event_level_tide_event_edges.jsonl").open("w", encoding="utf-8", newline="\n") as edge_output:
        for (source_id, target_id), score in sorted(event_tide_scores.items()):
            if score < threshold:
                continue
            edge_output.write(json.dumps({"source_event_id": source_id, "target_event_id": target_id,
                                          "source_agent_id": authors.get(source_id), "target_agent_id": authors.get(target_id),
                                          "score": score, "threshold": threshold}, ensure_ascii=False) + "\n")
    color_maps = _draw_networks(agent_ids, graphs, memberships, contingencies, output)
    _draw_community_heatmaps(contingencies, output)
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {"registered_agents": len(agent_ids), "full_registry_agents": len(agents),
              "agent_universe": "registered authors of any event in the full calibration cohort",
              "cohort_targets": len(cohort), "training_targets_excluded": train_count,
                  "holdout_targets": len(holdout), "public_completed_holdout": sum(event["event_id"] in public_by_id for event in holdout),
                  "private_completed_holdout": sum(event["event_id"] in private_by_id for event in holdout),
                  "joint_completed_holdout": len(target_ids), "explicit_chat_events": explicit_events,
                  "agent_universe_pairs": len(universe)},
        "policies": {"explicit": "Literal unambiguous @name/@slug or model alias in source chat; author -> mentioned; at least one mention",
                     "private": "Raw private dependency score, threshold before max aggregation; no independently specified private decay",
                 "public_sigmoid_decay": "Calibrated sigmoid and exponential lag decay, maximum event score per agent pair",
                 "public_ranking_only": "Sigmoid-only score calibration with no elapsed-time term, maximum event score per agent pair",
                     "self_loops": "excluded from all networks and comparisons", "private_threshold": private_threshold,
                     "public_threshold": threshold,
                     "ranking": "all non-self ordered cohort-author pairs; missing public candidate assigned zero only for reference-relative ranking",
                     "communities": "weighted Leiden modularity on undirected projection; reciprocal weights summed; singleton isolates included; seed 0",
                     "community_colors": "one union-graph layout; reference labels maximum-overlap matched to sigmoid+decay public labels; ranking-only labels matched to sigmoid+decay labels",
                     "agent_aggregation": "maximum event score per ordered agent pair; explicit weights count mention events"},
        "input_hashes": {"registry_sha256": file_sha256(registry), "events_sha256": public_retrieval["events_sha256"],
                         "public_retrieval_sha256": retrieval_hash, "public_rerank_snapshot_sha256": public_hash,
                         "private_rerank_snapshot_sha256": private_hash,
                         "calibration_sha256": file_sha256(calibration / "calibration.json"),
                         "calibration_matched_pairs_sha256": file_sha256(calibration / "matched_pairs.jsonl"),
                         "calibrated_edges_sha256": file_sha256(calibration / "public_decayed_edges.jsonl"),
                         "cohort_event_ids_sha256": cohort_hash},
        "candidate_filters": {"public": public_config.get("candidate_filter"), "private": private_config.get("candidate_filter")},
        "structures": structures, "comparisons": comparisons, "event_connections": event_connections,
        "event_level_explicit_reference": event_level_result,
        "community_color_alignment": {label: {str(key): value for key, value in mapping.items()}
                          for label, mapping in color_maps.items()},
        "agents": [{"number": index + 1, "id": agent_id, "name": agents[agent_id]["name"]}
                   for index, agent_id in enumerate(agent_ids)],
    }
    write_json(output / "metrics.json", result)
    _draw_precision_recall(pr_results, output)
    _write_compact_report(output, result)
    return result