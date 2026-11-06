from __future__ import annotations

import re
from dataclasses import replace
from dataclasses import dataclass
from typing import Iterable

from tide.schema import CandidateRelation, EventRecord, SupportEdge

VALUE_RE = re.compile(r"\b(?:[A-Z]{2,}-\d+|cks-\d+|[a-z_]+=\S+|\d+%|\d+ms)\b", re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-zA-Z0-9_\-./:]+")


@dataclass(frozen=True)
class SupportConfig:
    require_message_recipient_match: bool = True
    suppress_shared_task_cross_agent: bool = True
    require_strong_channel_for_cross_agent: bool = False
    abstain_on_ambiguous: bool = False


def _extract_values(*text: str | None) -> set[str]:
    out: set[str] = set()
    for t in text:
        if not t:
            continue
        out.update(v.lower() for v in VALUE_RE.findall(t))
    return out


def _contains_any(values: set[str], *text: str | None) -> set[str]:
    haystack = " ".join(t or "" for t in text).lower()
    return {v for v in values if v in haystack}


def _message_recipient(source: EventRecord) -> str | None:
    if source.communication_metadata:
        for key in ("recipient", "to"):
            value = source.communication_metadata.get(key)
            if value:
                return str(value)
    if source.input and "to:" in source.input:
        return source.input.split("to:", 1)[1].strip().split()[0]
    return None


def _token_overlap_score(source: EventRecord, target: EventRecord) -> float:
    src_tokens = set(t.lower() for t in TOKEN_RE.findall(" ".join([source.result or "", source.agent_output or ""])))
    tgt_tokens = set(t.lower() for t in TOKEN_RE.findall(" ".join([target.input or "", target.result or "", target.agent_output or ""])))
    if not src_tokens or not tgt_tokens:
        return 0.0
    overlap = src_tokens & tgt_tokens
    if not overlap:
        return 0.0
    return min(1.0, len(overlap) / max(1, len(src_tokens)))


def estimate_support(source: EventRecord, target: EventRecord, candidate: CandidateRelation, config: SupportConfig | None = None) -> SupportEdge:
    config = config or SupportConfig()
    evidence: list[str] = []
    alternatives: list[str] = []
    score = 0.0
    support_type = "none"
    abstain = True

    source_values = _extract_values(source.result, source.agent_output, " ".join(source.artifact_refs))
    reused_values = _contains_any(source_values, target.input, target.result, target.agent_output, " ".join(target.artifact_refs))

    source_artifacts = {a.lower() for a in source.artifact_refs}
    target_text = " ".join([target.input or "", target.result or "", target.agent_output or "", " ".join(target.artifact_refs)]).lower()
    artifact_hits = {a for a in source_artifacts if a and a in target_text}

    shared_public_fetch = (
        source.agent_id != target.agent_id
        and source.tool is not None
        and target.tool is not None
        and source.tool == target.tool
        and source.input is not None
        and target.input is not None
        and source.input == target.input
    )

    if source.agent_id != target.agent_id and artifact_hits:
        score = max(score, 0.9)
        support_type = "artifact-mediated"
        abstain = False
        evidence.append(f"artifact overlap: {sorted(artifact_hits)[:3]}")

    if source.agent_id != target.agent_id and reused_values and not shared_public_fetch:
        score = max(score, 0.88)
        support_type = "information-content" if support_type == "none" else support_type
        abstain = False
        evidence.append(f"value reuse: {sorted(reused_values)[:4]}")

    if source.action_type == "message_send" and source.agent_id != target.agent_id:
        recipient = _message_recipient(source)
        if reused_values and (
            (not config.require_message_recipient_match) or recipient is None or recipient == target.agent_id
        ):
            score = max(score, 0.95)
            support_type = "direct"
            abstain = False
            evidence.append("message content appears in target action")
        elif recipient is not None and recipient != target.agent_id:
            alternatives.append("message recipient mismatch")
            score = min(score, 0.2)

    if source.agent_id == target.agent_id:
        same_agent_score = 0.45 + 0.2 * max(0.0, candidate.temporal_score)
        if same_agent_score > score:
            score = min(0.7, same_agent_score)
            support_type = "contextual"
            abstain = False
            evidence.append("same-agent continuity")

    overlap_score = _token_overlap_score(source, target)
    if (
        source.agent_id != target.agent_id
        and score < 0.6
        and overlap_score > 0.0
        and source.action_type != "message_send"
        and not artifact_hits
    ):
        score = max(score, 0.55)
        support_type = "contextual"
        abstain = False
        evidence.append("lexical overlap without explicit transfer channel")

    if source.shared_goal and target.shared_goal and source.shared_goal == target.shared_goal:
        alternatives.append("shared task/goal")
    if source.action_type == target.action_type and source.agent_id != target.agent_id:
        alternatives.append("task-parallel similar action")
    if candidate.semantic_similarity > 0.75 and not reused_values and not artifact_hits:
        alternatives.append("semantic similarity without direct transfer evidence")

    if (
        source.agent_id != target.agent_id
        and not reused_values
        and not artifact_hits
        and score < 0.65
        and support_type in {"none", "ambiguous"}
    ):
        if alternatives:
            support_type = "independent"
            score = min(score, 0.2)
            abstain = False
            evidence.append("no direct transfer signal found")
    if shared_public_fetch and source.agent_id != target.agent_id:
        support_type = "independent"
        score = min(score, 0.15)
        abstain = False
        evidence.append("same public tool/input available to both agents")
        alternatives.append("common external source")

    if (
        config.require_strong_channel_for_cross_agent
        and source.agent_id != target.agent_id
        and support_type not in {"direct", "artifact-mediated"}
        and score < 0.9
    ):
        support_type = "ambiguous"
        score = min(score, 0.25)
        abstain = True
        alternatives.append("cross-agent edge lacks strong channel evidence")

    if config.abstain_on_ambiguous and source.agent_id != target.agent_id and alternatives and support_type in {"contextual", "information-content", "ambiguous"} and score < 0.9:
        support_type = "ambiguous"
        abstain = True
        score = min(score, 0.2)

    if abstain:
        support_type = "ambiguous" if alternatives else "none"
        score = min(score, 0.15)

    return SupportEdge(
        source_event_id=source.event_id,
        target_event_id=target.event_id,
        support_score=round(float(score), 4),
        support_probability=None,
        support_type=support_type,
        evidence=tuple(evidence),
        alternative_explanations=tuple(dict.fromkeys(alternatives)),
        abstain=abstain,
    )


def score_candidates(
    events_by_id: dict[str, EventRecord], candidates: Iterable[CandidateRelation], min_score: float = 0.0, config: SupportConfig | None = None
) -> list[SupportEdge]:
    config = config or SupportConfig()
    edges: list[SupportEdge] = []
    for c in candidates:
        source = events_by_id[c.source_event_id]
        target = events_by_id[c.target_event_id]
        edge = estimate_support(source, target, c, config=config)
        if edge.support_score >= min_score:
            edges.append(edge)

    # Cross-edge suppression when same-agent explanation is stronger for the same target.
    by_target: dict[str, list[SupportEdge]] = {}
    for edge in edges:
        by_target.setdefault(edge.target_event_id, []).append(edge)

    adjusted: list[SupportEdge] = []
    for target_id, target_edges in by_target.items():
        target_event = events_by_id[target_id]
        same_agent_scores = [
            e.support_score
            for e in target_edges
            if events_by_id[e.source_event_id].agent_id == target_event.agent_id
        ]
        max_same = max(same_agent_scores) if same_agent_scores else 0.0
        for edge in target_edges:
            source_event = events_by_id[edge.source_event_id]
            is_cross = source_event.agent_id != target_event.agent_id
            if (
                config.suppress_shared_task_cross_agent
                and
                is_cross
                and source_event.action_type != "message_send"
                and edge.support_type in {"information-content", "contextual"}
                and source_event.shared_goal is not None
                and source_event.shared_goal == target_event.shared_goal
                and edge.support_score <= max_same + 0.35
            ):
                edge = replace(
                    edge,
                    support_score=min(edge.support_score, 0.3),
                    support_type="independent",
                    evidence=tuple(list(edge.evidence) + ["suppressed by stronger same-agent predecessor"]),
                    alternative_explanations=tuple(
                        list(edge.alternative_explanations) + ["same-agent explanation stronger"]
                    ),
                )
            adjusted.append(edge)

    return adjusted
