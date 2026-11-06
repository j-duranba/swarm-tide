from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import hashlib
import math
import re
from typing import Iterable

from tide.schema import CandidateRelation, EventRecord

TOKEN_RE = re.compile(r"[a-zA-Z0-9_\-./:]+")


@dataclass(frozen=True)
class RetrievalConfig:
    top_k_cross_agent: int = 8
    same_agent_history_k: int = 3
    max_time_lag_seconds: int = 3600
    embedding_dim: int = 256
    alpha: float = 0.45
    beta: float = 0.2
    gamma: float = 0.25
    delta: float = 0.1
    enforce_temporal_order: bool = True


def _tokens(*values: str | None) -> list[str]:
    out: list[str] = []
    for value in values:
        if not value:
            continue
        out.extend(t.lower() for t in TOKEN_RE.findall(value))
    return out


def _event_text(event: EventRecord) -> str:
    return " ".join(
        filter(
            None,
            [
                event.action_type,
                event.action_summary,
                event.tool,
                event.input,
                event.result,
                event.agent_output,
                " ".join(event.artifact_refs),
            ],
        )
    )


def _hashed_embedding(text: str, dim: int) -> list[float]:
    vec = [0.0] * dim
    for token in _tokens(text):
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % dim
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def _parse_ts(ts: str | None) -> float | None:
    if not ts:
        return None
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


def _temporal_score(source: EventRecord, target: EventRecord, max_lag: int) -> float:
    s, t = _parse_ts(source.timestamp_utc), _parse_ts(target.timestamp_utc)
    if s is None or t is None:
        return 0.0
    lag = t - s
    if lag <= 0:
        return -1.0
    if lag >= max_lag:
        return 0.0
    return max(0.0, 1.0 - lag / max_lag)


def strictly_precedes(source: EventRecord, target: EventRecord) -> bool:
    s_ts = _parse_ts(source.timestamp_utc)
    t_ts = _parse_ts(target.timestamp_utc)
    if s_ts is not None and t_ts is not None:
        if s_ts < t_ts:
            return True
        if s_ts > t_ts:
            return False
        if source.event_index is not None and target.event_index is not None:
            return source.event_index < target.event_index
        return False
    if source.event_index is not None and target.event_index is not None:
        return source.event_index < target.event_index
    return False


def _artifact_overlap(source: EventRecord, target: EventRecord) -> float:
    source_items = set(a.lower() for a in source.artifact_refs)
    target_text = " ".join(
        filter(None, [target.input or "", target.result or "", target.agent_output or "", " ".join(target.artifact_refs)])
    ).lower()
    if not source_items or not target_text:
        return 0.0
    matches = sum(1 for item in source_items if item and item in target_text)
    return min(1.0, matches / max(1, len(source_items)))


def _compatibility(source: EventRecord, target: EventRecord) -> float:
    if source.action_type == "message_send" and target.agent_id != source.agent_id:
        return 0.8
    if source.tool and target.input and source.tool.lower() in target.input.lower():
        return 0.7
    if source.agent_id == target.agent_id:
        return 0.4
    if source.shared_goal and target.shared_goal and source.shared_goal == target.shared_goal:
        return 0.3
    return 0.1


def retrieve_candidates(events: Iterable[EventRecord], config: RetrievalConfig | None = None) -> dict[str, list[CandidateRelation]]:
    config = config or RetrievalConfig()
    per_episode: dict[str, list[EventRecord]] = defaultdict(list)
    for event in events:
        per_episode[event.episode_id].append(event)

    results: dict[str, list[CandidateRelation]] = {}
    for _, ep_events in per_episode.items():
        ordered = sorted(ep_events, key=lambda e: (e.timestamp_utc or "9999-12-31T23:59:59+00:00", e.event_index or 10**12, e.event_id))
        embeddings = {e.event_id: _hashed_embedding(_event_text(e), config.embedding_dim) for e in ordered}

        for idx, target in enumerate(ordered):
            past = ordered[:idx]
            if not past:
                results[target.event_id] = []
                continue
            if config.enforce_temporal_order:
                past = [p for p in past if strictly_precedes(p, target)]
                if not past:
                    results[target.event_id] = []
                    continue

            # Forced same-agent history candidates.
            same_agent = [e for e in reversed(past) if e.agent_id == target.agent_id][: config.same_agent_history_k]
            same_agent_ids = {e.event_id for e in same_agent}

            # Cross-agent candidates scored by retrieval components.
            cross_scores: list[tuple[float, EventRecord, float, float, float, float]] = []
            for source in past:
                if source.event_id in same_agent_ids:
                    continue
                if source.agent_id == target.agent_id:
                    continue
                sem = _cosine(embeddings[source.event_id], embeddings[target.event_id])
                temp = _temporal_score(source, target, config.max_time_lag_seconds)
                if temp < 0:
                    continue
                art = _artifact_overlap(source, target)
                comp = _compatibility(source, target)
                score = config.alpha * sem + config.beta * temp + config.gamma * art + config.delta * comp
                cross_scores.append((score, source, sem, temp, art, comp))

            cross_scores.sort(key=lambda x: x[0], reverse=True)
            selected_cross = cross_scores[: config.top_k_cross_agent]

            picked: list[CandidateRelation] = []
            for source in same_agent:
                sem = _cosine(embeddings[source.event_id], embeddings[target.event_id])
                temp = _temporal_score(source, target, config.max_time_lag_seconds)
                art = _artifact_overlap(source, target)
                comp = _compatibility(source, target)
                score = config.alpha * sem + config.beta * max(0.0, temp) + config.gamma * art + config.delta * comp
                picked.append(
                    CandidateRelation(
                        source_event_id=source.event_id,
                        target_event_id=target.event_id,
                        retrieval_score=score,
                        semantic_similarity=sem,
                        temporal_score=max(0.0, temp),
                        artifact_overlap=art,
                        compatibility_score=comp,
                    )
                )

            for score, source, sem, temp, art, comp in selected_cross:
                picked.append(
                    CandidateRelation(
                        source_event_id=source.event_id,
                        target_event_id=target.event_id,
                        retrieval_score=score,
                        semantic_similarity=sem,
                        temporal_score=max(0.0, temp),
                        artifact_overlap=art,
                        compatibility_score=comp,
                    )
                )

            picked.sort(key=lambda c: c.retrieval_score, reverse=True)
            results[target.event_id] = picked

    return results
