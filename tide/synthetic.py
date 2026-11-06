from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import random
from typing import Any

from tide.schema import EventRecord, SupportEdge, build_event


@dataclass(frozen=True)
class GroundTruthRelation:
    source_event_id: str
    target_event_id: str
    scenario: str
    direct: bool
    mediated: bool
    via: tuple[str, ...]
    support_type: str
    misleading: bool
    shared_external_cause: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_event_id": self.source_event_id,
            "target_event_id": self.target_event_id,
            "scenario": self.scenario,
            "direct": self.direct,
            "mediated": self.mediated,
            "via": list(self.via),
            "support_type": self.support_type,
            "misleading": self.misleading,
            "shared_external_cause": self.shared_external_cause,
        }


@dataclass(frozen=True)
class SyntheticBundle:
    events: tuple[EventRecord, ...]
    ground_truth_edges: tuple[SupportEdge, ...]
    ground_truth_relations: tuple[GroundTruthRelation, ...]
    metadata: dict[str, Any]


def _iso(base: datetime, step: int) -> str:
    return (base + timedelta(seconds=step)).isoformat()


def _event(
    *,
    eid: str,
    episode: str,
    agent: str,
    ts: str,
    idx: int,
    action_type: str,
    action_summary: str,
    tool: str | None = None,
    input: str | None = None,
    result: str | None = None,
    output: str | None = None,
    artifacts: list[str] | None = None,
    agent_goal: str | None = None,
    shared_goal: str | None = None,
) -> EventRecord:
    return build_event(
        {
            "event_id": eid,
            "episode_id": episode,
            "agent_id": agent,
            "timestamp": ts,
            "event_index": idx,
            "agent_goal": agent_goal,
            "shared_goal": shared_goal,
            "action_type": action_type,
            "action_summary": action_summary,
            "tool": tool,
            "input": input,
            "result": result,
            "agent_output": output,
            "artifact_refs": artifacts or [],
        },
        source_table="synthetic",
        source_record_id=eid,
    )


def generate_synthetic_trace(seed: int = 7, run_id: int = 0) -> SyntheticBundle:
    rng = random.Random(seed + 10_000 * run_id)
    base = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc) + timedelta(hours=run_id)
    events: list[EventRecord] = []
    edges: list[SupportEdge] = []
    truth: list[GroundTruthRelation] = []

    def add_relation(source: str, target: str, scenario: str, support_type: str, *, direct: bool = True, mediated: bool = False, via: tuple[str, ...] = (), misleading: bool = False, shared_external: bool = False) -> None:
        edges.append(SupportEdge(source, target, 0.99, None, support_type, (f"{scenario}:{support_type}",), (), False))
        truth.append(
            GroundTruthRelation(
                source_event_id=source,
                target_event_id=target,
                scenario=scenario,
                direct=direct,
                mediated=mediated,
                via=via,
                support_type=support_type,
                misleading=misleading,
                shared_external_cause=shared_external,
            )
        )

    # 1) independent identical-task agents
    episode = "independent_identical_task"
    shared_goal = "Summarize status bulletin"
    token = f"risk-{rng.randint(100,999)}"
    e1 = _event(eid="e1", episode=episode, agent="A", ts=_iso(base, 1), idx=1, action_type="tool_call", action_summary="fetch bulletin", tool="http_get", input="https://news/internal", result=f"bulletin token {token}", output="fetched", shared_goal=shared_goal)
    e2 = _event(eid="e2", episode=episode, agent="B", ts=_iso(base, 2), idx=2, action_type="tool_call", action_summary="fetch bulletin", tool="http_get", input="https://news/internal", result=f"bulletin token {token}", output="fetched", shared_goal=shared_goal)
    e3 = _event(eid="e3", episode=episode, agent="A", ts=_iso(base, 3), idx=3, action_type="output", action_summary="write summary", input="bulletin text", output=f"summary includes {token}", shared_goal=shared_goal)
    e4 = _event(eid="e4", episode=episode, agent="B", ts=_iso(base, 4), idx=4, action_type="output", action_summary="write summary", input="bulletin text", output=f"summary includes {token}", shared_goal=shared_goal)
    events.extend([e1, e2, e3, e4])

    # 2) shared external event
    episode = "shared_external_event"
    ext = f"shock-{rng.randint(10,99)}"
    e5 = _event(eid="e5", episode=episode, agent="ENV", ts=_iso(base, 11), idx=1, action_type="external_event", action_summary="market shock", result=f"{ext}=true", output="external feed update", shared_goal="react")
    e6 = _event(eid="e6", episode=episode, agent="A", ts=_iso(base, 12), idx=2, action_type="tool_call", action_summary="raise alert", input=f"{ext}=true", result="alert sent", output="raised alert", shared_goal="react")
    e7 = _event(eid="e7", episode=episode, agent="B", ts=_iso(base, 13), idx=3, action_type="tool_call", action_summary="pause jobs", input=f"{ext}=true", result="jobs paused", output="paused", shared_goal="react")
    events.extend([e5, e6, e7])
    add_relation("e5", "e6", "shared_external_event", "contextual", direct=True, shared_external=True)
    add_relation("e5", "e7", "shared_external_event", "contextual", direct=True, shared_external=True)

    # 3) explicit communication
    episode = "explicit_communication"
    ck = f"cks-{rng.randint(10000,99999)}"
    e8 = _event(eid="e8", episode=episode, agent="A", ts=_iso(base, 21), idx=1, action_type="tool_call", action_summary="compute checksum", tool="checksum", input="bundle-x", result=ck, output=f"checksum {ck}", shared_goal="ship")
    e9 = _event(eid="e9", episode=episode, agent="A", ts=_iso(base, 22), idx=2, action_type="message_send", action_summary="send checksum", input="to:B", result=f"checksum {ck}", output=f"sent {ck}", shared_goal="ship")
    e10 = _event(eid="e10", episode=episode, agent="B", ts=_iso(base, 23), idx=3, action_type="tool_call", action_summary="verify checksum", tool="verify", input=f"checksum={ck}", result="pass", output="verified", shared_goal="ship")
    events.extend([e8, e9, e10])
    add_relation("e8", "e9", "explicit_communication", "information-content")
    add_relation("e9", "e10", "explicit_communication", "direct")

    # 4) shared-resource mediation
    episode = "shared_resource"
    rid = f"RID-{rng.randint(1000,9999)}"
    e11 = _event(eid="e11", episode=episode, agent="A", ts=_iso(base, 31), idx=1, action_type="tool_call", action_summary="write shared id", tool="file_write", input="/shared/id.txt", result=rid, output="written", artifacts=["/shared/id.txt", rid], shared_goal="assemble packet")
    e12 = _event(eid="e12", episode=episode, agent="B", ts=_iso(base, 32), idx=2, action_type="tool_call", action_summary="read shared id", tool="file_read", input="/shared/id.txt", result=rid, output=f"read {rid}", artifacts=["/shared/id.txt", rid], shared_goal="assemble packet")
    e13 = _event(eid="e13", episode=episode, agent="B", ts=_iso(base, 33), idx=3, action_type="tool_call", action_summary="query packet", tool="db_query", input=f"rid={rid}", result="packet", output="queried", shared_goal="assemble packet")
    events.extend([e11, e12, e13])
    add_relation("e11", "e12", "shared_resource", "artifact-mediated")
    add_relation("e12", "e13", "shared_resource", "information-content")

    # 5) broker/aggregator
    episode = "broker_aggregator"
    lat = f"lat={rng.randint(100,180)}ms"
    err = f"err={rng.randint(7,14)}%"
    e14 = _event(eid="e14", episode=episode, agent="B", ts=_iso(base, 41), idx=1, action_type="tool_call", action_summary="collect latency", tool="metric", result=lat, output=lat, shared_goal="recommend action")
    e15 = _event(eid="e15", episode=episode, agent="C", ts=_iso(base, 42), idx=2, action_type="tool_call", action_summary="collect errors", tool="metric", result=err, output=err, shared_goal="recommend action")
    e16 = _event(eid="e16", episode=episode, agent="A", ts=_iso(base, 43), idx=3, action_type="output", action_summary="aggregate and recommend rollback", input=f"{lat};{err}", output=f"rollback because {lat} and {err}", shared_goal="recommend action")
    e17 = _event(eid="e17", episode=episode, agent="D", ts=_iso(base, 44), idx=4, action_type="tool_call", action_summary="execute rollback", tool="deploy", input="rollback", result="done", output="rolled back", shared_goal="recommend action")
    events.extend([e14, e15, e16, e17])
    add_relation("e14", "e16", "broker_aggregator", "information-content")
    add_relation("e15", "e16", "broker_aggregator", "information-content")
    add_relation("e16", "e17", "broker_aggregator", "direct")

    # 6) misleading information used
    episode = "misleading_information"
    bad_port = f"port={rng.choice([9001, 9010, 7777])}"
    e18 = _event(eid="e18", episode=episode, agent="A", ts=_iso(base, 51), idx=1, action_type="message_send", action_summary="send wrong config", input="to:B", result=bad_port, output=f"send {bad_port}", shared_goal="start service")
    e19 = _event(eid="e19", episode=episode, agent="B", ts=_iso(base, 52), idx=2, action_type="tool_call", action_summary="start with received config", tool="service_start", input=bad_port, result="failed start", output="startup failed", shared_goal="start service")
    events.extend([e18, e19])
    add_relation("e18", "e19", "misleading_information", "direct", misleading=True)

    # 7) delayed information use
    episode = "delayed_use"
    delayed_key = f"KEY-{rng.randint(100,999)}"
    e20 = _event(eid="e20", episode=episode, agent="A", ts=_iso(base, 61), idx=1, action_type="message_send", action_summary="share api key", input="to:B", result=delayed_key, output=f"sent {delayed_key}", shared_goal="finish task")
    e21 = _event(eid="e21", episode=episode, agent="B", ts=_iso(base, 62), idx=2, action_type="tool_call", action_summary="do unrelated step one", tool="lint", input="repo", result="ok", output="step one", shared_goal="finish task")
    e22 = _event(eid="e22", episode=episode, agent="B", ts=_iso(base, 75), idx=3, action_type="tool_call", action_summary="do unrelated step two", tool="test", input="subset", result="ok", output="step two", shared_goal="finish task")
    e23 = _event(eid="e23", episode=episode, agent="B", ts=_iso(base, 90), idx=4, action_type="tool_call", action_summary="call protected api", tool="api_call", input=f"key={delayed_key}", result="200", output="success", shared_goal="finish task")
    events.extend([e20, e21, e22, e23])
    add_relation("e20", "e23", "delayed_use", "direct")

    # 8) indirect dependency chain A -> B -> C (no direct A->C)
    episode = "indirect_chain"
    val = f"TOKEN-{rng.randint(1000,9999)}"
    e24 = _event(eid="e24", episode=episode, agent="A", ts=_iso(base, 101), idx=1, action_type="message_send", action_summary="send token to B", input="to:B", result=val, output=f"sent {val}", shared_goal="chain")
    e25 = _event(eid="e25", episode=episode, agent="B", ts=_iso(base, 102), idx=2, action_type="output", action_summary="transform token", input=val, output=f"{val}-X", result=f"{val}-X", shared_goal="chain")
    e26 = _event(eid="e26", episode=episode, agent="B", ts=_iso(base, 103), idx=3, action_type="message_send", action_summary="send transformed token", input="to:C", result=f"{val}-X", output="sent transformed token", shared_goal="chain")
    e27 = _event(eid="e27", episode=episode, agent="C", ts=_iso(base, 104), idx=4, action_type="tool_call", action_summary="use transformed token", tool="api_call", input=f"token={val}-X", result="ok", output="done", shared_goal="chain")
    events.extend([e24, e25, e26, e27])
    add_relation("e24", "e25", "indirect_chain", "direct")
    add_relation("e25", "e26", "indirect_chain", "information-content")
    add_relation("e26", "e27", "indirect_chain", "direct")
    truth.append(
        GroundTruthRelation(
            source_event_id="e24",
            target_event_id="e27",
            scenario="indirect_chain",
            direct=False,
            mediated=True,
            via=("e25", "e26"),
            support_type="indirect_path_only",
            misleading=False,
            shared_external_cause=False,
        )
    )

    # 9) unrelated coincidental similarity
    episode = "coincidental_similarity"
    e28 = _event(eid="e28", episode=episode, agent="U", ts=_iso(base, 111), idx=1, action_type="output", action_summary="produce phrase alpha beta", output="alpha beta gamma", shared_goal="task-u")
    e29 = _event(eid="e29", episode=episode, agent="V", ts=_iso(base, 112), idx=2, action_type="output", action_summary="produce phrase alpha beta", output="alpha beta gamma", shared_goal="task-v")
    events.extend([e28, e29])

    # 10) mixed conditions
    episode = "mixed_conditions"
    mx = f"MIX-{rng.randint(100,999)}"
    e30 = _event(eid="e30", episode=episode, agent="ENV", ts=_iso(base, 121), idx=1, action_type="external_event", action_summary="new regulation", result=f"reg={mx}", output="regulation update")
    e31 = _event(eid="e31", episode=episode, agent="A", ts=_iso(base, 122), idx=2, action_type="message_send", action_summary="send parsed regulation", input="to:B", result=f"rule:{mx}", output="sent parsed rule", shared_goal="comply")
    e32 = _event(eid="e32", episode=episode, agent="B", ts=_iso(base, 123), idx=3, action_type="tool_call", action_summary="apply parsed regulation", tool="policy_apply", input=f"rule:{mx}", result="policy updated", output="updated", shared_goal="comply")
    e33 = _event(eid="e33", episode=episode, agent="C", ts=_iso(base, 124), idx=4, action_type="tool_call", action_summary="independently react to regulation", tool="policy_pause", input=f"reg={mx}", result="paused batch", output="paused", shared_goal="comply")
    events.extend([e30, e31, e32, e33])
    add_relation("e30", "e31", "mixed_conditions", "contextual", shared_external=True)
    add_relation("e31", "e32", "mixed_conditions", "direct")
    add_relation("e30", "e33", "mixed_conditions", "contextual", shared_external=True)

    # 11) hard negative: identical tasks, offset timing.
    episode = "hardneg_timing_offset"
    fact = f"FACT-{rng.randint(100,999)}"
    e34 = _event(eid="e34", episode=episode, agent="A", ts=_iso(base, 141), idx=1, action_type="tool_call", action_summary="fetch reference", tool="http_get", input="https://kb/item", result=fact, output=fact, shared_goal="compile note")
    e35 = _event(eid="e35", episode=episode, agent="A", ts=_iso(base, 142), idx=2, action_type="output", action_summary="draft note", input=fact, output=f"note: {fact}", shared_goal="compile note")
    e36 = _event(eid="e36", episode=episode, agent="B", ts=_iso(base, 158), idx=3, action_type="tool_call", action_summary="fetch reference", tool="http_get", input="https://kb/item", result=fact, output=fact, shared_goal="compile note")
    e37 = _event(eid="e37", episode=episode, agent="B", ts=_iso(base, 160), idx=4, action_type="output", action_summary="draft note", input=fact, output=f"note: {fact}", shared_goal="compile note")
    events.extend([e34, e35, e36, e37])

    # 12) hard negative: paraphrased independent outputs.
    episode = "hardneg_paraphrased_independent"
    key = f"KEY-{rng.randint(100,999)}"
    e38 = _event(eid="e38", episode=episode, agent="A", ts=_iso(base, 171), idx=1, action_type="output", action_summary="status summary", output=f"critical key {key} is available", shared_goal="status report")
    e39 = _event(eid="e39", episode=episode, agent="B", ts=_iso(base, 172), idx=2, action_type="output", action_summary="status summary", output=f"availability confirmed for {key}", shared_goal="status report")
    events.extend([e38, e39])

    # 13) hard negative: equivalent action sequences, different tools.
    episode = "hardneg_equivalent_sequences"
    h = f"H-{rng.randint(1000,9999)}"
    e40 = _event(eid="e40", episode=episode, agent="A", ts=_iso(base, 181), idx=1, action_type="tool_call", action_summary="hash payload", tool="sha256", input="blob-a", result=h, output=h, shared_goal="verify payload")
    e41 = _event(eid="e41", episode=episode, agent="B", ts=_iso(base, 182), idx=2, action_type="tool_call", action_summary="hash payload", tool="blake2", input="blob-a", result=h, output=h, shared_goal="verify payload")
    e42 = _event(eid="e42", episode=episode, agent="A", ts=_iso(base, 183), idx=3, action_type="output", action_summary="verify hash", input=h, output="verified", shared_goal="verify payload")
    e43 = _event(eid="e43", episode=episode, agent="B", ts=_iso(base, 184), idx=4, action_type="output", action_summary="verify hash", input=h, output="verified", shared_goal="verify payload")
    events.extend([e40, e41, e42, e43])

    # 14) hard negative: verbosity differences.
    episode = "hardneg_verbosity"
    z = f"Z-{rng.randint(20,90)}"
    e44 = _event(eid="e44", episode=episode, agent="A", ts=_iso(base, 191), idx=1, action_type="output", action_summary="short report", output=f"{z} ok", shared_goal="report")
    e45 = _event(eid="e45", episode=episode, agent="B", ts=_iso(base, 192), idx=2, action_type="output", action_summary="long report", output=f"Detailed analysis: metric {z} remains within accepted thresholds and no intervention is required.", shared_goal="report")
    events.extend([e44, e45])

    # 15) hard negative: independent same-fact discovery.
    episode = "hardneg_independent_same_fact"
    same_fact = f"server={rng.choice(['s1','s2'])} status=degraded"
    e46 = _event(eid="e46", episode=episode, agent="A", ts=_iso(base, 201), idx=1, action_type="tool_call", action_summary="poll server", tool="monitor_read", input="cluster-a", result=same_fact, output=same_fact, shared_goal="triage")
    e47 = _event(eid="e47", episode=episode, agent="B", ts=_iso(base, 203), idx=2, action_type="tool_call", action_summary="poll server", tool="monitor_read", input="cluster-a", result=same_fact, output=same_fact, shared_goal="triage")
    events.extend([e46, e47])

    # 16) hard negative: long-running with repeated common observations.
    episode = "hardneg_repeated_common_obs"
    e48 = _event(eid="e48", episode=episode, agent="ENV", ts=_iso(base, 211), idx=1, action_type="external_event", action_summary="shared feed tick", result="temp=71", output="tick")
    e49 = _event(eid="e49", episode=episode, agent="A", ts=_iso(base, 212), idx=2, action_type="output", action_summary="log temperature", input="temp=71", output="A logged temp 71", shared_goal="monitor")
    e50 = _event(eid="e50", episode=episode, agent="B", ts=_iso(base, 213), idx=3, action_type="output", action_summary="log temperature", input="temp=71", output="B logged temp 71", shared_goal="monitor")
    e51 = _event(eid="e51", episode=episode, agent="ENV", ts=_iso(base, 214), idx=4, action_type="external_event", action_summary="shared feed tick", result="temp=72", output="tick")
    e52 = _event(eid="e52", episode=episode, agent="A", ts=_iso(base, 215), idx=5, action_type="output", action_summary="log temperature", input="temp=72", output="A logged temp 72", shared_goal="monitor")
    e53 = _event(eid="e53", episode=episode, agent="B", ts=_iso(base, 216), idx=6, action_type="output", action_summary="log temperature", input="temp=72", output="B logged temp 72", shared_goal="monitor")
    events.extend([e48, e49, e50, e51, e52, e53])
    add_relation("e48", "e49", "hardneg_repeated_common_obs", "contextual", shared_external=True)
    add_relation("e48", "e50", "hardneg_repeated_common_obs", "contextual", shared_external=True)
    add_relation("e51", "e52", "hardneg_repeated_common_obs", "contextual", shared_external=True)
    add_relation("e51", "e53", "hardneg_repeated_common_obs", "contextual", shared_external=True)

    # 17) hard negative: similar outputs at different times.
    episode = "hardneg_similar_diff_times"
    q = f"Q-{rng.randint(100,999)}"
    e54 = _event(eid="e54", episode=episode, agent="A", ts=_iso(base, 221), idx=1, action_type="output", action_summary="checkpoint update", output=f"checkpoint {q} complete", shared_goal="sync")
    e55 = _event(eid="e55", episode=episode, agent="B", ts=_iso(base, 251), idx=2, action_type="output", action_summary="checkpoint update", output=f"checkpoint {q} complete", shared_goal="sync")
    events.extend([e54, e55])

    # 18) hard negative: shared instruction but different observations.
    episode = "hardneg_shared_instruction_diff_observation"
    e56 = _event(eid="e56", episode=episode, agent="A", ts=_iso(base, 261), idx=1, action_type="tool_call", action_summary="check region east", tool="sensor_read", input="east", result="east=stable", output="east stable", shared_goal="regional check")
    e57 = _event(eid="e57", episode=episode, agent="B", ts=_iso(base, 262), idx=2, action_type="tool_call", action_summary="check region west", tool="sensor_read", input="west", result="west=unstable", output="west unstable", shared_goal="regional check")
    events.extend([e56, e57])

    # 19) hard positive: paraphrased transfer.
    episode = "hardpos_paraphrased_transfer"
    ref = f"REF-{rng.randint(100,999)}"
    e58 = _event(eid="e58", episode=episode, agent="B", ts=_iso(base, 271), idx=1, action_type="message_send", action_summary="share ref", input="to:A", result=f"use code {ref}", output=f"sent {ref}", shared_goal="deploy")
    e59 = _event(eid="e59", episode=episode, agent="A", ts=_iso(base, 272), idx=2, action_type="output", action_summary="paraphrase instruction", input=f"identifier {ref}", output=f"apply the reference token {ref} before deployment", shared_goal="deploy")
    events.extend([e58, e59])
    add_relation("e58", "e59", "hardpos_paraphrased_transfer", "direct")

    # 20) hard positive: summarize multi-facts from B and C.
    episode = "hardpos_summary_multi_source"
    f1 = f"lat={rng.randint(80,130)}ms"
    f2 = f"err={rng.randint(2,8)}%"
    e60 = _event(eid="e60", episode=episode, agent="B", ts=_iso(base, 281), idx=1, action_type="message_send", action_summary="send latency", input="to:A", result=f1, output=f1, shared_goal="decision")
    e61 = _event(eid="e61", episode=episode, agent="C", ts=_iso(base, 282), idx=2, action_type="message_send", action_summary="send errors", input="to:A", result=f2, output=f2, shared_goal="decision")
    e62 = _event(eid="e62", episode=episode, agent="A", ts=_iso(base, 283), idx=3, action_type="output", action_summary="summarize and decide", input=f"{f1}; {f2}", output="recommend canary", shared_goal="decision")
    events.extend([e60, e61, e62])
    add_relation("e60", "e62", "hardpos_summary_multi_source", "direct")
    add_relation("e61", "e62", "hardpos_summary_multi_source", "direct")

    # 21) hard positive: transformed identifier representation.
    episode = "hardpos_identifier_transform"
    raw_id = f"ID{rng.randint(1000,9999)}"
    encoded = f"{raw_id}-b64"
    e63 = _event(eid="e63", episode=episode, agent="A", ts=_iso(base, 291), idx=1, action_type="message_send", action_summary="send raw id", input="to:B", result=raw_id, output=raw_id, shared_goal="lookup")
    e64 = _event(eid="e64", episode=episode, agent="B", ts=_iso(base, 292), idx=2, action_type="output", action_summary="encode identifier", input=raw_id, result=encoded, output=encoded, shared_goal="lookup")
    e65 = _event(eid="e65", episode=episode, agent="B", ts=_iso(base, 293), idx=3, action_type="tool_call", action_summary="query encoded id", tool="db_query", input=f"encoded={encoded}", result="row found", output="ok", shared_goal="lookup")
    events.extend([e63, e64, e65])
    add_relation("e63", "e64", "hardpos_identifier_transform", "direct")
    add_relation("e64", "e65", "hardpos_identifier_transform", "information-content")

    # 22) hard positive: long delay with unrelated events.
    episode = "hardpos_long_delay"
    k2 = f"AUTH-{rng.randint(100,999)}"
    e66 = _event(eid="e66", episode=episode, agent="A", ts=_iso(base, 301), idx=1, action_type="message_send", action_summary="send auth token", input="to:B", result=k2, output=k2, shared_goal="integration")
    e67 = _event(eid="e67", episode=episode, agent="B", ts=_iso(base, 302), idx=2, action_type="tool_call", action_summary="compile module", tool="build", input="m1", result="ok", output="ok", shared_goal="integration")
    e68 = _event(eid="e68", episode=episode, agent="B", ts=_iso(base, 320), idx=3, action_type="tool_call", action_summary="run tests", tool="test", input="suite", result="ok", output="ok", shared_goal="integration")
    e69 = _event(eid="e69", episode=episode, agent="B", ts=_iso(base, 340), idx=4, action_type="tool_call", action_summary="authenticate and push", tool="api_call", input=f"token={k2}", result="accepted", output="done", shared_goal="integration")
    events.extend([e66, e67, e68, e69])
    add_relation("e66", "e69", "hardpos_long_delay", "direct")

    # 23) hard positive: indirect shared artifact communication.
    episode = "hardpos_indirect_artifact"
    aid = f"AID-{rng.randint(100,999)}"
    e70 = _event(eid="e70", episode=episode, agent="A", ts=_iso(base, 351), idx=1, action_type="tool_call", action_summary="write artifact", tool="file_write", input="/shared/aid.txt", result=aid, output=aid, artifacts=["/shared/aid.txt", aid], shared_goal="handoff")
    e71 = _event(eid="e71", episode=episode, agent="C", ts=_iso(base, 352), idx=2, action_type="tool_call", action_summary="read artifact", tool="file_read", input="/shared/aid.txt", result=aid, output=aid, artifacts=["/shared/aid.txt", aid], shared_goal="handoff")
    e72 = _event(eid="e72", episode=episode, agent="C", ts=_iso(base, 353), idx=3, action_type="message_send", action_summary="relay artifact id", input="to:B", result=aid, output=f"relay {aid}", shared_goal="handoff")
    e73 = _event(eid="e73", episode=episode, agent="B", ts=_iso(base, 354), idx=4, action_type="tool_call", action_summary="consume relayed id", tool="api_call", input=f"id={aid}", result="ok", output="done", shared_goal="handoff")
    events.extend([e70, e71, e72, e73])
    add_relation("e70", "e71", "hardpos_indirect_artifact", "artifact-mediated")
    add_relation("e71", "e72", "hardpos_indirect_artifact", "information-content")
    add_relation("e72", "e73", "hardpos_indirect_artifact", "direct")

    # 24) hard positive: misleading info relay chain.
    episode = "hardpos_misleading_relay"
    wrong = f"cfg={rng.randint(4000,4999)}"
    e74 = _event(eid="e74", episode=episode, agent="A", ts=_iso(base, 361), idx=1, action_type="message_send", action_summary="send wrong config to B", input="to:B", result=wrong, output=wrong, shared_goal="deploy",)
    e75 = _event(eid="e75", episode=episode, agent="B", ts=_iso(base, 362), idx=2, action_type="message_send", action_summary="relay config to C", input="to:C", result=wrong, output=f"relayed {wrong}", shared_goal="deploy")
    e76 = _event(eid="e76", episode=episode, agent="C", ts=_iso(base, 363), idx=3, action_type="tool_call", action_summary="use relayed config", tool="service_start", input=wrong, result="startup failed", output="failed", shared_goal="deploy")
    events.extend([e74, e75, e76])
    add_relation("e74", "e75", "hardpos_misleading_relay", "direct", misleading=True)
    add_relation("e75", "e76", "hardpos_misleading_relay", "direct", misleading=True)

    metadata = {
        "seed": seed,
        "run_id": run_id,
        "scenario_count": len({e.episode_id for e in events}),
        "event_count": len(events),
        "ground_truth_direct_edges": len([r for r in truth if r.direct and r.support_type != "indirect_path_only"]),
        "notes": "Synthetic data only. Ground truth used for evaluation only.",
    }
    return SyntheticBundle(tuple(events), tuple(edges), tuple(truth), metadata)
