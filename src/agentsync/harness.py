"""The three-way benchmark harness — the core deliverable.

For each workload × strategy:

1. Create one strategy replica per agent in the workload.
2. Apply that agent's local writes to its own replica (the "concurrent" phase —
   replicas never see each other here).
3. Merge every replica into every other (full-mesh import/export), so each
   replica ends up holding the union. This is the sync phase.
4. Measure: convergence (all replicas identical?), writes lost vs. seen,
   attribution completeness, model calls, wall-clock, peak memory.
5. Score a PASS/FAIL verdict from the workload's expectation.

Strategies are pluggable via :data:`STRATEGIES`; the harness never branches on
strategy identity, so adding a fourth is a one-line registry change.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Callable

try:
    import psutil
    _HAS_PSUTIL = True
    _proc = psutil.Process()
except ImportError:  # pragma: no cover - bench extra is optional
    _HAS_PSUTIL = False
    _proc = None  # type: ignore[assignment]

from .models import EndState, FieldKind, MergeStrategy, RunResult, Workload
from .strategies import make_strategy


def _peak_mem_kb() -> float:
    if _proc is None:
        return float("nan")
    # rss is resident set size of this process; the harness is single-process,
    # so this captures the strategy's memory footprint directly.
    return _proc.memory_info().rss / 1024.0


def run_one(
    strategy_name: str,
    workload: Workload,
    strategy_factory: Callable[[str], MergeStrategy] = make_strategy,
) -> RunResult:
    """Run ``workload`` through one strategy and return a measured result row.

    Full-mesh merge: every replica imports every other replica's exported
    state. With N replicas this is N*(N-1) imports; for our 2-agent MVP that's
    a single swap, but the loop generalizes to larger agent counts without
    touching measurement logic.
    """
    t0 = time.perf_counter()
    mem_before = _peak_mem_kb()
    notes: list[str] = []

    replica_ids = list(workload.writes_by_replica.keys())
    replicas: dict[str, MergeStrategy] = {
        rid: strategy_factory(strategy_name) for rid in replica_ids
    }

    # Phase 1 — local concurrent edits (replicas are isolated).
    for rid in replica_ids:
        for write in workload.writes_by_replica[rid]:
            replicas[rid].apply(write)

    # Phase 2 — full-mesh sync: each replica pulls every other replica's state.
    for rid in replica_ids:
        for other in replica_ids:
            if other == rid:
                continue
            replicas[rid].import_state(replicas[other].export_state())

    states = {rid: r.finalized_state() for rid, r in replicas.items()}
    metrics = {rid: r.metrics() for rid, r in replicas.items()}

    latency_ms = (time.perf_counter() - t0) * 1000.0
    peak_mem_kb = max(_peak_mem_kb(), mem_before)

    converged = len({str(sorted(s.items())) for s in states.values()}) == 1

    # --- Correctness scoring against the workload's expectation ---
    sample_metrics = metrics[replica_ids[0]]
    # writes_lost is measured structurally, NOT from per-apply counters: an
    # apply always succeeds locally, so counting seen-applied would be 0 even
    # when a merge later clobbers the write. Instead we ask, for every
    # mergeable field, how many issued writes are reflected in the converged
    # state. The difference is real loss — e.g. LWW keeps one agent's `tags`
    # and silently overwrites the other's.
    converged_state = states[replica_ids[0]]
    writes_lost = _count_lost_writes(converged_state, workload)
    escalations = getattr(sample_metrics, "escalations", 0)
    # Attribution: every surviving write should trace to an agent. We check this
    # structurally — a correct merge keeps one attribution per *write*, LWW
    # keeps one per *field* (overwritten agents vanish).
    attribution_complete = _check_attribution(
        workload, replicas, sample_metrics
    )

    # Outcome — HOW the strategy reached its end state. Derived, not declared,
    # because corruption is a measurement, not a self-report. Two rules, in
    # priority order:
    #   1. corrupted: a mergeable write that should have survived didn't, OR a
    #      real semantic conflict existed and the strategy produced no signal
    #      for it (LWW silently picks a winner). Either way intent was lost
    #      with no escalation — the baseline failure mode.
    #   2. otherwise: the strategy's declared mode for conflicts it actually
    #      handled — resolved (spent a model call) or escalated (flagged).
    #      On a clean merge with nothing lost, every strategy is auto_merged.
    had_conflict = not workload.expectation.clean_merge
    silent_on_conflict = had_conflict and escalations == 0
    if writes_lost > 0 or silent_on_conflict:
        outcome = EndState.corrupted
    elif had_conflict:
        outcome = getattr(
            replicas[replica_ids[0]], "conflict_mode", EndState.escalated
        )
    else:
        outcome = EndState.auto_merged

    verdict, fail_notes = _score(
        workload, converged, writes_lost, escalations, attribution_complete
    )
    notes.extend(fail_notes)

    return RunResult(
        strategy=strategy_name,
        workload=workload.name,
        converged=converged,
        writes_lost=writes_lost,
        attribution_complete=attribution_complete,
        escalations=escalations,
        model_calls=sample_metrics.model_calls,
        latency_ms=latency_ms,
        peak_mem_kb=peak_mem_kb,
        outcome=outcome,
        verdict=verdict,
        notes=notes,
    )


def _count_lost_writes(converged_state: dict, workload: Workload) -> int:
    """Count writes the strategy dropped, judged structurally from the merged state.

    For mergeable fields (grow_set / append_text) the correct converged value
    contains EVERY agent's contribution; each one missing is one lost write.
    For scalars there's nothing to lose — only one value can hold, so a
    conflict there is an escalation concern, not a "lost write". This mirrors
    the thesis: mergeable state must not lose writes; semantic state must
    escalate.
    """
    # Bucket writes by field so we know each field's expected contributions.
    by_field: dict[str, list] = {}
    for writes in workload.writes_by_replica.values():
        for w in writes:
            by_field.setdefault(w.field, []).append(w)

    lost = 0
    for field, writes in by_field.items():
        kind = writes[0].kind
        actual = converged_state.get(field)
        if kind is FieldKind.grow_set:
            expected = set()
            for w in writes:
                expected |= set(w.value)
            actual_set = set(actual) if isinstance(actual, (list, tuple, set)) else set()
            # Each distinct contribution missing from the union is a lost write.
            lost += len(expected - actual_set)
        elif kind is FieldKind.append_text:
            expected_frags = [w.value for w in writes]
            if actual is None:
                lost += len(expected_frags)
            else:
                # Order is unspecified on concurrent append, so each fragment
                # must simply appear somewhere in the merged text.
                lost += sum(1 for frag in expected_frags if frag not in actual)
        # scalar: no writes_lost accounting; conflicts are scored as escalations.
    return lost


def _check_attribution(
    workload: Workload, replicas: dict[str, MergeStrategy], metrics
) -> bool:
    """Attribution is complete iff every distinct write that survives to the
    merged state still carries an agent_id.

    We approximate via the per-write attribution the strategy exposes (if any).
    LWW exposes per-FIELD attribution only, so a field written by two agents
    has one attribution for two writes → incomplete. CRDT exposes per-write.
    """
    # Count how many distinct (agent, op) writes the workload issued.
    distinct_writes = set()
    for writes in workload.writes_by_replica.values():
        for w in writes:
            distinct_writes.add((w.agent_id, w.op_id, w.field))

    # Strategy-specific attribution introspection.
    surviving = set()
    for r in replicas.values():
        attr_fn = getattr(r, "attribution", None)
        if attr_fn is None:
            continue
        for key, meta in attr_fn().items():
            surviving.add((meta["agent_id"], meta["op_id"], key))
    if not surviving:
        # Strategy exposes no per-write attribution at all — can't be complete.
        return len(distinct_writes) == 0
    return len(surviving) >= len(distinct_writes)


def _score(
    workload: Workload,
    converged: bool,
    writes_lost: int,
    escalations: int,
    attribution_complete: bool,
) -> tuple[str, list[str]]:
    """Turn raw metrics into a verdict using the workload's expectation."""
    notes: list[str] = []
    ok = True

    if not converged:
        ok = False
        notes.append("replicas diverged (no convergence)")

    exp = workload.expectation
    if exp.all_writes_survive and writes_lost > 0:
        ok = False
        notes.append(f"lost {writes_lost} write(s) on a mergeable workload")

    if exp.clean_merge and escalations > 0:
        ok = False
        notes.append(f"{escalations} spurious escalation(s) on a clean merge")

    if exp.semantic_conflict_on and escalations < len(exp.semantic_conflict_on):
        ok = False
        notes.append(
            f"expected escalation on {exp.semantic_conflict_on}, "
            f"got {escalations}"
        )

    if exp.all_writes_survive and not attribution_complete:
        ok = False
        notes.append("attribution incomplete (a surviving write lost its agent)")

    return ("PASS" if ok else "FAIL"), notes


__all__ = ["run_one"]
