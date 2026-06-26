"""Workloads — scenarios run through every strategy.

A workload is the partial order of writes each agent performs locally; the
harness then merges replicas and measures correctness. We ship two from day
one, matching the Definition of Done:

* :func:`clean_merge` — mergeable concurrent writes. Correct strategy: zero
  writes lost, full attribution, zero escalations, zero model calls. LWW still
  corrupts it (overwrites the shared grow-set / appended text).
* :func:`semantic_conflict` — two agents set the same scalar to different
  values concurrently. This is un-mergeable; a correct strategy ESCALATES
  instead of silently picking a winner. LWW silently picks one (FAIL); CRDT
  escalates (PASS); transactional escalates via a model call (PASS but costly).

Workloads are plain data so new ones are trivial to add.
"""

from __future__ import annotations

from ..models import Expectation, FieldKind, Workload, Write


def clean_merge() -> Workload:
    """Two agents co-author shared context. All writes are mergeable.

    Layout of the shared doc:
      * ``title``   — scalar, written by ONE agent (no conflict possible)
      * ``tags``    — grow_set, BOTH agents add distinct tags concurrently
      * ``notes``   — append_text, BOTH agents append distinct text concurrently

    A correct merge keeps BOTH agents' tags and BOTH appends — 4 surviving
    contributions across 2 fields. LWW stores each field as an opaque value, so
    when the two replicas merge one agent's ``tags`` and one agent's ``notes``
    overwrite the other's: **writes are lost on a workload that should be
    conflict-free.** That is the headline corruption the CRDT eliminates.
    """
    return Workload(
        name="clean_merge",
        description=(
            "Two agents concurrently add distinct tags and append distinct notes "
            "to shared context. All writes are mergeable; nothing should be lost."
        ),
        writes_by_replica={
            "agent-A": [
                Write(agent_id="A", op_id=0, field="title", value="Project Sync", kind=FieldKind.scalar),
                Write(agent_id="A", op_id=1, field="tags", value={"crdt", "agents"}, kind=FieldKind.grow_set),
                Write(agent_id="A", op_id=2, field="notes", value="A: seeded the doc.", kind=FieldKind.append_text),
            ],
            "agent-B": [
                # B's writes are concurrent with A's — neither observed the other.
                Write(agent_id="B", op_id=0, field="tags", value={"benchmark", "loro"}, kind=FieldKind.grow_set),
                Write(agent_id="B", op_id=1, field="notes", value="B: added a note.", kind=FieldKind.append_text),
            ],
        },
        expectation=Expectation(
            clean_merge=True,
            all_writes_survive=True,
            semantic_conflict_on=(),
        ),
    )


def semantic_conflict() -> Workload:
    """Two agents set the SAME scalar to DIFFERENT values concurrently.

    ``status`` is a scalar (not a grow set): two agents racing to set it to
    ``"draft"`` vs ``"published"`` is a genuine semantic conflict — there is no
    merge that preserves both intents. The correct behavior is to ESCALATE.

    LWW silently keeps one and drops the other (FAIL: corrupts intent +
    no escalation). CRDT detects the divergence and escalates with both
    contenders attributed (PASS). Transactional escalates too, but only after
    spending a model call to notice the conflict.
    """
    return Workload(
        name="semantic_conflict",
        description=(
            "Two agents concurrently set the same scalar `status` to different "
            "values. Un-mergeable — a correct strategy escalates rather than "
            "silently picking a winner."
        ),
        writes_by_replica={
            "agent-A": [
                Write(agent_id="A", op_id=0, field="status", value="draft", kind=FieldKind.scalar),
            ],
            "agent-B": [
                Write(agent_id="B", op_id=0, field="status", value="published", kind=FieldKind.scalar),
            ],
        },
        expectation=Expectation(
            clean_merge=False,
            all_writes_survive=False,  # a scalar can only hold one value
            semantic_conflict_on=("status",),
        ),
    )


def all_workloads() -> list[Workload]:
    return [clean_merge(), semantic_conflict()]


__all__ = ["clean_merge", "semantic_conflict", "all_workloads"]
