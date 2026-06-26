"""The three-strategy contract.

Everything in the benchmark flows through these types so that LWW,
LLM-transactional, and CRDT (Loro) are measured behind ONE common interface.
The types here are deliberately small and backend-agnostic: a strategy is a
class with an `apply(write) -> Outcome` method and a `finalize_state()` call.

Two value kinds matter for the thesis:

* ``FieldKind.scalar``  — a single value; two concurrent writes to the same
  scalar with differing values are a SEMANTIC conflict (un-mergeable). LWW
  silently picks one; CRDT must ESCALATE.
* ``FieldKind.grow_set`` / ``FieldKind.append_text`` — monotone, mergeable.
  Concurrent writes union / concatenate; no write is lost. This is where the
  CRDT wins for free and LWW still corrupts (it overwrites the whole field).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Protocol


class EndState(str, Enum):
    """How a strategy reached its end state on a workload.

    This is intentionally SEPARATE from ``verdict`` (which only says "reached an
    acceptable end state"). Two strategies can both PASS yet get there by
    different, non-interchangeable routes — and conflating them is the trap this
    column exists to prevent.

    ``escalated`` vs ``resolved`` is the one that matters and is the easiest to
    misread. ``resolved`` spends work NOW to produce a field the next agent can
    act on. ``escalated`` is cheap precisely because it does NOT resolve — it
    defers BOTH the cost AND the correctness: the conflicting field stays
    divergent until someone drains the queue, so an agent that needs to read it
    on its next step is blocked. For async knowledge-merge that deferral is
    free; for a synchronous read it re-incurs the inference downstream, plus a
    stall. Escalate is the safer *primitive* (you can always bolt a resolver on
    later); it is not a free lunch on time-to-usable-state.

    * ``auto_merged`` — no semantic conflict in the workload; mergeable writes
                       unioned/concatenated. Every correct strategy lands here
                       on a clean merge. LWW lands here only if it didn't drop
                       writes — otherwise it's ``corrupted``.
    * ``corrupted``   — a conflict (or a concurrent write to a mergeable field)
                       existed and the strategy silently dropped or overwrote
                       it with NO signal. LWW. The corruption baseline.
    * ``resolved``    — a real conflict existed and the strategy spent work
                       (a model call) to autonomously decide and repair.
                       transactional. Correct end state, costs inference, acts
                       without a human in the loop.
    * ``escalated``   — a real conflict existed and the strategy flagged it for
                       a downstream consumer instead of deciding. crdt. Cheap
                       because it defers resolution — see the note above.
    """

    auto_merged = "auto_merged"
    corrupted = "corrupted"
    resolved = "resolved"
    escalated = "escalated"


class FieldKind(str, Enum):
    """How a field's concurrent writes should combine.

    ``scalar`` fields are the source of semantic conflicts. The grow_set and
    append_text kinds are mergeable by construction (union / concatenation) and
    are where the CRDT converges for free.
    """

    scalar = "scalar"
    grow_set = "grow_set"
    append_text = "append_text"


@dataclass(frozen=True)
class Write:
    """A single attributed write by one agent.

    ``agent_id`` and ``op_id`` survive every merge — attribution completeness
    is a first-class measured metric, not a side effect. ``op_id`` orders
    writes from the same agent; ``happens_after`` (optional) lets a workload
    express the partial order ("agent B's write saw agent A's write") so the
    harness can distinguish true concurrency from sequential causality.
    """

    agent_id: str
    op_id: int
    field: str
    value: Any
    kind: FieldKind = FieldKind.scalar
    happens_after: tuple[str, int] | None = None  # (agent_id, op_id) of a write this one observed


@dataclass
class Outcome:
    """What a strategy reports for a single applied write.

    ``applied`` is True if the write is reflected in finalized state. A False
    here under LWW means the write was overwritten (the corruption signal).
    The optional ``escalation`` is non-None only when the strategy detected a
    semantic conflict and surfaced it instead of silently merging.
    """

    applied: bool
    escalation: "Escalation | None" = None
    # Diagnostics the strategy may set; not all are meaningful for every
    # strategy (e.g. model_calls is nonzero only for transactional).
    overwrote_prior: bool = False


@dataclass
class Escalation:
    """A semantic conflict the strategy refused to auto-resolve.

    Per the thesis, CRDT auto-merges mergeable state and ESCALATES semantic /
    un-mergeable conflicts rather than silently picking a winner. LWW never
    escalates (it silently corrupts); transactional escalates via a model call.
    """

    field: str
    contenders: list["Contender"]
    reason: str = "semantic_conflict"

    @dataclass
    class Contender:
        agent_id: str
        op_id: int
        value: Any


@dataclass
class Metrics:
    """Per-run counters accumulated across all writes.

    These map 1:1 to the comparison-table columns. ``writes_lost`` is the
    convergence-correctness signal: it counts writes that were applied to a
    replica but did NOT survive into the final converged state.
    """

    writes_seen: int = 0
    writes_applied: int = 0
    writes_lost: int = 0
    escalations: int = 0
    model_calls: int = 0  # only nonzero for the transactional strategy


class MergeStrategy(Protocol):
    """The ONE interface all three strategies implement.

    A strategy owns one replica's view of the document. The harness creates one
    instance per (strategy, replica) pair, applies that replica's writes to it,
    then merges replicas via :meth:`export_state` / :meth:`import_state`.

    Why merge-by-export rather than a central ``apply_to_all``: it mirrors how
    real multi-agent systems actually replicate — each agent edits a local copy
    and the sync layer reconciles. LWW/transactional simulate this with a
    shared dict + timestamp; CRDT does it natively via Loro frontiers/blobs.
    """

    name: str

    def apply(self, write: Write) -> Outcome: ...

    def export_state(self) -> bytes:
        """Serialize this replica's state for transfer to another replica."""
        ...

    def import_state(self, blob: bytes) -> None:
        """Merge another replica's exported state into this one."""
        ...

    def finalized_state(self) -> dict[str, Any]:
        """Return the human-readable converged document for assertion / display."""
        ...

    def metrics(self) -> Metrics: ...


# ---------------------------------------------------------------------------
# Workload definitions
# ---------------------------------------------------------------------------


@dataclass
class Workload:
    """A named workload: the partial order of writes each agent performs.

    ``writes_by_replica[replica_id]`` is the sequence of writes that replica
    applies LOCALLY before any merge. The harness then merges replicas and
    checks convergence. ``expectation`` states what a CORRECT strategy should
    produce, so the table can show pass/fail per metric, not just numbers.
    """

    name: str
    description: str
    writes_by_replica: dict[str, list[Write]]
    expectation: "Expectation"


@dataclass
class Expectation:
    """What correct behavior looks like for a workload.

    Used to turn raw metrics into pass/fail verdicts in the comparison table.
    ``semantic_conflict_on`` lists fields where a correct strategy MUST
    escalate (non-empty) — empty means the workload is clean-merge and any
    escalation or lost write is a failure.
    """

    clean_merge: bool  # True => no semantic conflict is expected
    all_writes_survive: bool  # True => writes_lost must be 0 for a correct strategy
    semantic_conflict_on: tuple[str, ...] = ()


@dataclass
class RunResult:
    """One row of the comparison table: one strategy on one workload."""

    strategy: str
    workload: str
    converged: bool  # all replicas reached identical finalized_state
    writes_lost: int
    attribution_complete: bool  # every surviving write traces to an agent
    escalations: int
    model_calls: int
    latency_ms: float
    peak_mem_kb: float
    # How the strategy reached its end state — NOT interchangeable across
    # strategies even when both PASS. ``verdict`` says "acceptable end state";
    # ``outcome`` says *how*. This is the one column that keeps escalate vs
    # resolve vs corrupt from masquerading as the same green checkmark.
    outcome: EndState = EndState.auto_merged
    verdict: Literal["PASS", "FAIL"] = "PASS"
    notes: list[str] = field(default_factory=list)


__all__ = [
    "FieldKind",
    "EndState",
    "Write",
    "Outcome",
    "Escalation",
    "Metrics",
    "MergeStrategy",
    "Workload",
    "Expectation",
    "RunResult",
]
