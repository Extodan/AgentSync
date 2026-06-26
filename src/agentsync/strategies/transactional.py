"""LLM-mediated transactional strategy — approximates CoAgent MTPO.

The thesis frames this as the "correct but expensive" middle strategy. It:

1. Serializes writes in a total order (by ``(agent_id, op_id)``), the way a
   central coordinator / transaction log would.
2. Detects conflicts the same way CRDT does — two concurrent writes to the same
   scalar with different values, or — crucially for the comparison — it ALSO
   has to detect that mergeable concurrent writes are NOT conflicts (otherwise
   it would corrupt them like LWW). Doing that *correctly* requires the
   strategy to understand :class:`FieldKind`.
3. On a REAL (scalar) conflict, instead of CRDT's free structural escalation it
   **spends a model call** to "re-judge" the conflict — i.e. ask an LLM which
   value to keep. We STUB the model call: it sleeps for a fixed, realistic
   latency and returns a deterministic pick, but it still increments
   ``model_calls`` and adds wall-clock — so the *cost* of LLM-mediated
   resolution is measurable and comparable, which is the whole point.

This is the strategy the CRDT is benchmarked *against*. On the clean-merge
workload it should also be correct (0 lost, full attribution) but with 0 model
calls — same as CRDT there. On the conflict workload it escalates correctly
but pays 1+ model calls where CRDT pays 0. That delta is the thesis number.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..models import (
    EndState,
    Escalation,
    FieldKind,
    Metrics,
    MergeStrategy,
    Outcome,
    Write,
)

# A stubbed LLM "re-judge" call. Real MTPO spends ~hundreds of ms + tokens per
# conflict; we model that with a fixed sleep so latency is measurable without
# network dependency. Returns the contender the "model" picks (deterministic:
# lexicographically smallest value, so runs are reproducible).
_STUB_MODEL_LATENCY_S = 0.15


def _stub_model_call(contenders: list[Escalation.Contender]) -> Escalation.Contender:
    """Pretend to ask an LLM to resolve a semantic conflict.

    Deterministic pick (smallest value by str) keeps the benchmark reproducible;
    the *latency* and *call count* are the real signals, not the pick itself.
    """
    time.sleep(_STUB_MODEL_LATENCY_S)
    return min(contenders, key=lambda c: str(c.value))


@dataclass
class _Entry:
    """One write in the transaction log, with surviving attribution."""

    agent_id: str
    op_id: int
    field: str
    value: Any
    kind: FieldKind


class TransactionalStrategy:
    """Central-coordinator-style serialized log with LLM conflict re-judging."""

    name = "transactional"
    # On a real conflict transactional spends a model call to autonomously
    # decide and repair — a usable end state the next agent can read, at the
    # cost of inference and acting without a human.
    conflict_mode = EndState.resolved

    def __init__(
        self,
        model: Callable[[list[Escalation.Contender]], Escalation.Contender] = _stub_model_call,
    ) -> None:
        self._log: list[_Entry] = []
        self._metrics = Metrics()
        self._model = model
        self._escalations: list[Escalation] = []

    # ------------------------------------------------------------------
    # Local apply (writes are buffered into the log)
    # ------------------------------------------------------------------
    def apply(self, write: Write) -> Outcome:
        self._metrics.writes_seen += 1
        self._log.append(
            _Entry(
                agent_id=write.agent_id,
                op_id=write.op_id,
                field=write.field,
                value=_coerce(write.value),
                kind=write.kind,
            )
        )
        self._metrics.writes_applied += 1
        return Outcome(applied=True)

    # ------------------------------------------------------------------
    # Merge = append the other replica's log. The serialized total order is
    # derived at read-out, so import is just a union of logs.
    # ------------------------------------------------------------------
    def export_state(self) -> bytes:
        return json.dumps(
            [e.__dict__ | {"kind": e.kind.value} for e in self._log]
        ).encode()

    def import_state(self, blob: bytes) -> None:
        for rec in json.loads(blob):
            self._log.append(
                _Entry(
                    agent_id=rec["agent_id"],
                    op_id=rec["op_id"],
                    field=rec["field"],
                    value=rec["value"],
                    kind=FieldKind(rec["kind"]),
                )
            )

    # ------------------------------------------------------------------
    # Read-out: replay the serialized log, kind-aware. Scalar conflicts trigger
    # a model call; mergeable writes union/concat like the CRDT does.
    # ------------------------------------------------------------------
    def finalized_state(self) -> dict[str, Any]:
        self._escalations.clear()
        self._metrics.model_calls = 0

        # Dedup writes that arrived via both the local log and an import
        # (full-mesh merge can double-count). Identity = (agent, op, field).
        seen: set[tuple[str, int, str]] = set()
        ordered = sorted(
            (e for e in self._log if (e.agent_id, e.op_id, e.field) not in seen and not seen.add((e.agent_id, e.op_id, e.field))),
            key=lambda e: (e.field, e.agent_id, e.op_id),
        )

        # Group by field to apply kind-specific reconciliation.
        by_field: dict[str, list[_Entry]] = {}
        for e in ordered:
            by_field.setdefault(e.field, []).append(e)

        out: dict[str, Any] = {}
        for fld, entries in by_field.items():
            kind = entries[0].kind
            if kind is FieldKind.grow_set:
                union: set = set()
                for e in entries:
                    union |= set(e.value)
                out[fld] = sorted(union, key=lambda v: (str(type(v)), str(v)))
            elif kind is FieldKind.append_text:
                out[fld] = "".join(e.value for e in entries)
            else:  # scalar — conflict surface
                out[fld] = self._resolve_scalar(fld, entries)
        return out

    def _resolve_scalar(self, field: str, entries: list[_Entry]) -> Any:
        distinct = {str(e.value) for e in entries}
        if len(distinct) <= 1:
            return entries[0].value
        # Semantic conflict detected. MTPO's move: spend a model call to pick.
        contenders = [
            Escalation.Contender(agent_id=e.agent_id, op_id=e.op_id, value=e.value)
            for e in entries
        ]
        self._escalations.append(Escalation(field=field, contenders=contenders))
        self._metrics.model_calls += 1
        picked = self._model(contenders)
        # We record the escalation (so attribution + "it was a conflict"
        # survive) but ALSO take the model's pick as the finalized value —
        # that's the MTPO behavior: resolve rather than leave unresolved.
        return picked.value

    # ------------------------------------------------------------------
    def attribution(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        seen: set[tuple[str, int, str]] = set()
        for e in self._log:
            key = (e.agent_id, e.op_id, e.field)
            if key in seen:
                continue
            seen.add(key)
            out[f"{e.field}:{e.agent_id}:{e.op_id}"] = {
                "agent_id": e.agent_id,
                "op_id": e.op_id,
            }
        return out

    def metrics(self) -> Metrics:
        m = Metrics(
            writes_seen=self._metrics.writes_seen,
            writes_applied=self._metrics.writes_applied,
            model_calls=self._metrics.model_calls,
        )
        m.escalations = len(self._escalations)
        return m


def _coerce(value: Any) -> Any:
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=lambda v: (str(type(v)), str(v)))
    return value


__all__ = ["TransactionalStrategy"]
