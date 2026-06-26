"""Last-write-wins — the silent-corruption baseline.

LWW is deliberately dumb: it treats every field as an opaque value slot with a
logical timestamp and ignores :class:`FieldKind` entirely. It does not know a
set from a scalar, so concurrent contributions to a *mergeable* field (a grow
set, an appended text) are overwritten wholesale rather than unioned/concat'd.

That ignorance is the whole point of the baseline. On a clean-merge workload
(where a correct strategy loses nothing) LWW still drops writes: it converges
all replicas to the *same wrong state*. Convergence-without-correctness is the
exact failure mode the CRDT is built to eliminate.

Timestamps are a per-replica logical clock starting at 0; ties (concurrent
writes from different replicas land on the same clock value) break on
``(agent_id, op_id)`` for determinism. Real LWW uses a wall clock, which is
*more* arbitrary, not less — the corruption is independent of the tiebreak.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from ..models import EndState, FieldKind, Metrics, MergeStrategy, Outcome, Write


def _wins(challenger: dict, incumbent: dict) -> bool:
    """True if ``challenger`` should replace ``incumbent`` under LWW.

    Higher logical timestamp wins; on a tie the lexicographically larger
    ``(agent_id, op_id)`` wins. Both rules are total, so every pair of replicas
    that import each other converge to identical per-field winners.
    """
    if challenger["ts"] != incumbent["ts"]:
        return challenger["ts"] > incumbent["ts"]
    return (challenger["agent_id"], challenger["op_id"]) > (
        incumbent["agent_id"],
        incumbent["op_id"],
    )


class LWWStrategy:
    """One LWW replica. See module docstring for the corruption model."""

    name = "lww"
    # LWW never signals a conflict — it silently picks a winner and (on
    # mergeable fields) silently drops the loser's writes. That silence is the
    # corruption this strategy exists to demonstrate.
    conflict_mode = EndState.corrupted

    def __init__(self) -> None:
        self._clock = 0  # logical clock; advances only on local apply
        self._state: dict[str, dict[str, Any]] = {}  # field -> {value, ts, agent_id, op_id}
        self._metrics = Metrics()

    def apply(self, write: Write) -> Outcome:
        self._metrics.writes_seen += 1
        ts = self._clock
        self._clock += 1
        # FieldKind is intentionally ignored — LWW is kind-blind. Whether the
        # workload meant this as a scalar set, a set grow, or a text append,
        # LWW stores the opaque value and stamps it.
        self._state[write.field] = {
            "value": _coerce(write.value),
            "ts": ts,
            "agent_id": write.agent_id,
            "op_id": write.op_id,
        }
        self._metrics.writes_applied += 1
        return Outcome(applied=True)

    def export_state(self) -> bytes:
        return json.dumps(self._state, sort_keys=True).encode()

    def import_state(self, blob: bytes) -> None:
        other = json.loads(blob)
        for field, entry in other.items():
            cur = self._state.get(field)
            if cur is None or _wins(entry, cur):
                self._state[field] = entry
            # else: keep incumbent; the other replica's write is silently dropped.

    def finalized_state(self) -> dict[str, Any]:
        return {field: entry["value"] for field, entry in self._state.items()}

    def attribution(self) -> dict[str, dict[str, Any]]:
        """Per-field attribution that survived into finalized state.

        Under LWW this only records the *winning* agent per field; a losing
        agent's contribution (and its attribution) is gone. The harness uses
        this to score attribution completeness.
        """
        return {
            field: {"agent_id": e["agent_id"], "op_id": e["op_id"]}
            for field, e in self._state.items()
        }

    def metrics(self) -> Metrics:
        return self._metrics


def _coerce(value: Any) -> Any:
    """Normalize mutable values (set) into JSON-roundtrippable form.

    LWW persists state as JSON, and a ``set`` isn't JSON-serializable. We sort
    it to a list so export/import is deterministic. (Order inside a value does
    not affect LWW outcomes — only the opaque equality used by the harness.)
    """
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=lambda v: (str(type(v)), str(v)))
    return value


__all__ = ["LWWStrategy"]
