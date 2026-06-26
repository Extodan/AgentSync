"""CRDT strategy — Loro / eg-walker deterministic merge. The thesis.

How each FieldKind maps onto Loro — using **named root containers** so two
replicas reference the *same logical CRDT object* and concurrent edits merge:

* ``grow_set``    → ``LoroList`` at root name ``set:<field>``. Concurrent
  inserts union automatically; no write is ever lost. The CRDT's free win.
* ``append_text`` → ``LoroText`` at root name ``text:<field>``. Concurrent
  appends concatenate (order among concurrent appends is arbitrary but stable
  and identical at every replica — deterministic convergence).
* ``scalar``      → ``LoroMap`` at root name ``scalar:<field>``. Each scalar
  write is an attributed record keyed by ``"{agent}:{op}"``. This is THE
  semantic-conflict surface: two concurrent writes with different values
  coexist as contenders and the strategy DETECTS the divergence at finalize →
  explicit :class:`Escalation`. We deliberately do NOT use a single Loro
  scalar cell (that's last-writer-wins = silent corruption = the LWW baseline).

A separate ``schema`` root map records each field's kind so read-out knows how
to materialize it. Schema keys are plain scalar strings, so they survive the
merge and agree across replicas.

Export/import is straight Loro snapshot merge: ``export(ExportMode.Snapshot())``
→ ``import_(blob)``. Loro's eg-walker internals guarantee all replicas reach
byte-identical state with no coordinator and zero model calls.

Attribution: every write carries (agent_id, op_id). For mergeable fields we
record per-element attribution in a parallel ``attr:<field>`` LoroMap (the
element at index i was written by attr[i]); for scalars the contender record
itself carries the attribution. So after a merge we can enumerate exactly which
agent produced what — the thing LWW cannot do (it collapses N writers to 1
winner per field).
"""

from __future__ import annotations

import json
from typing import Any

import loro

from ..models import (
    EndState,
    Escalation,
    FieldKind,
    Metrics,
    MergeStrategy,
    Outcome,
    Write,
)


def _set_name(field: str) -> str:
    return f"set:{field}"


def _text_name(field: str) -> str:
    return f"text:{field}"


def _scalar_name(field: str) -> str:
    return f"scalar:{field}"


def _attr_name(field: str) -> str:
    return f"attr:{field}"


class CRDTStrategy:
    """One Loro-backed replica. See module docstring for the merge model."""

    name = "crdt"
    # On a real conflict CRDT flags it for a downstream consumer instead of
    # deciding — cheap because it defers BOTH cost and correctness; the
    # conflicting field stays divergent until someone drains the queue.
    conflict_mode = EndState.escalated

    def __init__(self) -> None:
        self._doc = loro.LoroDoc()
        self._metrics = Metrics()
        self._escalations: list[Escalation] = []

    # ------------------------------------------------------------------
    # Local apply
    # ------------------------------------------------------------------
    def apply(self, write: Write) -> Outcome:
        self._metrics.writes_seen += 1
        self._note_schema(write.field, write.kind)

        if write.kind is FieldKind.grow_set:
            self._apply_set(write)
        elif write.kind is FieldKind.append_text:
            self._apply_text(write)
        else:  # scalar
            self._apply_scalar(write)

        self._metrics.writes_applied += 1
        return Outcome(applied=True)

    def _apply_set(self, write: Write) -> None:
        lst = self._doc.get_list(_set_name(write.field))
        attr = self._doc.get_map(_attr_name(write.field))
        existing = {self._unwrap(v) for v in lst.to_vec()}
        # Append each new element; attribute the whole write under a single
        # ``agent:op`` key (NOT an element-index key, which collides across
        # concurrent writers and would silently drop attribution on merge).
        added = False
        for item in sorted(set(write.value), key=lambda v: (str(type(v)), str(v))):
            if item not in existing:
                lst.push(item)
                added = True
        if added:
            attr.insert(
                f"{write.agent_id}:{write.op_id}",
                json.dumps({"agent_id": write.agent_id, "op_id": write.op_id}),
            )

    def _apply_text(self, write: Write) -> None:
        text = self._doc.get_text(_text_name(write.field))
        # Append, with a separator when extending existing text so concurrent
        # appends remain distinguishable (and the loss-check can find each).
        current = text.to_string()
        frag = ("\n" if current else "") + write.value
        text.push_str(frag)
        # Attribution: mark the byte-range we just appended to this agent.
        attr = self._doc.get_map(_attr_name(write.field))
        attr.insert(
            f"{write.agent_id}:{write.op_id}",
            json.dumps({"agent_id": write.agent_id, "op_id": write.op_id}),
        )

    def _apply_scalar(self, write: Write) -> None:
        registry = self._doc.get_map(_scalar_name(write.field))
        key = f"{write.agent_id}:{write.op_id}"
        registry.insert(
            key,
            json.dumps(
                {"agent_id": write.agent_id, "op_id": write.op_id, "value": write.value},
                sort_keys=True,
            ),
        )

    def _note_schema(self, field: str, kind: FieldKind) -> None:
        schema = self._doc.get_map("schema")
        # set-if-absent: concurrent first-writers from different replicas all
        # write the same kind string, so no divergence regardless of order.
        if field not in [k for k, _ in schema.items()]:
            schema.insert(field, kind.value)

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------
    def export_state(self) -> bytes:
        self._doc.commit()
        return self._doc.export(loro.ExportMode.Snapshot())

    def import_state(self, blob: bytes) -> None:
        # Loro merge: union semantics, never discards local data, deterministic
        # at every replica. This single line is the entire "sync engine".
        self._doc.import_(blob)

    # ------------------------------------------------------------------
    # Read-out + semantic-conflict detection
    # ------------------------------------------------------------------
    def finalized_state(self) -> dict[str, Any]:
        """Materialize the merged doc, escalating any scalar divergence."""
        self._escalations.clear()
        schema = self._doc.get_map("schema")
        out: dict[str, Any] = {}
        for field, raw_kind in schema.items():
            kind_name = raw_kind.value if hasattr(raw_kind, "value") else raw_kind
            kind = FieldKind(kind_name)
            if kind is FieldKind.grow_set:
                out[field] = sorted(
                    (self._unwrap(v) for v in self._doc.get_list(_set_name(field)).to_vec()),
                    key=lambda v: (str(type(v)), str(v)),
                )
            elif kind is FieldKind.append_text:
                out[field] = self._doc.get_text(_text_name(field)).to_string()
            else:  # scalar
                out[field] = self._resolve_scalar(field)
        return out

    def _resolve_scalar(self, field: str) -> Any:
        registry = self._doc.get_map(_scalar_name(field))
        records = {}
        for key, raw in registry.items():
            val = raw.value if hasattr(raw, "value") else raw
            try:
                records[key] = json.loads(val)
            except (TypeError, json.JSONDecodeError):
                records[key] = {"value": val}
        distinct = {r["value"] for r in records.values()}
        if len(distinct) <= 1:
            return next(iter(distinct), None)
        # Divergence = semantic conflict. ESCALATE instead of picking a winner.
        contenders = [
            Escalation.Contender(
                agent_id=r["agent_id"], op_id=r["op_id"], value=r["value"]
            )
            for r in records.values()
        ]
        self._escalations.append(Escalation(field=field, contenders=contenders))
        # Sentinel identical across replicas so convergence comparison holds.
        return f"<escalated:{field}>"

    # ------------------------------------------------------------------
    # Attribution
    # ------------------------------------------------------------------
    def attribution(self) -> dict[str, dict[str, Any]]:
        """Per-write attribution that survived the merge.

        Returns one entry per surviving write, keyed ``"<field>:<agent>:<op>"``.
        The harness compares this against the workload's issued writes to score
        attribution completeness.
        """
        out: dict[str, dict[str, Any]] = {}
        schema = self._doc.get_map("schema")
        for field, raw_kind in schema.items():
            kind_name = raw_kind.value if hasattr(raw_kind, "value") else raw_kind
            kind = FieldKind(kind_name)
            if kind is FieldKind.grow_set:
                attr = self._doc.get_map(_attr_name(field))
                for key, raw in attr.items():
                    val = raw.value if hasattr(raw, "value") else raw
                    try:
                        rec = json.loads(val)
                    except (TypeError, json.JSONDecodeError):
                        continue
                    out[f"{field}:{key}"] = {
                        "agent_id": rec["agent_id"],
                        "op_id": rec["op_id"],
                    }
            elif kind is FieldKind.append_text:
                attr = self._doc.get_map(_attr_name(field))
                for key, raw in attr.items():
                    val = raw.value if hasattr(raw, "value") else raw
                    try:
                        rec = json.loads(val)
                    except (TypeError, json.JSONDecodeError):
                        continue
                    out[f"{field}:{key}"] = {
                        "agent_id": rec["agent_id"],
                        "op_id": rec["op_id"],
                    }
            else:  # scalar
                registry = self._doc.get_map(_scalar_name(field))
                for key, raw in registry.items():
                    val = raw.value if hasattr(raw, "value") else raw
                    try:
                        rec = json.loads(val)
                    except (TypeError, json.JSONDecodeError):
                        continue
                    out[f"{field}:{key}"] = {
                        "agent_id": rec["agent_id"],
                        "op_id": rec["op_id"],
                    }
        return out

    def metrics(self) -> Metrics:
        m = Metrics(
            writes_seen=self._metrics.writes_seen,
            writes_applied=self._metrics.writes_applied,
        )
        m.escalations = len(self._escalations)
        # CRDT resolves mergeable conflicts structurally — zero model calls.
        m.model_calls = 0
        return m

    # ------------------------------------------------------------------
    @staticmethod
    def _unwrap(v):
        return v.value if hasattr(v, "value") else v


__all__ = ["CRDTStrategy"]
