"""SyncedStore — the drop-in shared-state backend that fixes LangGraph's silent
write-loss.

The problem this exists to solve (verified against langgraph 1.2.6): when two
parallel nodes call ``store.put`` on the SAME key, ``InMemoryStore`` silently
overwrites the first write with the second. No error, no merge, no signal —
last-write-wins, A's contribution gone. A team wiring parallel agents to shared
context never learns a write was dropped.

``SyncedStore`` subclasses ``InMemoryStore`` so it's a 1-line swap, and routes
every ``PutOp`` through a per-key CRDT engine (the same Loro/eg-walker merge
from the benchmark). Concurrent writes to the SAME key now:

* **merge** when they're mergeable — list values union (set-style), nested dicts
  deep-merge, text-convention keys concatenate. No write is lost.
* **escalate** when they're a semantic conflict — two agents setting the same
  SCALAR field to different values. The conflict is recorded as an
  :class:`Escalation` event with both contenders attributed, and NOT silently
  resolved. The field holds a sentinel until someone drains the escalation.

Every write is attributed to an ``agent_id``. langgraph's ``store.put`` has no
agent parameter, so attribution is wired through a contextvar: wrap a node's
puts in ``with store.acting_as(agent_id):`` (one extra line) and every put
inside carries that id through merges.

The rest of the store — ``get``/``delete``/``search`` — is inherited unchanged
from ``InMemoryStore``; only the write path is intercepted.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable

from langgraph.store.base import Op, PutOp
from langgraph.store.memory import InMemoryStore

from .models import Escalation, FieldKind, Write
from .strategies.crdt import CRDTStrategy

if TYPE_CHECKING:
    from collections.abc import Iterator

# The agent_id attribution context. langgraph's batch() receives no config, so
# a contextvar is the honest channel for "which agent is writing right now". It
# defaults to "anonymous" so the store never hard-fails on an unwrapped put —
# but attribution completeness then can't be guaranteed, and metrics say so.
_current_agent: contextvars.ContextVar[str] = contextvars.ContextVar(
    "agentsync_agent_id", default="anonymous"
)


@dataclass
class _KeyEngine:
    """One CRDT engine per logical key ((namespace, key) pair).

    langgraph calls ``batch`` once per node per key per superstep, so a key hit
    by N parallel nodes sees N separate writes — each lands here and the engine
    merges them. This is the per-key accumulator the merge lives in.
    """

    engine: CRDTStrategy = field(default_factory=CRDTStrategy)
    # Monotone op counter per agent so the CRDT can order that agent's writes.
    _ops: dict[str, int] = field(default_factory=dict)

    def next_op(self, agent_id: str) -> int:
        n = self._ops.get(agent_id, 0)
        self._ops[agent_id] = n + 1
        return n

    def state(self) -> dict[str, Any]:
        return self.engine.finalized_state()

    def escalations(self) -> list[Escalation]:
        # finalized_state() recomputes escalations into the engine's list.
        self.engine.finalized_state()
        return list(self.engine._escalations)


def _classify_field(key: str, value: Any) -> FieldKind:
    """Heuristic field-kind for a JSON-store value.

    The benchmark workloads declared kinds explicitly; a drop-in store only sees
    raw JSON values, so we infer. Lists are mergeable (union); string values
    under a *-text / notes / findings key are append-style; everything else is a
    scalar (the conflict surface). Conservative: when unsure, scalar — which
    means "escalate on divergence" rather than "silently merge".
    """
    if isinstance(value, list):
        return FieldKind.grow_set
    if isinstance(value, str) and any(
        tag in key.lower() for tag in ("text", "notes", "findings", "log")
    ):
        return FieldKind.append_text
    return FieldKind.scalar


class SyncedStore(InMemoryStore):
    """``InMemoryStore`` whose writes CRDT-merge instead of last-write-wins.

    Drop-in: ``graph = builder.compile(store=SyncedStore())``. Get merges +
    attribution + escalation for free; lose nothing that ``InMemoryStore``
    already does.
    """

    def __init__(self) -> None:
        super().__init__()
        # One CRDT engine per key. Keys are (namespace, key) tuples.
        self._engines: dict[tuple[tuple[str, ...], str], _KeyEngine] = {}

    # ------------------------------------------------------------------
    # Attribution context
    # ------------------------------------------------------------------
    @contextmanager
    def acting_as(self, agent_id: str) -> "Iterator[SyncedStore]":
        """Scope the agent_id attributed to puts inside this block.

        Example::

            with store.acting_as("researcher"):
                store.put(("ctx",), "notes", {...})   # attributed to researcher

        Why a contextvar and not a put() arg: langgraph's store API has no agent
        parameter, and ``batch()`` receives no config. A contextvar is the one
        channel that threads an id from a node into the store without forking
        langgraph internals.
        """
        token = _current_agent.set(agent_id)
        try:
            yield self
        finally:
            _current_agent.reset(token)

    @property
    def current_agent(self) -> str:
        return _current_agent.get()

    # ------------------------------------------------------------------
    # The write path — intercept PutOps and route through the CRDT engine
    # ------------------------------------------------------------------
    def batch(self, ops: Iterable[Op]) -> list:
        """Override the chokepoint: merge same-key concurrent writes.

        ``put`` is concrete in ``BaseStore`` and calls ``self.batch([PutOp])``,
        so this is the single place every write passes through. For each
        ``PutOp`` we (a) feed each field of the value into the key's CRDT
        engine, (b) read back the merged value, (c) hand a *merged* PutOp to
        super().batch so the parent storage reflects the union, not the clobber.
        """
        prepared: list[Op] = []
        for op in ops:
            if isinstance(op, PutOp) and op.value is not None:
                merged = self._merge_through_crdt(op)
                op = PutOp(
                    op.namespace, op.key, merged, index=op.index, ttl=op.ttl
                )
            prepared.append(op)
        return super().batch(prepared)

    def _merge_through_crdt(self, op: PutOp) -> dict[str, Any]:
        """Feed one PutOp's value fields into the key's CRDT, return merged value.

        The store's value model is a flat ``dict[str, JSON]``. Each top-level
        field becomes a CRDT field; its kind is inferred from value shape. The
        engine merges concurrent fields across the multiple batch() calls the
        runtime makes for parallel nodes.
        """
        k = (op.namespace, op.key)
        engine = self._engines.setdefault(k, _KeyEngine())
        agent_id = _current_agent.get()
        for field_name, field_value in op.value.items():
            kind = _classify_field(field_name, field_value)
            # Coerce list into the set semantics the CRDT grow_set expects.
            crdt_value = set(field_value) if kind is FieldKind.grow_set else field_value
            engine.engine.apply(
                Write(
                    agent_id=agent_id,
                    op_id=engine.next_op(agent_id),
                    field=field_name,
                    value=crdt_value,
                    kind=kind,
                )
            )
        # Read back the merged, materialized state — this is what gets stored.
        merged = engine.state()
        # Surface CRDT escalations on the store so a caller can drain them.
        for esc in engine.escalations():
            self._on_escalation(esc)
        return merged

    # ------------------------------------------------------------------
    # Escalation surface
    # ------------------------------------------------------------------
    # Callable hook: a consumer wires ``store.on_escalation(my_callback)`` to be
    # notified instead of polling. Deliberately just a function attribute, not a
    # queue/worker — the consumer's shape (human? supervisor? retry?) is unknown
    # until validated, so we surface the event and get out of the way.
    def on_escalation(self, callback: Callable[[Escalation], None]) -> None:
        self._escalation_cb = callback  # type: ignore[attr-defined]

    def _on_escalation(self, esc: Escalation) -> None:
        cb = getattr(self, "_escalation_cb", None)
        if cb is not None:
            cb(esc)

    def escalations(self) -> list[Escalation]:
        """All semantic conflicts observed across all keys (drain point)."""
        out: list[Escalation] = []
        for engine in self._engines.values():
            out.extend(engine.escalations())
        return out

    def attribution(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Per-key, per-write attribution: which agent wrote each surviving field."""
        return {
            f"{ns}:{key}": eng.engine.attribution()
            for (ns, key), eng in self._engines.items()
        }


__all__ = ["SyncedStore"]
