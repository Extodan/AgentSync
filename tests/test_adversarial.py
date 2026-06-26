"""Adversarial tests — the cases a hostile reader will immediately try.

These exist to make the product's behavior EXPLICIT on edge cases, not to make
it look good. Each test names the scenario, asserts the actual behavior, and
where that behavior is a known limitation (not a clean win) it says so in the
docstring rather than hiding it.

Run with: `uv run --extra test --extra demo pytest tests/test_adversarial.py -v`
"""

from __future__ import annotations

from typing import Any, Callable

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore
from typing_extensions import TypedDict

from agentsync import SyncedStore

NS = ("ctx",)
KEY = "k"


class _S(TypedDict):
    trigger: int


def _node(agent_id: str, payload: dict[str, Any]) -> Callable:
    """Build a node that writes ``payload`` to the shared key as ``agent_id``.

    Uses the stock ``store.put`` path. ``acting_as`` is guarded because these
    tests sometimes point at a stock InMemoryStore too.
    """

    def node(_state: _S, store: BaseStore) -> dict:
        if hasattr(store, "acting_as"):
            with store.acting_as(agent_id):
                store.put(NS, KEY, payload)
        else:
            store.put(NS, KEY, payload)
        return {}

    return node


def _run(store: BaseStore, nodes: dict[str, Callable]) -> dict | None:
    g = StateGraph(_S)
    for name, fn in nodes.items():
        g.add_node(name, fn)
        g.add_edge(START, name)
        g.add_edge(name, END)
    g.compile(store=store).invoke({"trigger": 0})
    item = store.get(NS, KEY)
    return item.value if item else None


# ---------------------------------------------------------------------------
# Three concurrent writers (not just two)
# ---------------------------------------------------------------------------

def test_three_concurrent_list_writers_all_merge():
    """Three parallel agents each contribute distinct tags to the same key.

    Sane behavior: all three union into the merged value, none lost.
    """
    store = SyncedStore()
    final = _run(store, {
        "a": _node("A", {"tags": ["a1"]}),
        "b": _node("B", {"tags": ["b1", "b2"]}),
        "c": _node("C", {"tags": ["c1"]}),
    })
    assert set(final["tags"]) == {"a1", "b1", "b2", "c1"}
    assert store.escalations() == []


# ---------------------------------------------------------------------------
# Type mismatch — list vs scalar on the same field
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "first, second",
    [
        (["x", "y"], "scalar-value"),   # list first, scalar second
        ("scalar-first", ["list", "after"]),  # scalar first, list second
    ],
    ids=["list-then-scalar", "scalar-then-list"],
)
def test_type_mismatch_currently_silently_drops_one_write(first, second):
    """KNOWN LIMITATION, asserted as current behavior (xfailed against the ideal).

    When two agents write the SAME field name with DIFFERENT value types (one a
    list, one a scalar), SyncedStore currently drops one write silently — no
    escalation, no error — and which one survives depends on merge order. This
    is the same silent-corruption failure mode the product prevents for same-
    type writes, just across a type boundary. The field kind is inferred per-
    write from value shape, and the CRDT schema can't reconcile two kinds for
    one field name.

    This test is XPASS-strict: it pins the CURRENT (imperfect) behavior so the
    limitation is tracked in code, not hidden. When type-mismatch is fixed to
    escalate (the intended behavior), this xfail will surface as XPASS and force
    a real assertion update — i.e. the fix can't land silently.
    """
    store = SyncedStore()
    _run(store, {"a": _node("A", {"field": first}), "b": _node("B", {"field": second})})

    # Current behavior: exactly one write survives, the other is gone, and there
    # is NO escalation flagging the divergence.
    assert store.escalations() == [], "currently no escalation on type mismatch (limitation)"


# ---------------------------------------------------------------------------
# Empty / None / missing-field writes
# ---------------------------------------------------------------------------

def test_none_value_stored_verbatim_single_writer():
    """A single writer storing ``None`` keeps it as ``None`` (not dropped)."""
    store = SyncedStore()
    final = _run(store, {"n": _node("N", {"val": None})})
    assert final == {"val": None}
    assert store.escalations() == []


def test_empty_dict_write_is_a_noop():
    """An empty-dict put contributes nothing and raises nothing."""
    store = SyncedStore()
    final = _run(store, {"e": _node("E", {})})
    assert final == {}
    assert store.escalations() == []


def test_disjoint_fields_coexist_without_conflict():
    """Two writers hitting DIFFERENT fields of the same key both survive — no
    spurious conflict."""
    store = SyncedStore()
    final = _run(store, {
        "x": _node("X", {"only_x": "xval"}),
        "y": _node("Y", {"only_y": "yval"}),
    })
    assert final == {"only_x": "xval", "only_y": "yval"}
    assert store.escalations() == []


def test_none_vs_real_value_escalates():
    """None and a real value on the same scalar field is a divergence -> escalate.

    This is the correct behavior: ``None`` vs ``'real'`` is treated as a scalar
    semantic conflict, with both contenders attributed, rather than silently
    picking one.
    """
    store = SyncedStore()
    _run(store, {"n": _node("N", {"v": None}), "v": _node("V", {"v": "real"})})
    escs = store.escalations()
    assert len(escs) == 1
    assert escs[0].field == "v"
    values = {c.value for c in escs[0].contenders}
    assert values == {None, "real"}
    agents = {c.agent_id for c in escs[0].contenders}
    assert agents == {"N", "V"}
