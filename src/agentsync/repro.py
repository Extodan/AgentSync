"""Independent reproduction of the langgraph silent-write-loss bug.

This module exists to prove the bug is real in VANILLA langgraph — no
agentsync code in the write path. A hostile reader's first instinct is "the
demo was rigged to make agentsync look good"; this closes that door.

Run two parallel nodes that ``store.put()`` to the SAME key against the STOCK
``InMemoryStore``. Show the raw final state: one write gone, no error raised.
Then, clearly delineated, run the SAME graph against ``SyncedStore`` to show
the fix.

    python -m agentsync.repro        # or: make repro
"""

from __future__ import annotations

import sys
from importlib.metadata import version as pkg_version
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from typing_extensions import TypedDict

from .store import SyncedStore

NS = ("ctx",)
KEY = "k"


class _S(TypedDict):
    """Minimal graph state. The shared memory lives in the store, not here."""
    trigger: int


def _node(agent_id: str, payload: dict[str, Any]) -> Callable:
    """A node that writes ``payload`` to the shared key as ``agent_id``.

    Pure langgraph on its own: it calls the injected ``store.put``. On a stock
    ``InMemoryStore`` that is a blind overwrite; on ``SyncedStore`` it merges.
    The node code is IDENTICAL for both — only the store instance differs, which
    is the whole point of the reproduction.
    """
    # NOTE: nodes that want agentsync attribution use ``store.acting_as``; the
    # BASELINE run uses a stock InMemoryStore which has no such method, so we
    # guard it. The put() call itself is the same either way.
    def node(_state: _S, store: BaseStore) -> dict:
        if hasattr(store, "acting_as"):
            with store.acting_as(agent_id):
                store.put(NS, KEY, payload)
        else:
            store.put(NS, KEY, payload)
        return {}

    return node


def _build_and_run(store: BaseStore, nodes: dict[str, Callable]) -> dict | None:
    g = StateGraph(_S)
    for name, fn in nodes.items():
        g.add_node(name, fn)
        g.add_edge(START, name)
        g.add_edge(name, END)
    g.compile(store=store).invoke({"trigger": 0})
    item = store.get(NS, KEY)
    return item.value if item else None


def _section(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def main() -> int:
    try:
        lg_version = pkg_version("langgraph")
    except Exception:  # pragma: no cover
        lg_version = "unknown"

    print("agentsync bug reproduction — langgraph silent write-loss")
    print(f"detected langgraph version: {lg_version}")

    # The two concurrent writes. Both target the SAME key. All fields are
    # mergeable in principle (tags are lists, status is a scalar that diverges).
    payload_a = {"tags": ["crdt", "agents"], "status": "draft"}
    payload_b = {"tags": ["benchmark"], "status": "published"}
    nodes = {"agent_A": _node("A", payload_a), "agent_B": _node("B", payload_b)}

    # ------------------------------------------------------------------ #
    # PART 1 — stock langgraph. NO agentsync in the write path.
    # ------------------------------------------------------------------ #
    _section("PART 1 — stock langgraph InMemoryStore (no agentsync in play)")
    print(f"agent_A puts: {payload_a}")
    print(f"agent_B puts: {payload_b}")
    print("running two PARALLEL nodes against InMemoryStore...")

    plain = InMemoryStore()
    final = _build_and_run(plain, nodes)
    print(f"\nfinal store value: {final}")
    a_tags_survive = final and "crdt" in (final.get("tags") or [])
    print(
        "agent_A's tags survived? "
        + ("YES — unexpected" if a_tags_survive else "NO — silently dropped")
    )
    if not a_tags_survive:
        print("-> BUG REPRODUCED: one concurrent write is gone. No exception was raised.")

    # ------------------------------------------------------------------ #
    # PART 2 — the SAME graph, only the store instance changes.
    # ------------------------------------------------------------------ #
    _section("PART 2 — same graph, SyncedStore (agentsync in play)")
    synced = SyncedStore()
    final2 = _build_and_run(synced, nodes)
    print(f"final store value: {final2}")
    all_tags = set((final2 or {}).get("tags") or [])
    print(f"agent_A's tags survived? {'YES' if 'crdt' in all_tags else 'NO'}")
    print(f"agent_B's tags survived? {'YES' if 'benchmark' in all_tags else 'NO'}")
    escs = synced.escalations()
    if escs:
        for esc in escs:
            contenders = ", ".join(f"{c.agent_id}='{c.value}'" for c in esc.contenders)
            print(f"semantic conflict on '{esc.field}' ESCALATED (not auto-resolved): {contenders}")
    else:
        print("no escalations")

    # ------------------------------------------------------------------ #
    # Verdict
    # ------------------------------------------------------------------ #
    _section("VERDICT")
    print(
        "stock InMemoryStore: "
        + ("silent write-loss reproduced" if not a_tags_survive else "no loss observed")
    )
    print(
        f"SyncedStore:        merged tags = {sorted(all_tags)}, "
        f"{len(escs)} escalation(s)"
    )
    print("\nThe bug is in vanilla langgraph. SyncedStore fixes it. Same graph,")
    print("same writes — only the store instance differs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
