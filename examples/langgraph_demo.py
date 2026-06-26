"""agentsync × LangGraph — the 5-line swap that stops silent write-loss.

WHAT THIS SHOWS
Two parallel agents (researcher + writer) concurrently write to the SAME shared
context key. Run the same graph twice:

  1. with the stock InMemoryStore  -> last-write-wins silently drops a write.
  2. with agentsync.SyncedStore     -> writes merge, every write is attributed,
                                       and a semantic conflict gets escalated.

THE SWAP (the whole adoption story)
    from agentsync import SyncedStore                     # 1
    store = SyncedStore()                                 # 2
    graph = builder.compile(store=store)                  # 3
    # in a node:                                          # 4
    with store.acting_as("researcher"):                   # 5
        store.put(("ctx",), "notes", {...})               # 6  (ok, 6 lines)

Run:  uv run python examples/langgraph_demo.py
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from typing_extensions import TypedDict

from agentsync import SyncedStore
from agentsync.models import Escalation

NS = ("ctx",)
KEY = "shared"


class State(TypedDict):
    trigger: int


def _acting_as(store: BaseStore, agent_id: str):
    """Wrap puts with attribution, but only on a SyncedStore.

    On the stock InMemoryStore this is a no-op contextmanager — which is the
    point: the SAME node code runs against both stores, so the demo is a
    genuine one-line swap (the store constructor), nothing else changes. An
    InMemoryStore can't attribute, so baseline runs are anonymous.
    """
    if isinstance(store, SyncedStore):
        return store.acting_as(agent_id)

    import contextlib

    return contextlib.nullcontext()


# Each node wraps its writes in _acting_as(store, <agent>) so attribution
# threads through the merge on a SyncedStore (and no-ops on the baseline).
def researcher(_: State, store: BaseStore) -> dict:
    with _acting_as(store, "researcher"):
        store.put(NS, KEY, {
            "tags": ["crdt", "agents"],          # list -> mergeable (union)
            "findings": "researcher: found 3 papers",  # text -> mergeable (concat)
            "status": "draft",                   # scalar -> conflict surface
        })
    return {}


def writer(_: State, store: BaseStore) -> dict:
    with _acting_as(store, "writer"):
        store.put(NS, KEY, {
            "tags": ["benchmark"],
            "findings": "writer: drafted intro",
            "status": "published",               # divergent intent -> escalation
        })
    return {}


def build_graph(store: BaseStore):
    g = StateGraph(State)
    g.add_node("researcher", researcher)
    g.add_node("writer", writer)
    g.add_edge(START, "researcher")
    g.add_edge(START, "writer")
    g.add_edge("researcher", END)
    g.add_edge("writer", END)
    return g.compile(store=store)


def _bar(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def main() -> int:
    _bar("SCENARIO — two parallel agents write the SAME shared context key")
    print(f"  researcher: tags={{crdt,agents}}, findings=..., status='draft'")
    print(f"  writer:     tags={{benchmark}},   findings=..., status='published'")

    # ---- 1. Stock store: silent last-write-wins --------------------------
    _bar("BASELINE — langgraph InMemoryStore (the silent corruption)")
    plain = InMemoryStore()
    build_graph(plain).invoke({"trigger": 0})
    base = plain.get(NS, KEY)
    print(f"  final value: {base.value if base else None}")
    print("  -> the losing agent's tags/findings/status are GONE. No error.")
    researcher_survives = base and "crdt" in base.value.get("tags", [])

    # ---- 2. SyncedStore: merge + attribution + escalation ----------------
    _bar("SWAP — agentsync.SyncedStore (same graph, one-line store change)")
    store = SyncedStore()
    escalations: list[Escalation] = []
    store.on_escalation(lambda esc: escalations.append(esc))  # opt-in drain
    build_graph(store).invoke({"trigger": 0})
    synced = store.get(NS, KEY)
    print(f"  final value: {synced.value}")

    _bar("RESULT")
    print("  MERGEABLE WRITES (tags, findings):")
    print(f"    baseline tags : {base.value.get('tags') if base else None}")
    print(f"    synced   tags : {synced.value.get('tags')}   <- both agents preserved")
    print(f"    synced findings: {synced.value.get('findings')!r}   <- concatenated")

    print("\n  SEMANTIC CONFLICT (status: 'draft' vs 'published'):")
    print(f"    baseline status: {base.value.get('status')!r}   <- silently picked")
    if escalations:
        esc = escalations[0]
        contenders = ", ".join(
            f"{c.agent_id}='{c.value}'" for c in esc.contenders
        )
        print(f"    synced   status: ESCALATED (not auto-resolved)")
        print(f"      contenders: {contenders}")
        print(f"      -> flagged with full attribution; nothing silently lost")
    else:
        print("    synced   status: (no escalation recorded)")

    print("\n  ATTRIBUTION (per-write, survives the merge):")
    for key, attrs in store.attribution().items():
        print(f"    {key}:")
        for write_id, meta in attrs.items():
            print(f"      {write_id} -> agent={meta['agent_id']} op={meta['op_id']}")

    _bar("VERDICT")
    if researcher_survives:
        print("  !! baseline unexpectedly preserved researcher's write — demo stale")
    else:
        print("  baseline (InMemoryStore): SILENTLY LOST a concurrent write.")
        print("  synced   (SyncedStore):    MERGED mergeable writes,")
        print("                             ATTRIBUTED every surviving write,")
        print("                             ESCALATED the semantic conflict.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
