"""LangGraph demo — the asset to send, not plumbing to polish.

The 60-second pitch: watch LWW silently corrupt an agent's shared state, then
watch CRDT not. The SAME two-agent graph runs twice — once with an LWW store
backend, once with a CRDT store backend — and the demo prints exactly what each
agent wrote, what survived the merge, and (for CRDT) what got escalated.

Why each agent writes to its OWN replica then syncs (rather than both writing
one shared dict): that's how real multi-agent shared memory actually works.
Each subagent edits a local copy of the context and the sync layer reconciles.
langgraph 1.x raises ``InvalidUpdateError`` if two parallel nodes write the same
state key with no reducer — so we bypass state for the shared memory itself and
use the graph purely to model the *concurrency* (two agents acting at once),
which is the honest mapping.

Demo runs two scenarios back to back:

* ``clean_merge``  — both agents add distinct findings to shared notes/tags.
  LWW drops one agent's findings; CRDT keeps both. Convergence-without-loss.
* ``conflict``     — both agents set the same scalar to different values.
  LWW silently picks one; CRDT escalates with both contenders attributed.

The point is visual: same agents, same writes, opposite outcomes by backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from ..models import FieldKind, MergeStrategy, Write
from ..strategies import make_strategy


class GraphState(TypedDict):
    """Per-agent scratch passed through the graph. The shared memory itself
    lives in the Store (one replica per agent), NOT in graph state — otherwise
    langgraph's reducer guard would intercept the very concurrency we want to
    measure."""
    agent_id: str


@dataclass
class StoreHandle:
    """One agent's handle onto shared memory: its own replica + an op counter.

    The agent writes locally (its replica is isolated from the other agent's),
    then the harness syncs replicas. This mirrors each subagent holding a local
    copy of shared context and the sync engine reconciling after.
    """

    agent_id: str
    replica: MergeStrategy
    _op: int = 0

    def write(self, field: str, value, kind: FieldKind = FieldKind.scalar) -> None:
        self.replica.apply(
            Write(
                agent_id=self.agent_id,
                op_id=self._op,
                field=field,
                value=value,
                kind=kind,
            )
        )
        self._op += 1


def _agent_node(agent_id: str, field: str, value, kind: FieldKind):
    """Build a node fn that writes one contribution to THIS agent's replica.

    The agent_id is bound into the closure at construction (not read from graph
    state, which is shared across the whole graph). Returns an empty state
    update because the real payload lives in the replica, not in graph state.
    """

    def node(state: GraphState) -> dict:
        handle = _HANDLES[agent_id]
        handle.write(field, value, kind)
        return {}

    return node


# Module-level registry so node closures can find their handle without it being
# serializable graph state. Reset per-run by _run_scenario.
_HANDLES: dict[str, StoreHandle] = {}


def _make_handles(strategy_name: str, agent_ids: list[str]) -> None:
    _HANDLES.clear()
    for aid in agent_ids:
        _HANDLES[aid] = StoreHandle(agent_id=aid, replica=make_strategy(strategy_name))


def _run_scenario(strategy_name: str, agent_work: dict[str, list[tuple]]) -> dict:
    """Run one backend through the graph. Returns what each agent wrote and the
    merged result.

    ``agent_work`` maps agent_id -> list of (field, value, kind) writes. All
    agents run as parallel branches off START (concurrent), then a sync step
    merges every replica into every other (full mesh) and we read out.
    """
    agent_ids = list(agent_work.keys())
    _make_handles(strategy_name, agent_ids)

    g = StateGraph(GraphState)
    for aid in agent_ids:
        # Each agent becomes its own branch off START. Multiple writes per
        # agent chain into a linear sub-sequence ending at END.
        prev = START
        for i, (field, value, kind) in enumerate(agent_work[aid]):
            node_name = f"{aid}_{i}"
            g.add_node(node_name, _agent_node(aid, field, value, kind))
            g.add_edge(prev, node_name)
            prev = node_name
        g.add_edge(prev, END)
    compiled = g.compile()

    # Fan out: START feeds every agent's first node. We invoke with each agent
    # id present so nodes can route to their handle.
    compiled.invoke({"agent_id": agent_ids[0]})

    # Phase 2 — sync. Full-mesh merge of replicas (each imports every other).
    handles = [_HANDLES[aid] for aid in agent_ids]
    for h in handles:
        for other in handles:
            if other is h:
                continue
            h.replica.import_state(other.replica.export_state())

    final = handles[0].replica.finalized_state()
    escalations = []
    drain = getattr(handles[0].replica, "_escalations", None)
    if drain:
        for esc in drain:
            escalations.append(
                {
                    "field": esc.field,
                    "contenders": [
                        {"agent": c.agent_id, "op": c.op_id, "value": c.value}
                        for c in esc.contenders
                    ],
                }
            )
    return {"final_state": final, "escalations": escalations, "strategy": strategy_name}


# ---------------------------------------------------------------------------
# The two scenarios. Mirrored from the benchmark workloads so the demo and the
# table tell the same story.
# ---------------------------------------------------------------------------

_CLEAN_MERGE = {
    "researcher": [
        ("project", "AgentSync", FieldKind.scalar),
        ("findings", "found 3 CRDT papers", FieldKind.append_text),
        ("tags", {"crdt", "agents"}, FieldKind.grow_set),
    ],
    "writer": [
        ("findings", "drafted intro section", FieldKind.append_text),
        ("tags", {"benchmark"}, FieldKind.grow_set),
    ],
}

_CONFLICT = {
    "researcher": [("status", "draft", FieldKind.scalar)],
    "writer": [("status", "published", FieldKind.scalar)],
}


def _expected_clean() -> set:
    return {"crdt", "agents", "benchmark"}


def main() -> int:
    print()
    print("=" * 72)
    print("AGENTSYNC LANGGRAPH DEMO — same graph, two shared-memory backends")
    print("=" * 72)

    for label, work, strategy_pair in [
        ("SCENARIO 1 — clean merge (mergeable concurrent writes)", _CLEAN_MERGE, ("lww", "crdt")),
        ("SCENARIO 2 — semantic conflict (same scalar, different values)", _CONFLICT, ("lww", "crdt")),
    ]:
        print()
        print("-" * 72)
        print(label)
        print("-" * 72)
        what = {}
        for aid, writes in work.items():
            what[aid] = [(f, v) for f, v, _ in writes]
            print(f"  {aid} writes: {what[aid]}")

        for strat in strategy_pair:
            res = _run_scenario(strat, work)
            print(f"\n  [{strat}] merged state: {res['final_state']}")
            if res["escalations"]:
                print(f"  [{strat}] ESCALATED (flagged, not auto-resolved):")
                for esc in res["escalations"]:
                    print(f"      field={esc['field']!r} contenders={esc['contenders']}")
            else:
                print(f"  [{strat}] no escalations")

    # Verdict line: make the corruption-vs-convergence contrast explicit.
    print()
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    lww_clean = _run_scenario("lww", _CLEAN_MERGE)
    crdt_clean = _run_scenario("crdt", _CLEAN_MERGE)
    lww_tags = set(lww_clean["final_state"].get("tags", []))
    crdt_tags = set(crdt_clean["final_state"].get("tags", []))
    expected = _expected_clean()
    print(f"  clean merge — expected tags: {sorted(expected)}")
    print(f"  LWW  kept tags: {sorted(lww_tags)}  -> {'OK' if lww_tags == expected else 'LOST ' + str(sorted(expected - lww_tags))}")
    print(f"  CRDT kept tags: {sorted(crdt_tags)} -> {'OK' if crdt_tags == expected else 'LOST ' + str(sorted(expected - crdt_tags))}")

    crdt_conflict = _run_scenario("crdt", _CONFLICT)
    lww_conflict = _run_scenario("lww", _CONFLICT)
    print(f"\n  semantic conflict on 'status':")
    print(f"  LWW  -> status={lww_conflict['final_state'].get('status')!r}, escalated: {bool(lww_conflict['escalations'])}  (silently picked)")
    print(f"  CRDT -> status={crdt_conflict['final_state'].get('status')!r}, escalated: {bool(crdt_conflict['escalations'])}  (flagged for review)")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
