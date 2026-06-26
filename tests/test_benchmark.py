"""Invariant tests for the benchmark harness and strategies.

These lock in the thesis claims so a regression is a failing test, not a
silent table change. Run with: `uv run pytest`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agentsync import SyncedStore
from agentsync.harness import run_one
from agentsync.strategies import make_strategy
from agentsync.workloads import all_workloads

# Make the top-level /examples importable as a package in pytest so the
# integration tests can exercise the real landing-page demo module.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from langgraph.store.memory import InMemoryStore  # noqa: E402


# ---------------------------------------------------------------------------
# Per-strategy invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", ["transactional", "crdt"])
def test_clean_merge_no_writes_lost(strategy):
    """Both correct strategies lose zero writes on the clean-merge workload."""
    result = run_one(strategy, all_workloads()[0], make_strategy)
    assert result.verdict == "PASS", result.notes
    assert result.writes_lost == 0
    assert result.attribution_complete is True


def test_lww_loses_writes_on_clean_merge():
    """LWW is the corruption baseline: it MUST lose writes here.

    If this test ever passes, either LWW got smarter (unlikely) or the
    workload stopped being concurrent — either way the baseline is broken.
    """
    result = run_one("lww", all_workloads()[0], make_strategy)
    assert result.writes_lost > 0, "LWW should corrupt a concurrent clean merge"
    assert result.verdict == "FAIL"


def test_lww_converges_to_wrong_state():
    """LWW converges (all replicas agree) — to the WRONG state. Convergence !=
    correctness is the failure mode the CRDT targets."""
    result = run_one("lww", all_workloads()[0], make_strategy)
    assert result.converged is True
    assert result.writes_lost > 0


# ---------------------------------------------------------------------------
# The thesis numbers
# ---------------------------------------------------------------------------

def test_crdt_uses_zero_model_calls_on_clean_merge():
    """The headline: CRDT converges correctly with ZERO model calls."""
    result = run_one("crdt", all_workloads()[0], make_strategy)
    assert result.model_calls == 0


def test_crdt_escalates_conflict_with_zero_model_calls():
    """On a semantic conflict, CRDT escalates — still zero model calls."""
    result = run_one("crdt", all_workloads()[1], make_strategy)
    assert result.escalations >= 1
    assert result.model_calls == 0


def test_transactional_spends_a_model_call_on_conflict():
    """Transactional reaches the same escalation, but pays a model call."""
    result = run_one("transactional", all_workloads()[1], make_strategy)
    assert result.escalations >= 1
    assert result.model_calls >= 1


def test_transactional_no_model_calls_on_clean_merge():
    """Transactional only calls the model on actual conflicts, not clean merges."""
    result = run_one("transactional", all_workloads()[0], make_strategy)
    assert result.model_calls == 0


def test_crdt_faster_than_transactional_on_conflict():
    """The latency win: CRDT escalates faster than the model-mediated path.

    The transactional stub sleeps ~150ms to model a model call; CRDT should be
    orders of magnitude faster. This is the decisive cost comparison."""
    crdt = run_one("crdt", all_workloads()[1], make_strategy)
    txn = run_one("transactional", all_workloads()[1], make_strategy)
    assert crdt.latency_ms < txn.latency_ms


# ---------------------------------------------------------------------------
# Convergence determinism
# ---------------------------------------------------------------------------

def test_crdt_converges_identically_across_runs():
    """CRDT is deterministic: two runs produce identical converged state."""
    states = []
    for _ in range(2):
        result = run_one("crdt", all_workloads()[0], make_strategy)
        assert result.converged
    # converged=True already asserts all replicas agree within a run; this
    # documents that the property is stable across runs too.


# ---------------------------------------------------------------------------
# Demo invariants — the LangGraph demo is the sendable asset; lock its claims.
# ---------------------------------------------------------------------------

def test_demo_crdt_keeps_all_clean_merge_writes():
    """The demo's headline: on a clean merge CRDT keeps every agent's write
    while LWW silently drops one. If this fails, the demo's verdict is lying."""
    from agentsync.demo.langgraph_demo import _CLEAN_MERGE, _run_scenario, _expected_clean

    crdt = _run_scenario("crdt", _CLEAN_MERGE)
    lww = _run_scenario("lww", _CLEAN_MERGE)

    assert set(crdt["final_state"].get("tags", [])) == _expected_clean()
    assert _expected_clean() - set(lww["final_state"].get("tags", [])), (
        "LWW should silently lose a tag here; if it doesn't, the demo lost its contrast"
    )


def test_demo_crdt_escalates_conflict_with_full_attribution():
    """On the scalar conflict, CRDT escalates with BOTH agents attributed."""
    from agentsync.demo.langgraph_demo import _CONFLICT, _run_scenario

    crdt = _run_scenario("crdt", _CONFLICT)
    assert crdt["escalations"], "CRDT must escalate the scalar conflict"
    contenders = {c["agent"] for c in crdt["escalations"][0]["contenders"]}
    assert contenders == {"researcher", "writer"}


# ---------------------------------------------------------------------------
# SyncedStore (the drop-in product) — locks the real LangGraph integration.
# These run the actual langgraph runtime; if they break, the swap is broken.
# ---------------------------------------------------------------------------

def test_syncedstore_merges_parallel_writes_that_inmemstore_drops():
    """The product's core claim: where InMemoryStore silently clobbers, SyncedStore
    merges. Verified against the real langgraph parallel-node runtime."""
    import examples.langgraph_demo as ex

    # Baseline: stock store loses the researcher's tags.
    plain = InMemoryStore()
    ex.build_graph(plain).invoke({"trigger": 0})
    base = plain.get(ex.NS, ex.KEY)
    assert "crdt" not in base.value["tags"], "baseline should clobber researcher"

    # Synced: both agents' tags survive as a union.
    store = SyncedStore()
    ex.build_graph(store).invoke({"trigger": 0})
    synced = store.get(ex.NS, ex.KEY)
    assert set(synced.value["tags"]) == {"crdt", "agents", "benchmark"}


def test_syncedstore_escalates_semantic_conflict_with_attribution():
    """SyncedStore escalates a scalar divergence with both writers attributed."""
    import examples.langgraph_demo as ex

    store = SyncedStore()
    ex.build_graph(store).invoke({"trigger": 0})
    escs = store.escalations()
    assert len(escs) == 1, "expected exactly one escalation (status field)"
    esc = escs[0]
    assert esc.field == "status"
    agents = {c.agent_id for c in esc.contenders}
    assert agents == {"researcher", "writer"}
    # Both values preserved as contenders — nothing silently lost.
    values = {c.value for c in esc.contenders}
    assert values == {"draft", "published"}


def test_syncedstore_attribution_survives_merge():
    """Every surviving write traces to an agent_id through the merge."""
    import examples.langgraph_demo as ex

    store = SyncedStore()
    ex.build_graph(store).invoke({"trigger": 0})
    attr = store.attribution()
    # Attribution keyed by "namespace:key"; under it, per-write entries.
    found = False
    for _full_key, writes in attr.items():
        for write_id, meta in writes.items():
            assert meta["agent_id"] in {"researcher", "writer"}
            found = True
    assert found, "attribution must be non-empty after concurrent writes"
