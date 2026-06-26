"""Strategies for reconciling concurrent agent writes to shared state.

Each module implements :class:`agentsync.models.MergeStrategy` behind the ONE
common interface so the benchmark harness can swap them without touching
workload or measurement code.

* :mod:`lww`           — last-write-wins. The silent-corruption baseline.
* :mod:`transactional` — LLM-mediated conflict resolution (CoAgent MTPO approx).
* :mod:`crdt`          — Loro / eg-walker deterministic merge.

Exported here as factory callables (``make_strategy(name)``) so the harness and
workloads share a single registry.
"""

from __future__ import annotations

from ..models import MergeStrategy


def make_strategy(name: str) -> MergeStrategy:
    """Construct a fresh replica for the named strategy.

    Raising on unknown names (rather than returning None) keeps a typo from
    silently turning the benchmark into a two-strategy comparison.
    """
    # Imported lazily so that a missing optional dep (e.g. loro) only blows up
    # when its strategy is actually selected, not on package import.
    if name == "lww":
        from .lww import LWWStrategy

        return LWWStrategy()
    if name == "transactional":
        from .transactional import TransactionalStrategy

        return TransactionalStrategy()
    if name == "crdt":
        from .crdt import CRDTStrategy

        return CRDTStrategy()
    raise ValueError(f"unknown strategy: {name!r}")


__all__ = ["make_strategy"]
