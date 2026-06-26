"""Agent-State Sync Engine — benchmark-first.

The thesis under test: for multi-agent MERGEABLE shared state (notes, context,
structured docs), an event-graph CRDT (eg-walker family, here via Loro) gives
deterministic coordinator-free convergence with full write attribution at
ZERO model calls — beating naive last-write-wins (silent corruption) and
LLM-mediated transactional control (costs inference per conflict).

The benchmark is the product. See README for the thesis, build order, and
results.
"""

__version__ = "0.0.1"

from .store import SyncedStore

__all__ = ["SyncedStore"]
