# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-26

First public (alpha) release. The benchmark is the product; this version adds
the drop-in LangGraph store on top of it.

### Added
- `SyncedStore` — a drop-in for langgraph's `InMemoryStore` (`compile(store=SyncedStore())`)
  that CRDT-merges concurrent writes instead of silently dropping them. Lists
  union, text concatenates, nested dicts deep-merge; every write is attributed
  via `store.acting_as(agent_id)`.
- Semantic-conflict escalation: when two agents set the same scalar to different
  values, the conflict is recorded (both contenders attributed) and NOT silently
  resolved. Drain via `store.on_escalation(cb)` or `store.escalations()`.
- Three-way benchmark harness (`lww` / `transactional` / `crdt`) behind one
  common `MergeStrategy` interface, with two workloads (clean-merge and
  semantic-conflict) and an honest `outcome` column distinguishing
  `corrupted` / `auto_merged` / `resolved` / `escalated`.
- `/examples/langgraph_demo.py` — the landing-page demo: same two-agent graph on
  `InMemoryStore` vs `SyncedStore`, showing silent write-loss vs merge +
  attribution + escalation.
- `py.typed` marker (PEP 561); MIT license; CI on Python 3.10/3.11/3.12.

### Verified
- langgraph 1.2.6 store API: parallel-node `put`s produce two separate `batch()`
  calls; `InMemoryStore` silently clobbers the first, `SyncedStore` merges.
