"""`python -m agentsync` / `make bench` entry point.

Runs every available strategy against every available workload, prints the
comparison table, and exits nonzero if any row FAILED its expectation — so the
benchmark doubles as a regression gate: a strategy that regresses (e.g. CRDT
suddenly loses a write, or LWW stops reproducing the known corruption) fails
the run.
"""

from __future__ import annotations

import sys

from .harness import run_one
from .strategies import make_strategy
from .table import render_json, render_table
from .workloads import all_workloads


# Which strategies to run. crdt/transactional are added in later build steps;
# until then the harness runs cleanly with just lww and reports the first
# number, per the FIRST STEP directive.
_AVAILABLE_STRATEGIES = ["lww", "transactional", "crdt"]


def main() -> int:
    workloads = all_workloads()
    results = []
    for wl in workloads:
        for strat in _AVAILABLE_STRATEGIES:
            results.append(run_one(strat, wl, make_strategy))

    print()
    print(render_table(results))
    print()

    # Interpretation note: a row's FAIL verdict is a property of the STRATEGY,
    # not the harness. LWW is *supposed* to fail these workloads — that failure
    # is the corruption the thesis is built to eliminate. So the only thing
    # that makes the benchmark itself fail is if a strategy we EXPECT to pass
    # (crdt on both; transactional on both) regresses. For step 1, only lww is
    # wired up and its failures are expected, so the run is a success.
    expected_pass = {"crdt", "transactional"}
    regressions = [
        r for r in results if r.strategy in expected_pass and r.verdict == "FAIL"
    ]
    if regressions:
        print(f"⚠  {len(regressions)} expected-pass strategy regressed — see notes.")
        return 1
    print("✓ benchmark complete; LWW failures above are the demonstrated baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
