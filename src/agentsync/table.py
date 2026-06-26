"""Comparison-table renderer for the benchmark harness.

One row per (strategy × workload). Columns map 1:1 to the Definition of Done:
correctness (converged + writes_lost + verdict), attribution, model_calls,
latency_ms, peak_mem. Output is plain ASCII so it renders in any terminal and
diffs cleanly in CI; a machine-readable JSON dump is emitted alongside.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from .models import RunResult


# Column spec: (header, width, extractor). width includes padding.
_COLUMNS = [
    ("workload",      18, lambda r: r.workload),
    ("strategy",      12, lambda r: r.strategy),
    ("verdict",        7, lambda r: r.verdict),
    # The honest-labeling column: three greens are NOT the same. `outcome`
    # sits right next to verdict so the tradeoff is impossible to miss.
    ("outcome",       11, lambda r: r.outcome.value),
    ("converged",      9, lambda r: "yes" if r.converged else "NO"),
    ("writes_lost",   12, lambda r: str(r.writes_lost)),
    ("attrib",         9, lambda r: "full" if r.attribution_complete else "PARTIAL"),
    ("escals",         7, lambda r: str(r.escalations)),
    ("model_calls",   12, lambda r: str(r.model_calls)),
    ("latency_ms",    11, lambda r: f"{r.latency_ms:.3f}"),
    ("peak_mem_kb",   12, lambda r: f"{r.peak_mem_kb:.0f}"),
]


def render_table(results: list[RunResult]) -> str:
    header = "  ".join(h.ljust(w) for h, w, _ in _COLUMNS)
    sep = "  ".join("-" * w for _, w, _ in _COLUMNS)
    lines = [header, sep]
    for r in results:
        lines.append(
            "  ".join(fn(r).ljust(w) for _, w, fn in _COLUMNS)
        )
    # Notes (failure explanations) on trailing lines, indented under the row.
    for r in results:
        for n in r.notes:
            lines.append(f"    ↳ [{r.workload}/{r.strategy}] {n}")
    return "\n".join(lines)


def render_json(results: list[RunResult]) -> str:
    return json.dumps([asdict(r) for r in results], indent=2)


__all__ = ["render_table", "render_json"]
