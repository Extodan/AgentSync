# assets/

Recording for the README and launch post.

## `agentsync.gif`

A ~7s terminal recording of `make repro` (`python -m agentsync.repro`) showing:
1. the stock `InMemoryStore` silently dropping one of two parallel writes (no error), then
2. `SyncedStore` keeping both writes AND escalating the `status` semantic conflict with attribution.

Embedded at the top of `README.md`.

## Re-recording / producing an asciinema cast

The GIF was rendered deterministically with [`vhs`](https://github.com/charmbracelet/vhs)
from `agentsync.tape`. To regenerate (requires `vhs` + `ttyd`, both via Homebrew):

```bash
brew install vhs ttyd
vhs assets/agentsync.tape        # -> assets/agentsync.gif
```

For an asciinema cast instead of a GIF:

```bash
brew install asciinema
asciinema rec --command="uv run python -m agentsync.repro" assets/agentsync.cast
```

## Storyboard (what each frame shows, for a manual one-take recording)

| Beat | Time | On screen |
|---|---|---|
| 1 | 0–2s | `$ make repro` typed and run |
| 2 | 2–4s | `detected langgraph version: 1.2.6` header |
| 3 | 4–7s | **PART 1** — `final store value: {'tags': ['benchmark'], ...}` + `agent_A's tags survived? NO — silently dropped` + `BUG REPRODUCED: ... No exception was raised.` |
| 4 | 7–10s | **PART 2** — `final store value: {'status': '<escalated:status>', 'tags': ['agents','benchmark','crdt']}` + `semantic conflict on 'status' ESCALATED: A='draft', B='published'` |
| 5 | 10–12s | **VERDICT** — `stock InMemoryStore: silent write-loss reproduced` / `SyncedStore: merged tags = [...], 1 escalation(s)` |

The single command `make repro` produces beats 1–5 end-to-end in ~0.3s of compute; the recording just types the command and dwells.
