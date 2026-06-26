.PHONY: setup sync bench test demo run clean

setup: ## Create the 3.12 venv and install deps
	uv sync --all-extras

sync: ## Re-resolve and sync deps
	uv sync --all-extras

bench: ## Run the three-way benchmark harness and print the comparison table
	uv run --extra bench python -m agentsync

test: ## Run the invariant test suite
	uv run --extra test pytest -q

demo: ## Run the LangGraph lww-vs-crdt demo (the sendable artifact)
	uv run python -m agentsync.demo.langgraph_demo

example: ## Run the /examples LangGraph swap demo (the landing-page asset)
	uv run --extra demo python examples/langgraph_demo.py

repro: ## Reproduce the stock-langgraph silent-write-loss bug (no agentsync in the write path)
	uv run python -m agentsync.repro

run: ## Run the dev shared-state API (port 8000)
	uv run uvicorn agentsync.api:app --reload --port 8000

clean: ## Remove caches and venv
	rm -rf .venv __pycache__ .ruff_cache *.egg-info
