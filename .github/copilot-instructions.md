# Project Guidelines

## Architecture

- Keep drafting logic centralized in `cricket_draft_tool/core.py`.
- Keep Streamlit UI code in `cricket_draft_tool/web.py`.
- Keep launch/bootstrap code in `cricket_draft_tool/cli.py` and `cricket_draft_tool/__main__.py` only.
- Do not add root-level launcher files for the app; keep executable entrypoints inside the package.

## Algorithm

- Optimize recommendations for **win probability** and **expected margin versus the opponent**, not just raw team score.
- Preserve deterministic simulations by passing explicit seeds into the Monte Carlo engine.
- When changing scoring or pick strategy, add or update focused tests in `tests/test_core.py`.

## Build and Validation

- Use `uv` for dependency management and execution.
- Sync dependencies with `uv sync --dev`.
- Run the app with `uv run cricket-draft-assistant`.
- Run validation with `uv run pytest` and `uv run ruff check .`.

## Conventions

- Avoid duplicate state models or recommendation code across files.
- Recommendation refresh behavior should key off the serialized `DraftState`, simulation settings, and explicit refresh nonce so the UI does not show stale advice.
- Do not keep backward-compatible launchers or redundant app entry files once the package-native path exists.
