# Cricket Draft Assistant

An opponent-aware Streamlit app for drafting balanced cricket teams from a shared player pool.

It uses deterministic Monte Carlo simulation to recommend picks based on **win probability** and **expected margin versus the opponent**, not just raw player ratings. The app is designed for captain-vs-captain draft formats where each side starts with a captain and alternates picks from the remaining pool.

## What it does

- Upload a CSV of players with batting and bowling ratings.
- Define both captains and who picks first.
- Run a guided draft through a modern Streamlit UI.
- Get pick recommendations powered by opponent-aware Monte Carlo simulations.
- See win probability, expected margin, steal risk, uncertainty, and close-call groupings.
- Save a draft state to JSON and restore it later.
- Review final teams and projected strengths when the draft finishes.

## Highlights

- **Opponent-aware recommendations** using weighted opponent drafting personas
- **Deterministic simulations** through explicit seeded randomness
- **Balanced team evaluation** with batting depth, bowling coverage, overs capacity, and all-rounder value
- **Adaptive simulation strategy** that screens the whole pool and refines the most relevant candidates
- **Close-call signaling** when several top picks are statistically within the uncertainty band
- **Fresh recommendation caching** keyed to draft state, simulation settings, and an explicit refresh nonce so the UI does not show stale advice
- **Package-native Python app** managed with `uv`

## How recommendations work

The recommendation engine lives in `cricket_draft_assistant/core.py`.

At a high level, the app:

1. Evaluates the current draft state.
2. Simulates the rest of the draft many times for each available candidate.
3. Models the opponent as a weighted mix of drafting personas:
   - balanced
   - run-feast aggressor
   - bowling coverage prioritizer
4. Scores the resulting teams using:
   - batting depth
   - bowling strength
   - bowling overs coverage
   - reliable bowler count
   - all-rounder value
   - expected margin over the opponent
5. Ranks candidates using a balanced score centered on:
   - win probability
   - expected margin
   - outcome volatility
   - steal risk / urgency

This is intentionally better than a naive “pick the highest total rating” approach. For example, the engine avoids overvaluing weak threshold bowlers when stronger all-rounders or true impact players provide a better overall path to winning.

## Tech stack

- **Python 3.11+**
- **Streamlit** for the UI
- **pandas** for tabular display helpers
- **pytest** for tests
- **ruff** for linting
- **uv** for environment and dependency management

## Project structure

```text
cricket_draft_assistant/
├── __init__.py
├── __main__.py
├── cli.py
├── core.py
└── web.py
tests/
├── test_core.py
pyproject.toml
uv.lock
sample_players.csv
```

### Important files

- `cricket_draft_assistant/core.py` — draft models, scoring, Monte Carlo engine, and serialization
- `cricket_draft_assistant/web.py` — Streamlit UI and recommendation display
- `cricket_draft_assistant/cli.py` — package-native launcher used by the CLI entrypoint
- `tests/test_core.py` — focused regression coverage for the draft engine
- `pyproject.toml` — project metadata, dependencies, scripts, and tool configuration

## Requirements

Before running the app, make sure you have:

- Python 3.11 or newer
- `uv` installed

If `uv` is not installed yet, see the official `uv` installation docs for your platform.

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/bimal1988/cricket-draft-assistant.git
cd cricket-draft-assistant
```

### 2. Create the environment and install dependencies

```bash
uv sync --dev
```

This creates the local `.venv` and installs both runtime and development dependencies.

### 3. Launch the app

```bash
uv run cricket-draft-assistant
```

This starts the Streamlit app using the package entrypoint.

## Using the app

### Start a new draft

1. Launch the app.
2. Upload a player CSV.
3. Enter your captain and the opponent captain ratings.
4. Choose who picks first.
5. Select simulation count and rating uncertainty.
6. Start the draft.

### During the draft

- When it is **your turn**, the app shows ranked recommendations.
- When it is the **opponent’s turn**, record the opponent’s selection.
- Use **Undo last pick** if needed.
- Use **Download save file** to export the current draft state.

### Finish and review

When the draft ends, the app shows:

- projected score for each team
- projected margin
- bowling plans for each side
- final downloadable JSON state

## Input CSV format

The app expects a CSV with three columns in this order:

```text
Player Name, Batting Strength, Bowling Strength
```

Example:

```csv
Player Name, Batting Strength, Bowling Strength
A,5,4
B,3,5
C,2.5,4.5
```

### CSV notes

- The app ignores invalid rows that cannot be parsed into a name and two numeric values.
- Duplicate player names are rejected.
- Captains are removed from the draft pool automatically if they appear in the uploaded CSV.
- After removing captains, the remaining player count must be even so both captains receive the same number of picks.

## Recommendation metrics explained

### Win probability

Estimated chance that your final drafted team finishes stronger than the opponent across the simulations.

### Expected margin

Average projected score difference between your team and the opponent after simulating the rest of the draft.

### Risk (σ)

Standard deviation of simulated margin. Bigger values mean the outcome is more volatile.

### Steal risk

Estimated probability that the opponent would take that player before the draft comes back around.

### ± Win

Approximate confidence interval width around the simulated win probability.

### Close call

Flags candidates that are statistically close enough to the best option that you should use team fit and steal risk as tiebreakers.

## Draft model details

The scoring model is designed for short-format team construction and includes:

- weighted batting value with lineup depth decay
- bowling contribution based on actual overs capacity
- penalties for uncovered overs
- bonuses for reliable bowling coverage
- bonuses for all-rounders
- batting depth and bowling variety adjustments

This helps the engine prefer teams that are actually playable over teams that simply look good on paper.

## Save and restore

The app can export the full draft state as JSON.

That save file includes:

- captains
- current picks for both sides
- full player pool
- pick order information
- pick history
- team size and draft configuration

You can restore a saved draft from the setup screen.

## Development

### Run tests

```bash
uv run pytest
```

### Run lint checks

```bash
uv run ruff check .
```

### Recommended validation flow

```bash
uv sync --dev
uv run ruff check .
uv run pytest
uv run cricket-draft-assistant
```

## Current test coverage focus

The tests in `tests/test_core.py` currently verify:

- duplicate player name rejection
- draft state round-tripping and undo behavior
- deterministic engine behavior for the same seed
- bowling coverage prioritization when one pick remains
- balanced ranking so impact all-rounders outrank weak threshold-only options
- adaptive shortlist refinement and close-call metadata

## Design principles

- Keep draft logic centralized in `cricket_draft_assistant/core.py`.
- Keep the Streamlit UI in `cricket_draft_assistant/web.py`.
- Keep app launch/bootstrap code only in `cricket_draft_assistant/cli.py` and `cricket_draft_assistant/__main__.py`.
- Avoid duplicate recommendation logic across files.
- Prefer model quality based on win probability and opponent-relative margin.
- Preserve deterministic simulation behavior through explicit seeds.
- Keep recommendations fresh by invalidating cache inputs when draft state or simulation settings change.

## Known assumptions

- The current draft flow assumes two captains alternating picks from a shared pool.
- Ratings are expected on roughly a 0 to 5 scale.
- Team evaluation is heuristic and simulation-driven, not based on ball-by-ball match simulation.
- The UI is optimized for local use through Streamlit.

## Future improvement ideas

- Import/export multiple draft scenarios
- Custom opponent persona weights
- Different match formats and overs rules
- Ball-by-ball or innings-level simulation
- Player tags such as wicketkeeper, opener, death bowler, spinner, pace bowler
- Historical analytics comparing recommended vs actual picks

## Troubleshooting

### Blank page when launching

This app should be launched through:

```bash
uv run cricket-draft-assistant
```

The package entrypoint starts Streamlit with the correct app module.

### Recommendations look stale

Recommendation caching is keyed off:

- serialized `DraftState`
- simulation count
- noise setting
- refresh nonce

Changing draft state or simulation settings should invalidate cached results immediately.

### Duplicate player error

Make sure every player name in the CSV is unique.

## License

Apache License, Version 2.0, (LICENSE-APACHE or https://www.apache.org/licenses/LICENSE-2.0)
