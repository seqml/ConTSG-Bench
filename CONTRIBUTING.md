# Contributing to ConTSG-Bench

Thanks for your interest in contributing.

## Development Setup

```bash
git clone https://github.com/seqml/ConTSG-Bench.git
cd ConTSG-Bench
uv sync --extra full --extra dev
```

## Pull Request Workflow

1. Fork the repository and create a feature branch.
2. Keep changes focused and small when possible.
3. Run checks locally before opening a PR.
4. Open a PR with a clear description of motivation and changes.

## Local Checks

```bash
uv run --extra dev pytest -v
uv run --extra dev ruff check .
uv run --extra dev black --check .
```

## Documentation Update Rule

If your PR changes user-facing behavior, you must update at least one relevant document:

- `README.md`, or
- a file in `docs/`.

Examples of user-facing changes:

- CLI flags / command behavior
- config keys / defaults
- dataset or model registration names
- metric definitions / leaderboard rules

## Model Submission

If you are submitting benchmark results, follow `docs/submission_guide.md`.
