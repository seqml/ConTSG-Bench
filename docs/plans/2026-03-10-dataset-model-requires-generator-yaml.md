# Dataset Model YAML Resolution Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `contsg train --dataset <dataset> --model <model>` require and load `configs/generators/<model>_<dataset>.yaml` instead of silently building a default config.

**Architecture:** Move the dataset+model resolution rule into `ConfigLoader.from_args()` so every caller gets the same behavior. Keep the CLI path mostly unchanged, but surface a clear error when the required generator YAML is missing.

**Tech Stack:** Python, Typer, PyYAML, pytest

---

### Task 1: Lock Behavior With Loader Tests

**Files:**
- Modify: `tests/config/test_loader_paths.py`

**Step 1: Write the failing tests**

- Add a test asserting `from_args(dataset="synth-m", model="verbalts")` loads values from `configs/generators/verbalts_synth-m.yaml`.
- Add a test asserting `from_args(...)` raises when `configs/generators/<model>_<dataset>.yaml` is missing.

**Step 2: Run test to verify it fails**

Run: `pytest tests/config/test_loader_paths.py -q`

**Step 3: Implement minimal loader change**

- Resolve `configs/generators/<model>_<dataset>.yaml`
- Load that YAML as the base experiment config
- Apply CLI overrides on top
- Raise a clear `FileNotFoundError` if the file is absent

**Step 4: Run test to verify it passes**

Run: `pytest tests/config/test_loader_paths.py -q`

### Task 2: Make CLI Error Output Clear

**Files:**
- Modify: `contsg/cli.py`

**Step 1: Add minimal handling**

- Catch the loader’s missing-YAML error in the train command
- Print a concise message telling the user which config file is required
- Exit with status 1

**Step 2: Run targeted verification**

Run: `pytest tests/config/test_loader_paths.py -q`

### Task 3: Final Verification

**Files:**
- Modify: `contsg/config/loader.py`
- Modify: `contsg/cli.py`
- Modify: `tests/config/test_loader_paths.py`

**Step 1: Run verification**

Run: `pytest tests/config/test_loader_paths.py -q`

**Step 2: Inspect behavior manually**

Run: `python - <<'PY' ... ConfigLoader(...).from_args(...) ... PY`

