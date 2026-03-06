# ConTSG-Bench Submission Guide

This guide explains how to submit your model and evaluation results to the ConTSG-Bench leaderboard.

For canonical benchmark scope (dataset/model/metric counts), see
`docs/benchmark_spec.md`.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Benchmark Datasets](#2-benchmark-datasets)
3. [Evaluation Metrics](#3-evaluation-metrics)
4. [Submission File Format](#4-submission-file-format)
5. [Step-by-Step Submission Process](#5-step-by-step-submission-process)
6. [Running Evaluation with ConTSG](#6-running-evaluation-with-contsg)
7. [Submission Validation](#7-submission-validation)
8. [How Ranking Works](#8-how-ranking-works)

---

## 1. Overview

ConTSG-Bench is an open benchmark for **Conditional Time Series Generation**. To ensure reproducibility and comparability, all submissions must be evaluated using the ConTSG framework.

**Submission workflow:**

1. Evaluate your model on one or more benchmark datasets.
2. Create a YAML file with your results.
3. Validate your YAML locally with `validate_submission`.
4. Open a GitHub Pull Request adding the file to the `submissions/` directory.
5. Maintainers review and merge valid submissions.
6. Maintainers regenerate and validate leaderboard snapshots before publishing updates.

---

## 2. Benchmark Datasets

The benchmark consists of **10 datasets** spanning 6 domains and 2 semantic levels:

| Dataset ID | Domain | Semantic Level | Variables | Seq Length |
|---|---|---|---|---|
| `synth-m` | Synthetic | Morphological | 2 | 128 |
| `synth-u` | Synthetic | Morphological | 1 | 128 |
| `ettm1` | Energy | Morphological | 1 | 120 |
| `weather_concept` | Weather | Conceptual | 10 | 36 |
| `weather_morphology` | Weather | Morphological | 10 | 36 |
| `telecomts_segment` | Telecom | Morphological | 2 | 128 |
| `istanbul_traffic` | Traffic | Morphological | 1 | 144 |
| `airquality_beijing` | Environment | Morphological | 6 | 24 |
| `ptbxl_concept` | Health | Conceptual | 12 | 1000 |
| `ptbxl_morphology` | Health | Morphological | 12 | 1000 |

**Notes:**
- All datasets provide aligned conditions across all three modalities (text, attribute, label). Models using any condition type can be evaluated and compared on every dataset.
- You do not need to evaluate on all 10 datasets. Submit results for whichever datasets you have.
- The `condition_modality` field in your submission should match the condition type used during training/generation (text, attribute, or label).

---

## 3. Evaluation Metrics

The leaderboard tracks **15 metrics** across 3 groups:

### 3.1 Fidelity (7 metrics)

Measures how realistic the generated time series are.

| Metric ID | Display Name | Direction |
|---|---|---|
| `acd` | ACD | Lower is better |
| `sd` | SD | Lower is better |
| `kd` | KD | Lower is better |
| `mdd` | MDD | Lower is better |
| `fid` | FID | Lower is better |
| `prdc_f1.precision` | kNN-PRF Precision | Higher is better |
| `prdc_f1.recall` | kNN-PRF Recall | Higher is better |

### 3.2 Adherence (4 metrics)

Measures how well the generated time series follow the given condition.

| Metric ID | Display Name | Direction |
|---|---|---|
| `jftsd` | JFTSD | Lower is better |
| `joint_prdc_f1.precision` | Joint kNN-PRF Precision | Higher is better |
| `joint_prdc_f1.recall` | Joint kNN-PRF Recall | Higher is better |
| `cttp` | CTTP | Higher is better |

### 3.3 Utility (4 metrics)

Measures practical quality via sample-level distances.

| Metric ID | Display Name | Direction |
|---|---|---|
| `dtw` | DTW | Lower is better |
| `crps` | CRPS | Lower is better |
| `ed` | ED | Lower is better |
| `wape` | WAPE | Lower is better |

**Notes:**
- You do not need to report all 15 metrics. The leaderboard displays a "coverage" score showing what fraction of metrics you reported.
- Metric IDs use dot notation for nested metrics (e.g., `prdc_f1.precision`, not `prdc_f1_precision`).

---

## 4. Submission File Format

Each submission is a single YAML file placed in the `submissions/` directory, named `<your_model_name>.yaml`.

### 4.1 Full Schema

```yaml
model:
  name: my_model                    # Required. Unique model identifier.
  model_type: diffusion             # Required. One of: diffusion, flow, vae, gan, retrieval, other
  org: "My Lab"                     # Optional. Organization or author name.
  paper_link: "https://arxiv.org/abs/2401.xxxxx"  # Optional.
  code_link: "https://github.com/user/repo"       # Recommended. Source code repository.
  model_link: "https://huggingface.co/user/model" # Recommended. HF checkpoint repository.
  ckpt_scope: ["synth-u", "synth-m"]              # Optional. Public checkpoint coverage.
  reproducibility:                                # Recommended for reproducible leaderboard entries.
    script_link: "https://github.com/user/repo/blob/main/scripts/reproduce_contsg.sh"
    command: "bash scripts/reproduce_contsg.sh --dataset synth-u --seed 0"
    commit: "0123456789abcdef0123456789abcdef01234567"
    notes: "One-click script to reproduce reported numbers from checkpoints."
  params: "45M"                     # Optional. Number of trainable parameters.
  notes: "Brief description"        # Optional. One-line model description.

results:
  - dataset: synth-m                # Required. Must be one of the 10 benchmark dataset IDs.
    condition_modality: text         # Required. One of: text, attribute, label.
    n_runs: 3                        # Required. Number of random seeds used (>=1).
    metrics:
      dtw:  { mean: 1.234, std: 0.056 }
      fid:  { mean: 12.34, std: 1.20 }
      cttp: { mean: 0.87,  std: 0.02 }
      # ... add any subset of the 15 benchmark metrics

  - dataset: ettm1
    condition_modality: text
    n_runs: 3
    metrics:
      dtw:  { mean: 2.345, std: 0.089 }
      fid:  { mean: 8.76,  std: 0.95 }
      # ...
```

### 4.2 Field Details

#### `model` section

| Field | Required | Description |
|---|---|---|
| `name` | Yes | Unique identifier for your model. Use lowercase with no spaces (e.g., `my_model`). |
| `model_type` | Yes | Architecture family: `diffusion`, `flow`, `vae`, `gan`, `retrieval`, or `other`. |
| `org` | No | Your organization or team name. |
| `paper_link` | No | URL to the paper (arXiv, conference, etc.). |
| `code_link` | Recommended | URL to the source code repository. Needed for `Replication Code` and `Open-source only` filter on the leaderboard. |
| `model_link` | Recommended | URL to model weights/checkpoints (prefer Hugging Face model repo). Used for `HF Weights` display. |
| `ckpt_scope` | No | List of dataset IDs for which checkpoints are publicly available (e.g., `["synth-u","synth-m"]`). |
| `reproducibility.script_link` | Recommended | URL to one-click reproduction script in your code repository. |
| `reproducibility.command` | Recommended | Exact command to reproduce reported metrics. |
| `reproducibility.commit` | Recommended | Git commit hash corresponding to reported numbers. |
| `reproducibility.notes` | No | Extra details for maintainers/reviewers. |
| `params` | No | Number of parameters as a human-readable string (e.g., `"45M"`, `"1.2B"`). |
| `notes` | No | A brief one-line description of the model. |

#### `results` section (list)

Each entry describes results on one dataset:

| Field | Required | Description |
|---|---|---|
| `dataset` | Yes | One of the 10 benchmark dataset IDs (see [Section 2](#2-benchmark-datasets)). |
| `condition_modality` | Yes | The condition type used: `text`, `attribute`, or `label`. |
| `n_runs` | Yes | Number of independent runs (random seeds) that were averaged. Must be >= 1. |
| `metrics` | Yes | A mapping of metric IDs to `{mean, std}` values. |

#### `metrics` values

Each metric is specified as:

```yaml
metric_name:
  mean: 1.234      # Required. The mean value across runs.
  std: 0.056       # Optional (default: 0.0). Standard deviation across runs.
```

- `mean` must be a finite number (no NaN, no Infinity).
- `std` must be >= 0.
- If you only ran one seed, set `std: 0.0` (or omit it).

### 4.3 Minimal Example

A minimal valid submission with only 1 dataset and 2 metrics:

```yaml
model:
  name: my_baseline
  model_type: other

results:
  - dataset: synth-m
    condition_modality: text
    n_runs: 1
    metrics:
      dtw: { mean: 5.67 }
      fid: { mean: 45.2 }
```

---

## 5. Step-by-Step Submission Process

### Step 1: Fork the repository

```bash
git clone https://github.com/<your-user-or-org>/ConTSG-Bench.git
cd ConTSG-Bench
git checkout -b add-my-model-submission
```

### Step 2: Create your submission file

```bash
# Create the file
vim submissions/my_model.yaml
# (paste your YAML content)
```

### Step 3: Validate locally (optional but recommended)

```bash
pip install -e .
python -m contsg.leaderboard.validate_submission submissions/my_model.yaml
```

Expected output for a valid file:

```
INFO: Submission 'my_model': 3 datasets, 48 recognized metrics
Validation PASSED (0 warnings)
```

### Step 4: Commit and push

```bash
git add submissions/my_model.yaml
git commit -m "Add my_model results to leaderboard"
git push origin add-my-model-submission
```

### Step 5: Open a Pull Request

Open a PR on GitHub. Maintainers will review your submission and run validation/aggregation steps before publishing leaderboard updates.

### Open-source display behavior

The leaderboard uses `model.code_link` to:

- show `Model (code)` hyperlink in the ranking table
- populate the `Replication Code` column (`Yes` when `code_link` is provided)
- support the `Open-source only` ranking filter

The leaderboard also uses `model.model_link` to:

- show `HF Weights` hyperlink for public checkpoints (when provided)

### Reproducibility requirements (new submissions)

For new submissions, we strongly recommend including:

- checkpoint link(s) (`model_link`)
- one-click reproduction script (`reproducibility.script_link`)
- exact reproduction command (`reproducibility.command`)
- pinned code revision (`reproducibility.commit`)

Template files are provided in:

- `templates/repro_submission/submission.example.yaml`
- `templates/repro_submission/reproduce_contsg.sh`
- `templates/repro_submission/README.md`

Validation note:

- current `validate_submission` focuses on metric/schema integrity
- reproducibility fields above are reviewed by maintainers during PR review

Current official ConTSG checkpoint release scope is:

- `synth-u`
- `synth-m`

Official checkpoint repository:

- https://huggingface.co/mldi-lab/ConTSG-Bench-Checkpoints

---

## 6. Running Evaluation with ConTSG

If you are using the ConTSG framework, you can generate results directly:

### Training

```bash
pip install -e .
contsg train -d synth-m -m my_model --seed 0
contsg train -d synth-m -m my_model --seed 1
contsg train -d synth-m -m my_model --seed 2
```

### Evaluation

```bash
contsg evaluate experiments/<run_dir_seed0>
contsg evaluate experiments/<run_dir_seed1>
contsg evaluate experiments/<run_dir_seed2>
```

Each evaluation produces `results/eval_results.json` containing all 15 metrics. You can then aggregate the 3 seeds into a submission YAML manually or programmatically.

---

## 7. Submission Validation

### Local validation

```bash
python -m contsg.leaderboard.validate_submission submissions/my_model.yaml
```

The validator checks:
- YAML syntax is correct.
- All required fields are present (`model.name`, `model.model_type`, `results`, etc.).
- Dataset names are valid benchmark datasets.
- Metric values are finite numbers.
- Standard deviations are non-negative.

**Errors** will block the submission. **Warnings** (e.g., unknown metric names) are informational and will not block.

### Maintainer post-merge workflow (manual)

After approved submissions are merged, maintainers should regenerate snapshot files and validate them before releasing leaderboard updates:

```bash
# 1) Aggregate submissions/*.yaml into snapshot files
python -m contsg.leaderboard.aggregate submissions ./snapshots --version vYYYY.MM.DD

# 2) Validate generated snapshot files
python -m contsg.leaderboard.validate ./snapshots
```

If snapshot validation fails, do not publish the update until all errors are resolved.

### Common warnings

| Warning | Meaning |
|---|---|
| `Unknown metric 'xxx'` | This metric is not in the benchmark catalog. It will be silently ignored. |
| `Duplicate dataset entry` | You have two entries for the same dataset. Only the first will be used. |

---

## 8. How Ranking Works

The leaderboard uses a **percentile normalization + weighted aggregation** ranking system. Only **Fidelity** and **Adherence** groups participate in the overall ranking. Utility metrics are tracked but do not affect rankings.

### Step 1: Percentile Normalization

For each metric within each **dataset**, compute the percentile rank across **all models regardless of condition modality** (text, attribute, label). Because ConTSG-Bench provides aligned conditions across all three modalities for each dataset, models using different condition types are directly comparable on the same data.

- For "lower is better" metrics: `norm_score = 1 - percentile_rank`
- For "higher is better" metrics: `norm_score = percentile_rank`
- Result: `norm_score` in [0, 1], where **higher is always better**.

### Step 2: Group Aggregation

For each model, compute the average `norm_score` within each ranking group:
- **Fidelity** score (average of 7 fidelity norm_scores)
- **Adherence** score (average of 4 adherence norm_scores)

Utility metrics (DTW, CRPS, ED, WAPE) are excluded from ranking.

### Step 3: Overall Score

Three ranking policies are available:

| Policy | Description |
|---|---|
| **Balanced** | Average of fidelity and adherence scores |
| **Fidelity-only** | Rank solely by fidelity score |
| **Adherence-only** | Rank solely by adherence score |

Under the default "Balanced" policy, the overall score is the arithmetic mean: `(fidelity + adherence) / 2`.

### Coverage

The leaderboard shows **coverage** — the fraction of the 15 benchmark metrics that you reported. Models with higher coverage have more robust rankings.
