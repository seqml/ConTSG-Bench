"""
Submission & Version Tab — submission guidelines and version history.

Provides documentation on how to submit results to the leaderboard,
displays the version history, and shows citation information.
"""

from __future__ import annotations

import gradio as gr
import pandas as pd

from src.data_loader import SnapshotData


SUBMISSION_GUIDE = """
## How to Submit Results

### Overview

ConTSG-Bench accepts evaluation results from any time series generation model.
Submit a single YAML file via **GitHub Pull Request** — no package installation required.

### Step 1: Create a Submission File

Create a YAML file named `submissions/<your_model>.yaml`:

```yaml
model:
  name: my_model
  model_type: diffusion       # diffusion | flow | vae | gan | retrieval | other
  org: "My Lab"
  paper_link: "https://arxiv.org/abs/..."     # optional
  code_link: "https://github.com/..."          # source code repo
  model_link: "https://huggingface.co/..."     # checkpoint repo (recommended)
  reproducibility:
    script_link: "https://github.com/.../scripts/reproduce_contsg.sh"
    command: "bash scripts/reproduce_contsg.sh --dataset synth-u --seed 0"
    commit: "0123456789abcdef0123456789abcdef01234567"
  params: "45M"               # optional

results:
  - dataset: synth-m
    condition_modality: text   # text | attribute | label
    n_runs: 3                  # number of random seeds
    metrics:
      dtw:  { mean: 1.23, std: 0.05 }
      fid:  { mean: 12.34, std: 1.2 }
      cttp: { mean: 0.87, std: 0.02 }
      # include any subset of the 15 benchmark metrics

  - dataset: ettm1
    condition_modality: text
    n_runs: 3
    metrics:
      dtw:  { mean: 2.34, std: 0.08 }
      # ...
```

### Step 2: Validate Locally (Optional)

```bash
pip install -e .
python -m contsg.leaderboard.validate_submission submissions/my_model.yaml
```

### Step 3: Open a Pull Request

Push your YAML file and open a PR to the repository.
CI will automatically validate the submission format.
Once merged, the leaderboard updates.

### Open-Source Metadata

- `code_link` is used by the leaderboard to:
  - mark `Replication Code` as **Yes/No**
  - support the **Open-source only** ranking filter
  - render `Model (code)` hyperlink in the ranking table
- `model_link` is used to render **HF Weights** hyperlink when checkpoints are public
- reproducibility templates: `templates/repro_submission/`

### Benchmark Datasets (10)

| Dataset | Domain | Semantic Level |
|---------|--------|----------------|
| `synth-m` | synthetic | morphological |
| `synth-u` | synthetic | morphological |
| `ettm1` | energy | morphological |
| `weather_concept` | weather | conceptual |
| `weather_morphology` | weather | morphological |
| `telecomts_segment` | telecom | morphological |
| `istanbul_traffic` | traffic | morphological |
| `airquality_beijing` | environment | morphological |
| `ptbxl_concept` | health | conceptual |
| `ptbxl_morphology` | health | morphological |

### Metric Groups (3)

| Group | Metrics | Direction | In Ranking? |
|-------|---------|-----------|-------------|
| **Fidelity** | ACD, SD, KD, MDD, FID, PRDC Precision, PRDC Recall | Mixed | Yes |
| **Adherence** | JFTSD, Joint PRDC Precision, Joint PRDC Recall, CTTP | Mixed | Yes |
| **Utility** | DTW, CRPS, ED, WAPE | Lower better | No (tracked only) |

Overall ranking is based on Fidelity and Adherence scores only.
Utility metrics are displayed but do not affect rankings.
You do not need to report all metrics — the leaderboard will show coverage
as the fraction of metrics you reported.
"""


def create_submission_tab(data: SnapshotData) -> gr.Blocks:
    """Create the Submission & Version tab component."""

    with gr.Blocks() as tab:
        gr.Markdown(SUBMISSION_GUIDE)

        gr.Markdown("---")
        gr.Markdown("## Version History")

        # Build version history table
        versions = data.version_manifest.get("versions", [])
        if versions:
            version_rows = []
            for v in versions:
                version_rows.append(
                    {
                        "Version": v.get("version", ""),
                        "Release Date": v.get("release_date", ""),
                        "Models": v.get("n_models", 0),
                        "Datasets": v.get("n_datasets", 0),
                        "Metrics": v.get("n_metrics", 0),
                        "Changelog": v.get("changelog", ""),
                    }
                )
            version_df = pd.DataFrame(version_rows)
            gr.Dataframe(
                value=version_df,
                label="Version History",
                interactive=False,
                wrap=True,
            )
        else:
            gr.Markdown("*No version history available.*")

        gr.Markdown("---")
        gr.Markdown(
            """
## Citation

If you use ConTSG-Bench in your research, please cite:

Paper: https://arxiv.org/abs/2603.04767

```bibtex
@article{contsgbench2026,
  title={ConTSG-Bench: A Unified Benchmark for Conditional Time Series Generation},
  author={Shaocheng Lan and Shuqi Gu and Zhangzhi Xiong and Kan Ren},
  journal={arXiv preprint arXiv:2603.04767},
  year={2026}
}
```
"""
        )

    return tab
