# ConTSG-Bench: A Unified Benchmark for Conditional Time Series Generation

[![arXiv](https://img.shields.io/badge/arXiv-2506.XXXXX-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2506.XXXXX)
[![HuggingFace Space](https://img.shields.io/badge/HuggingFace-Space-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/spaces/mldi-lab/ConTSG-Bench-Leaderboard)
[![HuggingFace Dataset](https://img.shields.io/badge/HuggingFace-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/mldi-lab/ConTSG-Bench-Datasets)
[![HuggingFace Checkpoints](https://img.shields.io/badge/HuggingFace-Checkpoints-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/mldi-lab/ConTSG-Bench-Checkpoints)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Conditional time series generation with varying conditioning modalities (text, attribute, class label) and semantic abstraction levels (morphological vs. conceptual).*

---

> **Call for Models & Datasets** -- ConTSG-Bench is an **open and evolving** benchmark. We welcome the community to submit new models and datasets to the leaderboard. See the **[Submission Guide](docs/submission_guide.md)** for details.
> Reproducibility template files are available in **[`templates/repro_submission/`](templates/repro_submission/README.md)**.

> Benchmark scope (datasets/models/metrics) is maintained in **[docs/benchmark_spec.md](docs/benchmark_spec.md)**.

---

## Overview

**Conditional time series generation (ConTSG)** has emerged as a transformative capability for scientific and industrial advancement. Its applications span from realistic data simulation for healthcare and climate analysis, causal inference, to privacy-preserving data synthesis. While unconditional generation has seen significant progress with established benchmarks, the research frontier has shifted toward *controllable* synthesis: generating high-fidelity time series that strictly adheres to user-defined, multimodal conditions.

However, the landscape of ConTSG remains **highly fragmented**. Current methods are isolated by their specific conditioning modalities — some rely on discrete class labels, others on structured attributes, and recent works explore natural language descriptions. These models are typically evaluated on incompatible datasets with different condition modalities, making it infeasible to systematically compare conditional generation effectiveness.

**ConTSG-Bench** addresses this critical gap by providing the **first unified benchmark** for conditional time series generation. Our benchmark systematically disentangles condition types along two axes:

- **Modality**: *class label*, *attribute*, and *text*
- **Semantic abstraction**: *morphological* (describing observable temporal structures) vs. *conceptual* (describing high-level domain semantics)

<p align="center">
  <img src="assets/figure_task_condition.png" width="88%" alt="Condition modality and semantic abstraction overview in ConTSG-Bench."/>
</p>

<p align="center">
  <em>ConTSG-Bench unifies condition modalities (label, attribute, text) and semantic abstraction levels (morphological, conceptual) in a single benchmark setting.</em>
</p>

**Key features:**

- **Single CLI Entry Point**: All operations via `contsg` command
- **Registry Pattern**: Easy model/dataset extension via decorators
- **PyTorch Lightning**: Standardized training with built-in best practices
- **Experiment Tracking**: Git commit tracking, config snapshots, checkpoint management

## Datasets

ConTSG-Bench comprises **10 benchmark datasets** spanning diverse domains, with aligned
conditions across text / attribute / label modalities:


| Dataset ID             | Domain            | Variates | Seq Length | Semantic Level |
| ---------------------- | ----------------- | -------- | ---------- | -------------- |
| `synth-u`              | Synthetic         | 1        | 128        | Morphological  |
| `synth-m`              | Synthetic         | 2        | 128        | Morphological  |
| `ettm1`                | Energy            | 1        | 120        | Morphological  |
| `istanbul_traffic`     | Transportation    | 1        | 144        | Morphological  |
| `airquality_beijing`   | Environment       | 6        | 24         | Morphological  |
| `telecomts_segment`    | Network Telemetry | 2        | 128        | Morphological  |
| `ptbxl_morphology`     | Healthcare (ECG)  | 12       | 1000       | Morphological  |
| `ptbxl_concept`        | Healthcare (ECG)  | 12       | 1000       | Conceptual     |
| `weather_morphology`   | Meteorology       | 10       | 36         | Morphological  |
| `weather_concept`      | Meteorology       | 10       | 36         | Conceptual     |


Each dataset provides **aligned multimodal conditions**: text descriptions, structured attributes, and class labels are derived from the same underlying semantics, enabling controlled cross-modality comparison.

Dataset release: **[ConTSG-Bench Dataset on Hugging Face](https://huggingface.co/datasets/mldi-lab/ConTSG-Bench-Datasets)**

Checkpoint release (current public scope: `synth-u`, `synth-m`): **[ConTSG-Bench Checkpoints on Hugging Face](https://huggingface.co/mldi-lab/ConTSG-Bench-Checkpoints)**

## Supported Models

ConTSG-Bench benchmarks **11 representative generation models** spanning all three conditioning modalities:

### Text-Conditioned Models


| Model           | Registry ID   | Backbone                    | Reference                                                                 |
| --------------- | ------------- | --------------------------- | ------------------------------------------------------------------------- |
| **VerbalTS**    | `verbalts`    | Patch Transformer + DDPM    | [Gu et al., ICML 2025](https://icml.cc/virtual/2025/poster/45631)          |
| **T2S**         | `t2s`         | Transformer + Flow Matching | [Ge et al., IJCAI 2025](https://arxiv.org/abs/2505.02417)                 |
| **BRIDGE**      | `bridge`      | UNet + DDPM                 | [Li et al., ICML 2025](https://arxiv.org/abs/2503.02445)                  |
| **DiffuSETS**   | `diffusets`   | VAE + Latent DDPM           | [Lai et al., Patterns 2025](https://doi.org/10.1016/j.patter.2025.101291) |
| **Text2Motion** | `text2motion` | Conv AE + GRU-VAE           | [Guo et al., CVPR 2022](https://doi.org/10.1109/CVPR52688.2022.00509)     |
| **Retrieval**   | `retrieval`   | Nearest Neighbor (Baseline) | —                                                                         |


### Attribute-Conditioned Models


| Model          | Registry ID  | Backbone                 | Reference                                                                |
| -------------- | ------------ | ------------------------ | ------------------------------------------------------------------------ |
| **TimeWeaver** | `timeweaver` | Transformer + DDPM       | [Narasimhan et al., ICML 2024](https://arxiv.org/abs/2403.02682)         |
| **WaveStitch** | `wavestitch` | S4 + DDPM                | [Shankar et al., PACM 2025](https://doi.org/10.1145/3769842)             |
| **TEdit**      | `tedit`      | Patch Transformer + DDPM | [Jing et al., NeurIPS 2024](https://openreview.net/forum?id=qu5NTwZtxA)  |


### Label-Conditioned Models


| Model         | Registry ID | Backbone                    | Reference                                                    |
| ------------- | ----------- | --------------------------- | ------------------------------------------------------------ |
| **TimeVQVAE** | `timevqvae` | VQ-VAE + Masked Transformer | [Lee et al., AISTATS 2023](https://arxiv.org/abs/2303.04743) |
| **TTS-CGAN**  | `ttscgan`   | Transformer + GAN           | [Li et al., 2022](https://arxiv.org/abs/2206.13676)          |


## Key Results

<p align="center">
  <img src="assets/model_ranking.png" width="90%" alt="Model ranking across generation fidelity and condition adherence."/>
</p>

*Model ranking under two metric groups: generation fidelity (left) and condition adherence (right). Text-conditioned models offer the highest performance ceiling but also the largest variance.*

**Key findings from our benchmark:**

- **Good generation fidelity does not guarantee condition adherence.** Some models perform consistently well on both dimensions, while others show significant rank differences, confirming the need to evaluate these two aspects separately.
- **Text conditioning offers the highest performance ceiling but also the largest variance.** Text-conditioned models span the full range from top to bottom, whereas attribute-conditioned methods cluster in the upper-middle tier.

**Explore the full results on our interactive leaderboard:** **[ConTSG-Bench Leaderboard](https://huggingface.co/spaces/mldi-lab/ConTSG-Bench-Leaderboard)**

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/contsg.git
cd contsg

# Install full runtime + development tools (recommended)
uv sync --extra full --extra dev

# Alternative: install with pip
pip install -e ".[full,dev]"
```

If you only need the core package APIs (without full benchmark dependencies), use
`pip install -e .`.

## Quick Start

### Training a Model

```bash
# Fast smoke test (no dataset files required)
contsg train -d debug -m verbalts --smoke

# Basic training with dataset and model specification (requires dataset + checkpoints)
contsg train --dataset synth-m --model verbalts

# With custom parameters
contsg train -d synth-m -m verbalts --epochs 1000 --lr 5e-4

# From configuration file
contsg train --config configs/generators/verbalts_ettm1.yaml

# Resume training from a checkpoint
contsg train -d synth-m -m verbalts --resume experiments/exp1/
```

For benchmark config placeholders (`<CTTP_CONFIG>`, `<CTTP_CHECKPOINT>`), see
`configs/README.md`.

### Evaluating a Model

```bash
# Evaluate best checkpoint
contsg evaluate experiments/20250101_synth-m_verbalts/

# Specify checkpoint and metrics
contsg evaluate experiments/exp1/ --checkpoint last --metrics dtw,fid
```

### Listing Available Resources

```bash
contsg list-models        # List all registered models
contsg list-datasets      # List all registered datasets
contsg list-experiments   # List experiments
contsg info experiments/exp1/  # Show experiment details
```

## Extending the Benchmark

ConTSG-Bench uses a decorator-based registry — models and datasets are auto-discovered at runtime. See the **[full extending guide](docs/extending_guide.md)** for detailed documentation including configuration reference, multi-stage training, and complete examples.

### Adding a New Model

Subclass `BaseGeneratorModule` and implement three methods: `_build_model()`, `forward()`, and `generate()`.

```python
# contsg/models/my_model.py
from contsg.models.base import BaseGeneratorModule
from contsg.registry import Registry

@Registry.register_model("my_model")
class MyModelModule(BaseGeneratorModule):
    """My custom generation model."""

    def _build_model(self):
        cfg = self.config.model
        data_cfg = self.config.data
        self.encoder = nn.Linear(cfg.channels, data_cfg.n_var * data_cfg.seq_length)
        self.decoder = nn.Linear(cfg.channels, data_cfg.n_var * data_cfg.seq_length)

    def forward(self, batch):
        ts = batch["ts"]           # (B, L, C) — time series
        cap_emb = batch["cap_emb"] # (B, D)   — text embedding
        # ... compute loss ...
        return {"loss": loss}      # must return dict with "loss" key

    def generate(self, condition, n_samples=1, **kwargs):
        # condition: (B, D) — conditioning tensor
        # return: (B, n_samples, L, C)
        return samples
```

Then use it immediately:

```bash
contsg train -d synth-m -m my_model
contsg train -d debug -m my_model --smoke  # quick validation, no data files needed
```

Model-specific schema is optional. If you want strict validation for your custom model, register with
`@Registry.register_model("my_model", config_class=MyModelConfig)`, then run with `--strict-schema`.
Without `config_class`, your model still works in default relaxed mode.

### Adding a New Dataset

For standard file format (`{split}_ts.npy`, `{split}_cap_emb.npy`, etc.), registration requires no custom code:

```python
# contsg/data/datasets/my_dataset.py
from contsg.data.datamodule import BaseDataModule
from contsg.registry import Registry

@Registry.register_dataset("my_dataset")
class MyDataModule(BaseDataModule):
    """My custom dataset."""
    pass  # uses default TimeSeriesDataset loader
```

Place data files in `datasets/my_dataset/` following this structure:

```
datasets/my_dataset/
├── meta.json              # Dataset metadata
├── train_ts.npy           # (N, L, C) time series
├── train_cap_emb.npy      # (N, D) text embeddings
├── valid_ts.npy, valid_cap_emb.npy
└── test_ts.npy,  test_cap_emb.npy
```

For non-standard formats, override `_create_dataset()` — see the [extending guide](docs/extending_guide.md#32-custom-dataset) for details.

## Project Structure

```text
contsg/
├── contsg/                     # Main package
│   ├── cli.py                  # CLI entry point (train / evaluate / generate)
│   ├── registry.py             # Model / dataset / metric registration
│   ├── tracker.py              # Experiment tracking
│   ├── config/                 # Pydantic configuration system
│   ├── models/                 # 11 generation models
│   ├── data/                   # Data handling & text embedding precomputation
│   ├── train/                  # Training utilities & multi-stage support
│   └── eval/                   # Evaluation system (15 leaderboard metrics + auxiliary metrics)
├── configs/                    # YAML configuration files
│   ├── cttp/                   # CTTP contrastive model configs
│   └── generators/             # Generator model configs
├── datasets/                   # Dataset files (gitignored)
└── experiments/                # Experiment outputs (gitignored)
```

## Development & Quality Checks

```bash
# Run tests
uv run --extra dev pytest -v

# Lint and type checks
uv run --extra dev ruff check .
uv run --extra dev black --check .
uv run --extra dev mypy contsg
```

CI runs tests and markdown link checks on pull requests.

## Community

- Contributing guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- Security policy: [SECURITY.md](SECURITY.md)
- Changelog: [CHANGELOG.md](CHANGELOG.md)

## Contact

- Shaocheng Lan: lanshch2024@shanghaitech.edu.cn
- Shuqi Gu: gushq@shanghaitech.edu.cn
- Zhangzhi Xiong: xiongzhzh2023@shanghaitech.edu.cn

## Citation

If you find ConTSG-Bench useful in your research, please cite our paper:

```bibtex
@article{contsgbench2025,
  title={ConTSG-Bench: A Unified Benchmark for Conditional Time Series Generation},
  author={Shaocheng Lan and Shuqi Gu and Zhangzhi Xiong and Kan Ren},
  journal={arXiv preprint},
  year={2025}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
