# Configuration Files

This directory contains configuration files for training and evaluation.

## Directory Structure

```
configs/
├── cttp/           # CTTP (Contrastive Text-Time Series Pre-training) model configs
├── generators/     # Generator model configs (Bridge, DiffuSETS, T2S, etc.)
└── README.md
```

## Placeholder Paths

Some configuration files contain placeholder paths that need to be replaced with your actual paths before use:

### 1. `pretrain_model_path` (in CTTP configs)

```yaml
pretrain_model_path: ./checkpoints/longclip
```

This should point to your local LongCLIP model directory. For the public ConTSG-Bench release, use the exact upstream Hugging Face model `zer0int/LongCLIP-GmP-ViT-L-14` and place the full HF model directory (for example `config.json`, tokenizer files, and `model.safetensors`) under `./checkpoints/longclip/`. Do **not** substitute the original `BeichenZhang/LongCLIP-L`; the released CTTP resources were checked against the GmP fine-tuned LongCLIP variant.

The public checkpoint release does **not** mirror LongCLIP weights inside `ConTSG-Bench-Checkpoints`. LongCLIP remains an external upstream dependency and should be downloaded separately.

Recommended download command (creates the exact directory expected by the public CTTP configs):

```bash
mkdir -p ./checkpoints
git lfs install
git clone https://huggingface.co/zer0int/LongCLIP-GmP-ViT-L-14 ./checkpoints/longclip
```

If a released CTTP runtime config contains `${LONGCLIP_ROOT}`, replace it with the absolute path of your local LongCLIP directory, for example `$(pwd)/checkpoints/longclip`.

### 2. `clip_config_path` (in Generator configs)

```yaml
clip_config_path: ./configs/cttp/cttp_<dataset>_<mode>.yaml
```

Points to the corresponding CTTP config file. Some configs already provide a concrete path,
while others keep `<CTTP_CONFIG>` as an explicit placeholder.

### Released CTTP Resources vs. Local CTTP Templates

`configs/cttp/*.yaml` in this repository are public CTTP **training templates**. The checkpoint release on Hugging Face may additionally provide runtime files such as `resources/cttp/<dataset>/model_configs.yaml`. Those released runtime configs are the files used during generator evaluation and may differ textually from the local templates, for example by replacing machine-specific LongCLIP paths with the portable placeholder `${LONGCLIP_ROOT}`. That difference is expected; what must stay consistent is the CTTP checkpoint plus the exact LongCLIP model family.

For the current public release scope (`synth-m`, `synth-u`), you can download the released CTTP runtime files directly into local checkpoint directories with:

```bash
mkdir -p ./checkpoints/cttp/synth-m ./checkpoints/cttp/synth-u

curl -L https://huggingface.co/mldi-lab/ConTSG-Bench-Checkpoints/resolve/main/resources/cttp/synth-m/model_configs.yaml \
  -o ./checkpoints/cttp/synth-m/model_configs.yaml
curl -L https://huggingface.co/mldi-lab/ConTSG-Bench-Checkpoints/resolve/main/resources/cttp/synth-m/clip_model_best.pth \
  -o ./checkpoints/cttp/synth-m/clip_model_best.pth

curl -L https://huggingface.co/mldi-lab/ConTSG-Bench-Checkpoints/resolve/main/resources/cttp/synth-u/model_configs.yaml \
  -o ./checkpoints/cttp/synth-u/model_configs.yaml
curl -L https://huggingface.co/mldi-lab/ConTSG-Bench-Checkpoints/resolve/main/resources/cttp/synth-u/clip_model_best.pth \
  -o ./checkpoints/cttp/synth-u/clip_model_best.pth
```

Then replace `${LONGCLIP_ROOT}` in the released runtime configs with your local LongCLIP directory, for example:

```bash
LONGCLIP_ROOT="$(pwd)/checkpoints/longclip"
sed -i "s|\${LONGCLIP_ROOT}|${LONGCLIP_ROOT}|g" ./checkpoints/cttp/synth-m/model_configs.yaml
sed -i "s|\${LONGCLIP_ROOT}|${LONGCLIP_ROOT}|g" ./checkpoints/cttp/synth-u/model_configs.yaml
```

A generator config that uses the released `synth-m` CTTP files should then point to:

```yaml
clip_config_path: ./checkpoints/cttp/synth-m/model_configs.yaml
clip_model_path: ./checkpoints/cttp/synth-m/clip_model_best.pth
```

### 3. `clip_model_path` (in Generator configs)

```yaml
clip_model_path: ./checkpoints/cttp/<CTTP_CHECKPOINT>.ckpt
```

**Replace `<CTTP_CHECKPOINT>` with your trained CTTP model checkpoint filename.**

Example:
```yaml
clip_model_path: ./checkpoints/cttp/cttp_ettm1_instance_best.ckpt
```

## Dataset Naming Convention

| Dataset Name | Description |
|-------------|-------------|
| `ettm1` | ETT-m1 time series dataset |
| `istanbul_traffic` | Istanbul traffic flow dataset |
| `ptbxl_concept` | PTB-XL ECG dataset (concept-level descriptions) |
| `ptbxl_morphology` | PTB-XL ECG dataset (morphology-level descriptions) |
| `weather_concept` | Weather dataset (concept-level descriptions) |
| `weather_morphology` | Weather dataset (morphology-level descriptions) |
| `telecomts_segment` | Telecom time series dataset (segment-level) |
| `airquality_beijing` | Beijing air quality dataset |

CTTP naming convention: use `cttp_<dataset>.yaml` by default. Only `telecomts_segment`
keeps two explicit modes: `cttp_telecomts_instance.yaml` and
`cttp_telecomts_segment.yaml`.

## Quick Start

1. **Run a smoke test first (no data files needed):**

   ```bash
   contsg train -d debug -m verbalts --smoke
   ```

2. **Prepare datasets**: Place your datasets in `./datasets/<dataset_name>/`

3. **Download LongCLIP**: Download the exact upstream Hugging Face model `zer0int/LongCLIP-GmP-ViT-L-14` to `./checkpoints/longclip/`. If a released CTTP `model_configs.yaml` uses `${LONGCLIP_ROOT}`, replace that placeholder with your local LongCLIP directory.

   ```bash
   mkdir -p ./checkpoints
   git lfs install
   git clone https://huggingface.co/zer0int/LongCLIP-GmP-ViT-L-14 ./checkpoints/longclip
   ```

4. **Train CTTP model**:
   ```bash
   contsg train --config configs/cttp/cttp_ettm1.yaml
   ```

5. **Update generator config**:
   - Replace `clip_model_path: ./checkpoints/cttp/<CTTP_CHECKPOINT>.ckpt`
   - If present, replace `clip_config_path: ./configs/cttp/<CTTP_CONFIG>.yaml`

6. **Train generator model**:
   ```bash
   contsg train --config configs/generators/bridge_ettm1.yaml
   ```
