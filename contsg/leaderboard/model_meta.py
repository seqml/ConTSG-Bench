"""
Model metadata registry for the ConTSG-Bench leaderboard.

Maps each benchmark model to its type and metadata for the model cards table.
Fields like ``org``, ``paper_link``, ``code_link``, and ``params`` capture
leaderboard metadata for built-in models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class ModelMeta:
    """Metadata for a single benchmark model."""

    name: str
    model_type: str  # diffusion, flow, vae, gan, retrieval
    native_condition: str = "text"  # text, attribute, label
    org: str = "ConTSG"
    model_link: str = ""
    code_link: str = ""
    paper_link: str = ""
    params: str = ""  # e.g. "45M"
    training_data_compliance: bool = True
    testdata_leakage: bool = False
    replication_code_available: bool = True
    notes: str = ""


# ---------------------------------------------------------------------------
# Canonical model metadata (11 generator models)
# ---------------------------------------------------------------------------

MODEL_META: Dict[str, ModelMeta] = {
    "verbalts": ModelMeta(
        name="verbalts",
        model_type="diffusion",
        native_condition="text",
        notes="Multi-scale patch diffusion with AdaLN conditioning.",
    ),
    "t2s": ModelMeta(
        name="t2s",
        model_type="flow",
        native_condition="text",
        notes="Flow matching with patch transformer (two-stage).",
    ),
    "bridge": ModelMeta(
        name="bridge",
        model_type="diffusion",
        native_condition="text",
        notes="Bridge matching for text-conditioned generation.",
    ),
    "diffusets": ModelMeta(
        name="diffusets",
        model_type="diffusion",
        native_condition="text",
        notes="VAE + latent diffusion (two-stage).",
    ),
    "timeweaver": ModelMeta(
        name="timeweaver",
        model_type="diffusion",
        native_condition="attribute",
        notes="GRU + diffusion with attribute conditioning.",
    ),
    "wavestitch": ModelMeta(
        name="wavestitch",
        model_type="diffusion",
        native_condition="attribute",
        notes="S4 + diffusion with attribute conditioning.",
    ),
    "tedit": ModelMeta(
        name="tedit",
        model_type="diffusion",
        native_condition="attribute",
        notes="Multi-scale patch diffusion with attribute conditioning.",
    ),
    "timevqvae": ModelMeta(
        name="timevqvae",
        model_type="vae",
        native_condition="label",
        notes="VQ-VAE + MaskGIT (two-stage, label-conditioned).",
    ),
    "ttscgan": ModelMeta(
        name="ttscgan",
        model_type="gan",
        native_condition="label",
        notes="Transformer GAN for time series generation.",
    ),
    "text2motion": ModelMeta(
        name="text2motion",
        model_type="vae",
        native_condition="text",
        notes="CVPR'22 Text-to-Motion adapted for time series (two-stage, Conv AE + GRU-VAE).",
    ),
    "retrieval": ModelMeta(
        name="retrieval",
        model_type="retrieval",
        native_condition="text",
        notes="Nearest-neighbor retrieval baseline (no training required).",
    ),
}

# Non-generator model (not included in leaderboard rankings)
AUXILIARY_MODELS: Dict[str, ModelMeta] = {
    "cttp": ModelMeta(
        name="cttp",
        model_type="contrastive",
        notes="Contrastive text-time series pre-training model (used for CTTP metric).",
    ),
}

BENCHMARK_MODEL_NAMES = frozenset(MODEL_META.keys())


def get_model_meta(name: str) -> ModelMeta:
    """Look up model metadata by name.

    Raises:
        KeyError: If the model name is not registered.
    """
    if name not in MODEL_META:
        raise KeyError(
            f"Unknown model '{name}'. "
            f"Registered models: {sorted(MODEL_META.keys())}"
        )
    return MODEL_META[name]
