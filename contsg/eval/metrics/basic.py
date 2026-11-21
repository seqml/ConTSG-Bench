"""
Basic evaluation metrics (no CLIP dependency).

This module implements fundamental time series generation metrics that don't
require CLIP embeddings, including DTW, WAPE, Euclidean Distance, and CRPS.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import torch
from torch import Tensor

from contsg.eval.metrics.base import AccumulativeMetric, BaseMetric, CollectiveMetric
from contsg.registry import Registry

logger = logging.getLogger(__name__)

# Import DTW with graceful fallback
try:
    from tslearn.metrics import dtw as tslearn_dtw
    _HAS_TSLEARN = True
except ImportError:
    _HAS_TSLEARN = False
    logger.warning("tslearn not installed. DTWMetric will be unavailable.")


@Registry.register_metric("dtw")
class DTWMetric(CollectiveMetric):
    """
    Dynamic Time Warping distance metric.

    Computes DTW distance between generated and ground truth time series,
    averaged over all samples. DTW is robust to temporal misalignment.

    Legacy source: metrics.py:334-369
    Requires: tslearn library
    """

    def __init__(self, name: str = "dtw"):
        """
        Initialize DTW metric.

        Args:
            name: Metric identifier
        """
        if not _HAS_TSLEARN:
            raise ImportError(
                "DTWMetric requires tslearn. Install with: pip install tslearn"
            )
        super().__init__(name)

    def update(self, batch_data: Dict[str, Any]) -> None:
        """
        Collect batch predictions and ground truth.

        Args:
            batch_data: Must contain 'pred' (B, L, F) and 'ts' (B, L, F)
        """
        pred: Tensor = batch_data["pred"]
        ts: Tensor = batch_data["ts"]

        self.collected_data.append({
            "pred": pred.cpu().numpy(),
            "ts": ts.cpu().numpy(),
        })

    def compute(self) -> float:
        """
        Compute mean DTW distance over all collected samples.

        Returns:
            Mean DTW distance, or NaN if no data
        """
        if not self.collected_data:
            return float("nan")

        total_dtw = 0.0
        total_samples = 0

        for batch_data in self.collected_data:
            pred = batch_data["pred"]  # (B, L, F)
            ts = batch_data["ts"]  # (B, L, F)
            batch_size = pred.shape[0]

            for i in range(batch_size):
                try:
                    # tslearn expects shape (seq_len, n_features)
                    dtw_score = tslearn_dtw(ts[i], pred[i])
                    total_dtw += dtw_score
                    total_samples += 1
                except Exception as e:
                    logger.warning(f"DTW calculation failed for sample {i}: {e}")
                    continue

        return total_dtw / total_samples if total_samples > 0 else float("nan")


@Registry.register_metric("wape")
class WAPEMetric(BaseMetric):
    """
    Weighted Absolute Percentage Error metric.

    WAPE = sum(|actual - predicted|) / sum(|actual|) * 100

    More robust than MAPE for values near zero. Computed per-feature
    and averaged across features.

    Legacy source: metrics.py:371-419
    """

    def __init__(self, name: str = "wape"):
        """
        Initialize WAPE metric.

        Args:
            name: Metric identifier
        """
        super().__init__(name)
        self.total_absolute_error: float = 0.0
        self.total_actual_sum: float = 0.0

    def reset(self) -> None:
        """Reset accumulators."""
        self.total_absolute_error = 0.0
        self.total_actual_sum = 0.0

    def update(self, batch_data: Dict[str, Any]) -> None:
        """
        Accumulate WAPE components from batch.

        Args:
            batch_data: Contains 'pred' and 'ts' (B, L, F) and optionally 'ts_len' (B,)
        """
        pred: Tensor = batch_data["pred"]
        ts: Tensor = batch_data["ts"]

        pred_np = pred.cpu().numpy()  # (B, L, F)
        ts_np = ts.cpu().numpy()  # (B, L, F)

        lengths = batch_data.get("ts_len")
        if lengths is not None:
            valid = np.arange(ts_np.shape[1])[None, :] < lengths.cpu().numpy()[:, None]
            pred_np, ts_np = pred_np[valid], ts_np[valid]

        n_features = pred_np.shape[-1]
        feature_absolute_errors = []
        feature_actual_sums = []

        for feature_idx in range(n_features):
            pred_feature = pred_np[..., feature_idx]
            ts_feature = ts_np[..., feature_idx]

            # Sum of absolute differences
            absolute_diff = np.abs(ts_feature - pred_feature)
            absolute_error = np.sum(absolute_diff)

            # Sum of absolute actual values
            actual_sum = np.sum(np.abs(ts_feature))

            feature_absolute_errors.append(absolute_error)
            feature_actual_sums.append(actual_sum)

        # Average across features
        avg_absolute_error = np.mean(feature_absolute_errors)
        avg_actual_sum = np.mean(feature_actual_sums)

        self.total_absolute_error += avg_absolute_error
        self.total_actual_sum += avg_actual_sum

    def compute(self) -> float:
        """
        Compute WAPE percentage.

        Returns:
            WAPE as percentage, or NaN if denominator is zero
        """
        if self.total_actual_sum == 0.0:
            return float("nan")

        return (self.total_absolute_error / self.total_actual_sum) * 100


@Registry.register_metric("ed")
class EuclideanDistanceMetric(AccumulativeMetric):
    """
    Euclidean Distance metric with masking for variable lengths.

    For each sample, computes sqrt(sum_{t,f} (pred - ts)^2) within the valid
    prefix defined by ts_len. Final result is averaged over all samples.

    Legacy source: metrics.py:808-829
    """

    def __init__(self, name: str = "ed"):
        """
        Initialize Euclidean Distance metric.

        Args:
            name: Metric identifier
        """
        super().__init__(name)

    def update(self, batch_data: Dict[str, Any]) -> None:
        """
        Accumulate Euclidean distances from batch.

        Args:
            batch_data: Must contain 'pred' (B, L, F), 'ts' (B, L, F), 'ts_len' (B,)
        """
        pred: Tensor = batch_data["pred"]  # (B, L, F)
        ts: Tensor = batch_data["ts"]  # (B, L, F)
        ts_len: Tensor = batch_data["ts_len"]  # (B,)

        device = pred.device
        B, L, F = pred.shape

        # Create mask for valid time steps
        t = torch.arange(L, device=device)[None, :]  # (1, L)
        mask = (t < ts_len[:, None]).unsqueeze(-1).float()  # (B, L, 1)

        # Compute squared differences with masking
        diff2 = (pred - ts) ** 2
        s = (diff2 * mask).sum(dim=(1, 2))  # (B,)

        # Euclidean distance per sample
        d = torch.sqrt(torch.clamp(s, min=0.0))

        self.total_value += d.detach().float().sum().item()
        self.total_samples += int(B)


@Registry.register_metric("crps")
class CRPSMetric(CollectiveMetric):
    """
    Continuous Ranked Probability Score metric.

    CRPS = E[|forecast - observation|] - 0.5 * E[|forecast - forecast'|]

    A proper scoring rule for probabilistic forecasts that measures both
    calibration and sharpness. Requires multiple prediction samples.

    Legacy source: metrics.py:178-254
    """

    def __init__(self, name: str = "crps"):
        """
        Initialize CRPS metric.

        Args:
            name: Metric identifier
        """
        super().__init__(name)

    def update(self, batch_data: Dict[str, Any]) -> None:
        """
        Collect multi-sample predictions and ground truth.

        Args:
            batch_data: Must contain:
                - 'multi_preds': (n_samples, B, L, F) multiple forecast samples
                - 'ts': (B, L, F) ground truth
        """
        multi_preds: Tensor = batch_data["multi_preds"]
        ts: Tensor = batch_data["ts"]

        self.collected_data.append({
            "multi_preds": multi_preds.cpu().numpy(),
            "ts": ts.cpu().numpy(),
        })

    def compute(self) -> float:
        """
        Compute mean CRPS over all collected samples.

        Returns:
            Mean CRPS, or NaN if no data
        """
        if not self.collected_data:
            return float("nan")

        total_crps = 0.0
        total_samples = 0

        for batch_data in self.collected_data:
            multi_preds = batch_data["multi_preds"]  # (n_samples, B, L, F)
            ts = batch_data["ts"]  # (B, L, F)

            batch_size = ts.shape[0]
            n_samples = multi_preds.shape[0]

            for i in range(batch_size):
                # Get predictions and ground truth for this sample
                forecasts = multi_preds[:, i, :, :]  # (n_samples, L, F)
                observation = ts[i, :, :]  # (L, F)

                # Flatten for CRPS computation
                forecasts_flat = forecasts.reshape(n_samples, -1)  # (n_samples, L*F)
                observation_flat = observation.flatten()  # (L*F,)

                # Compute CRPS for this sample
                crps_sample = self._compute_crps_sample(forecasts_flat, observation_flat)
                total_crps += crps_sample
                total_samples += 1

        return total_crps / total_samples if total_samples > 0 else float("nan")

    def _compute_crps_sample(
        self, forecasts: np.ndarray, observation: np.ndarray
    ) -> float:
        """
        Compute CRPS for a single sample.

        Args:
            forecasts: (n_samples, n_dims) forecast ensemble
            observation: (n_dims,) observed values

        Returns:
            CRPS value for this sample
        """
        n_samples, n_dims = forecasts.shape

        # Term 1: E[|forecast - observation|]
        abs_diff_obs = np.abs(forecasts - observation[None, :])  # (n_samples, n_dims)
        term1 = np.mean(abs_diff_obs)

        # Term 2: E[|forecast - forecast'|]
        abs_diff_forecast = 0.0
        count = 0
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                abs_diff_forecast += np.mean(np.abs(forecasts[i] - forecasts[j]))
                count += 1

        term2 = abs_diff_forecast / count if count > 0 else 0.0

        # CRPS = E[|forecast - observation|] - 0.5 * E[|forecast - forecast'|]
        crps = term1 - 0.5 * term2

        return crps


__all__ = [
    "DTWMetric",
    "WAPEMetric",
    "EuclideanDistanceMetric",
    "CRPSMetric",
]
