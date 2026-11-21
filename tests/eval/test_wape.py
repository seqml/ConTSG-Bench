import math
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from contsg.config.schema import ExperimentConfig
from contsg.eval import Evaluator
from contsg.eval.metrics.basic import WAPEMetric


@pytest.mark.parametrize(
    ("padded_prediction", "padded_actual"),
    [(99.0, 0.0), (-1e6, 42.0), (float("nan"), float("inf"))],
)
def test_perfect_valid_prediction_ignores_padding(padded_prediction, padded_actual):
    metric = WAPEMetric()
    metric.update(
        {
            "pred": torch.tensor([[[2.0], [padded_prediction]]]),
            "ts": torch.tensor([[[2.0], [padded_actual]]]),
            "ts_len": torch.tensor([1]),
        }
    )

    assert metric.compute() == 0.0


def test_pools_valid_errors_and_actuals_across_batches_and_features():
    metric = WAPEMetric()
    first = {
        "pred": torch.tensor(
            [[[4.0, -6.0], [-10.0, 100.0]], [[1.0, -2.0], [5.0, 9.0]], [[2.0, -3.0], [40.0, 50.0]]]
        ),
        "ts": torch.tensor(
            [[[2.0, -6.0], [40.0, 30.0]], [[1.0, -2.0], [4.0, 8.0]], [[50.0, 90.0], [90.0, 800.0]]]
        ),
        "ts_len": torch.tensor([1, 2, 0]),
    }
    before = {key: value.clone() for key, value in first.items()}
    metric.update(first)
    assert metric.compute() == pytest.approx(100 * 4 / 23)

    metric.update(
        {
            "pred": torch.tensor([[[-14.0, 9.0], [7.0, -8.0]]]),
            "ts": torch.tensor([[[-10.0, 7.0], [4.0, -8.0]]]),
            "ts_len": torch.tensor([2]),
        }
    )

    # Valid entries contribute errors of 4 and 9, and actual magnitudes of 23 and 29.
    assert metric.total_absolute_error == pytest.approx(13 / 2)
    assert metric.total_actual_sum == pytest.approx(52 / 2)
    assert metric.compute() == pytest.approx(100 * 13 / 52)
    for key, value in before.items():
        torch.testing.assert_close(first[key], value)


@pytest.mark.parametrize("supply_lengths", [False, True])
def test_full_length_score_and_reset(supply_lengths):
    batch = {
        "pred": torch.tensor([[[3.0], [10.0]]]),
        "ts": torch.tensor([[[2.0], [8.0]]]),
    }
    if supply_lengths:
        batch["ts_len"] = torch.tensor([2])
    metric = WAPEMetric()
    metric.update(batch)
    metric.update(batch)

    assert metric.total_absolute_error == 6.0
    assert metric.total_actual_sum == 20.0
    assert metric.compute() == pytest.approx(30.0)
    metric.reset()
    assert metric.total_absolute_error == metric.total_actual_sum == 0.0
    assert math.isnan(metric.compute())


def test_empty_batch_keeps_wape_undefined():
    metric = WAPEMetric()
    metric.update(
        {
            "pred": torch.empty(0, 3, 2),
            "ts": torch.empty(0, 3, 2),
            "ts_len": torch.empty(0, dtype=torch.long),
        }
    )

    assert metric.total_absolute_error == metric.total_actual_sum == 0.0
    assert math.isnan(metric.compute())


@pytest.mark.parametrize("length", [0, 1])
def test_zero_valid_actual_sum_keeps_wape_undefined(length):
    metric = WAPEMetric()
    metric.update(
        {
            "pred": torch.tensor([[[1.0], [20.0]]]),
            "ts": torch.tensor([[[0.0], [10.0]]]),
            "ts_len": torch.tensor([length]),
        }
    )

    assert metric.total_actual_sum == 0.0
    assert math.isnan(metric.compute())


class FixedPredictionModel(torch.nn.Module):
    use_condition = False

    def generate(self, condition, n_samples, **kwargs):
        return torch.tensor([[[2.0], [99.0]], [[1.0], [5.0]]])


@pytest.mark.parametrize(
    ("lengths", "expected"),
    [
        ([1, 2], {"wape": 100 * 2 / 6, "ed": 1.0}),
        (None, {"wape": 100 * 101 / 6, "ed": 50.5}),
    ],
)
def test_evaluator_uses_sequence_lengths_in_fresh_and_cached_runs(tmp_path, lengths, expected):
    samples = [{"ts": torch.tensor([[2.0], [0.0]])}, {"ts": torch.tensor([[1.0], [3.0]])}]
    if lengths is not None:
        for sample, length in zip(samples, lengths, strict=True):
            sample["ts_len"] = torch.tensor(length)
    datamodule = SimpleNamespace(test_dataloader=lambda: DataLoader(samples, batch_size=2))
    config = ExperimentConfig(
        data={"name": "debug", "data_folder": tmp_path, "n_var": 1, "seq_length": 2},
        model={"name": "verbalts"},
        eval={"metrics": ["wape", "ed"], "n_samples": 1},
    )
    cache_path = tmp_path / "predictions.pkl"
    evaluator = Evaluator(FixedPredictionModel(), config, datamodule, device=torch.device("cpu"))
    fresh = evaluator.evaluate(use_cache=True, cache_path=cache_path)
    cached_evaluator = Evaluator(
        None, config, datamodule, device=torch.device("cpu"), cache_only_mode=True
    )
    cached = cached_evaluator.evaluate(use_cache=True, cache_path=cache_path)

    assert fresh["ed"] == cached["ed"] == expected["ed"]
    assert fresh == pytest.approx(expected)
    assert cached == pytest.approx(expected)
