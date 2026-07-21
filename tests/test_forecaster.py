"""Checks for the rate head, its persistence offset, and training utilities.

Author: James Edward Ball
"""

import math
import unittest

import torch

from quakecast.forecaster import (
    WRAPPER_NAME,
    ExponentialMovingAverage,
    RateForecaster,
    build_model,
    cosine_schedule,
)
from quakecast.model import SmaAtUNet


class ForecasterTest(unittest.TestCase):
    def test_plain_head_matches_the_bare_network(self) -> None:
        model = RateForecaster()
        self.assertEqual(model(torch.randn(2, 3, 20, 20)).shape, (2, 1, 20, 20))
        # One extra scalar over the paper size: the learnable quiet-cell floor.
        self.assertEqual(sum(p.numel() for p in model.parameters()), 4_032_206)

    def test_plain_head_keeps_a_spatially_varying_output_layer(self) -> None:
        # Zeroing the head without an offset starts the forecast flat and it
        # stays diffuse, so only the offset variant may zero it.
        model = RateForecaster(output_bias=-1.1)
        self.assertTrue(bool(model.core.output.weight.abs().sum() > 0))
        self.assertAlmostEqual(float(model.core.output.bias.mean()), -1.1, places=5)

    def test_offset_head_starts_at_the_persistence_rate(self) -> None:
        model = RateForecaster(baseline_offset=True, output_bias=0.0, floor=1e-6).eval()
        weekly_counts = torch.randint(0, 30, (2, 1, 20, 20)).float()
        features = torch.cat((torch.log1p(weekly_counts), torch.zeros(2, 2, 20, 20)), dim=1)
        with torch.no_grad():
            rate = model(features).exp()
        # A zeroed output layer plus a zero bias leaves exactly the seven-day mean.
        self.assertTrue(torch.allclose(rate, weekly_counts / 7.0, atol=1e-4))

    def test_offset_head_stays_finite_in_empty_cells(self) -> None:
        model = RateForecaster(baseline_offset=True)
        prediction = model(torch.zeros(2, 3, 20, 20))
        self.assertTrue(bool(torch.isfinite(prediction).all()))

    def test_metadata_round_trip_rebuilds_the_same_module(self) -> None:
        for offset in (False, True):
            model = RateForecaster(width=32, baseline_offset=offset)
            metadata = {"model_wrapper": WRAPPER_NAME, "width": 32, "baseline_offset": offset}
            rebuilt = build_model(metadata)
            rebuilt.load_state_dict(model.state_dict())
            self.assertEqual(rebuilt.baseline_offset, offset)

    def test_checkpoints_without_a_wrapper_flag_load_the_bare_network(self) -> None:
        self.assertIsInstance(build_model({}), SmaAtUNet)

    def test_average_tracks_the_live_weights(self) -> None:
        model = RateForecaster(width=8)
        averager = ExponentialMovingAverage(model, decay=0.9)
        with torch.no_grad():
            model.core.output.bias.fill_(5.0)
        for _ in range(200):
            averager.update(model)
        self.assertAlmostEqual(float(averager.shadow.core.output.bias.mean()), 5.0, places=3)

    def test_cosine_schedule_warms_up_then_decays(self) -> None:
        self.assertLess(cosine_schedule(0, 100, 10), 1.0)
        self.assertAlmostEqual(cosine_schedule(9, 100, 10), 1.0)
        self.assertAlmostEqual(cosine_schedule(99, 100, 10), 0.01, places=3)
        self.assertAlmostEqual(cosine_schedule(500, 100, 10), 0.01, places=3)

    def test_width_scales_the_parameter_count(self) -> None:
        narrow = sum(p.numel() for p in SmaAtUNet(width=32).parameters())
        wide = sum(p.numel() for p in SmaAtUNet(width=64).parameters())
        self.assertLess(narrow, wide)
        self.assertEqual(SmaAtUNet(width=32)(torch.randn(2, 3, 20, 20)).shape, (2, 1, 20, 20))


class SamplerTest(unittest.TestCase):
    def test_capped_sampler_draws_distinct_examples(self) -> None:
        import numpy as np

        from train_real import CappedComponentSampler

        class Stub:
            component_ids = np.array(["a", "a", "a", "b", "b", "c"])
            indexes = np.arange(6)

        sampler = CappedComponentSampler(Stub(), cap=2, seed=7)
        drawn = list(sampler)
        self.assertEqual(len(drawn), len(sampler))
        self.assertEqual(len(drawn), len(set(drawn)))
        # Two from "a", two from "b", one from "c".
        self.assertEqual(len(drawn), 5)


if __name__ == "__main__":
    unittest.main()
