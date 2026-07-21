"""Pipeline checks for the earthquake-rate playground.

Author: James Edward Ball
"""

from datetime import datetime, timedelta, timezone
import unittest

import numpy as np
import torch

from quakecast.data import Event, events_to_sample, grid_arrays
from quakecast.model import SmaAtUNet


class PipelineTest(unittest.TestCase):
    def test_model_matches_paper_size_and_grid(self) -> None:
        model = SmaAtUNet()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 4_032_205)
        self.assertEqual(model(torch.randn(2, 3, 20, 20)).shape, (2, 1, 20, 20))

    def test_catalogue_window_and_filters(self) -> None:
        trigger_time = datetime(2025, 1, 8, tzinfo=timezone.utc)
        trigger = Event(trigger_time, 35.0, 140.0, 10.0, 4.5)
        events = [
            Event(trigger_time - timedelta(days=1), 35.05, 140.05, 8.0, 2.2),
            Event(trigger_time - timedelta(hours=1), 35.05, 140.05, 45.0, 3.0),
            trigger,
            Event(trigger_time + timedelta(hours=2), 35.05, 140.05, 7.0, 2.5),
        ]
        features, target = events_to_sample(events, trigger)
        self.assertEqual(features.shape, (3, 20, 20))
        self.assertEqual(target.shape, (1, 20, 20))
        self.assertAlmostEqual(float(np.expm1(target).sum()), 1.0)

    def test_grid_handles_antimeridian_and_exclusive_north_east_edges(self) -> None:
        trigger_time = datetime(2025, 1, 8, tzinfo=timezone.utc)
        seconds = trigger_time.timestamp()
        inputs, target = grid_arrays(
            np.asarray([seconds, seconds + 1, seconds + 2, seconds + 3]),
            np.asarray([0.0, 0.0, 1.0, -1.0]),
            np.asarray([179.8, -179.8, 179.8, 178.8]),
            np.asarray([5.0, 6.0, 7.0, 8.0]),
            np.asarray([4.2, 2.1, 2.2, 2.3]),
            trigger_time,
            0.0,
            179.8,
        )
        self.assertEqual(int(np.expm1(inputs[0]).round().sum()), 1)
        self.assertEqual(int(target.sum()), 2)

    def test_trigger_is_input_and_post_trigger_event_is_target(self) -> None:
        trigger_time = datetime(2025, 1, 8, tzinfo=timezone.utc)
        seconds = trigger_time.timestamp()
        inputs, target = grid_arrays(
            np.asarray([seconds, seconds + 0.001]),
            np.asarray([0.0, 0.0]),
            np.asarray([0.0, 0.0]),
            np.asarray([5.0, 5.0]),
            np.asarray([4.0, 2.0]),
            trigger_time,
            0.0,
            0.0,
        )
        self.assertEqual(int(np.expm1(inputs[0]).round().sum()), 1)
        self.assertEqual(int(target.sum()), 1)


if __name__ == "__main__":
    unittest.main()
