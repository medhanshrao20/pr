"""Method checks and mocked selection checks; execute only when explicitly run."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch

from decomposition import decompose
from model import KANInformer, ProbAttention
from run import ALPHA_GRID, Config, clean_data, feature_matrix, select_features, select_vmd_parameters, split_masks


class MethodChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.config = Config(d_model=8, n_heads=2, kan_hidden=4, ewt_modes=3)
        rng = np.random.default_rng(1)
        self.x = 2 + np.sin(np.arange(256) * 0.3) + rng.normal(0, 0.1, 256)

    def test_odd_length_and_ewt_synthesis(self):
        parts, report = decompose(self.x[:255], 3, self.config, fixed_high=[1, 2])
        self.assertEqual(parts.shape, (255, 4))
        self.assertTrue(np.isfinite(parts).all())
        self.assertLess(report["ewt_synthesis_max_error"], 1e-8)

    def test_full_size_paper_and_author_forward_backward(self):
        for architecture in ["paper", "author"]:
            config = Config(architecture=architecture)
            model = KANInformer(17, config, torch.device("cpu"))
            x = torch.rand(2, 7, 17)
            output = model(x)
            self.assertEqual(output.shape, (2, 17) if architecture == "author" else (2, 3))
            output.square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            with torch.no_grad():
                self.assertEqual(model.forecast(x[:1]).shape, (1, 3))

    def test_probattention_matches_published_sampling_equations(self):
        q, k, v = [torch.rand(2, 2, 32, 4) for _ in range(3)]
        attention = ProbAttention(False, factor=3)
        torch.manual_seed(17)
        actual = attention(q, k, v)
        torch.manual_seed(17)
        sample_k = min(32, 3 * int(np.ceil(np.log(32))))
        index_sample = torch.randint(32, (32, sample_k))
        sampled = k[:, :, index_sample, :]
        scores_sample = torch.matmul(q.unsqueeze(-2), sampled.transpose(-2, -1)).squeeze(-2)
        measure = scores_sample.max(-1).values - scores_sample.sum(-1) / 32
        indices = measure.topk(sample_k, sorted=False).indices
        batch, heads = torch.arange(2)[:, None, None], torch.arange(2)[None, :, None]
        reduced = q[batch, heads, indices]
        scores = reduced @ k.transpose(-2, -1) / 2
        expected = v.mean(-2, keepdim=True).expand(-1, -1, 32, -1).clone()
        expected[batch, heads, indices] = scores.softmax(-1) @ v
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_future_suffix_cannot_change_forecast_inputs(self):
        from run import VARIABLES
        rng = np.random.default_rng(18)
        frame = pd.DataFrame(rng.normal(size=(256, len(VARIABLES))), columns=VARIABLES)
        frame["WS"] = self.x
        altered = frame.copy()
        altered.iloc[220:] = 1000
        clean, _ = clean_data(frame, 180, True)
        other, _ = clean_data(altered, 180, True)
        pd.testing.assert_frame_equal(clean.iloc[:220], other.iloc[:220])
        self.assertEqual(select_features(clean.iloc[:180])[0], select_features(other.iloc[:180])[0])
        a, report = decompose(clean["WS"].iloc[:180].to_numpy(), 3, self.config)
        high = report["high_indices"]
        for data in [clean, other]:
            parts, _ = decompose(data["WS"].iloc[:220].to_numpy(), 3, self.config, fixed_high=high)
            inputs = feature_matrix(data.iloc[:220], ["WS"], parts)[-7:]
            if data is clean:
                reference = inputs
            else:
                np.testing.assert_array_equal(inputs, reference)

    def test_targets_do_not_cross_split_boundaries(self):
        origins = np.arange(7, 98)
        train, val, test = split_masks(origins, self.config, 80, 90)
        self.assertTrue(np.all(origins[train] + 2 < 80))
        self.assertTrue(np.all(origins[val] >= 80))
        self.assertTrue(np.all(origins[val] + 2 < 90))
        self.assertTrue(np.all(origins[test] >= 90))


class SelectionChecks(unittest.TestCase):
    """Mocks avoid real VMD and neural training while checking the two-stage rule."""

    def setUp(self):
        self.config = Config(vmd_k_min=2, vmd_k_max=4)
        self.frame = pd.DataFrame({"WS": np.ones(64)})

    def windows(self, frame, selected, k, config, train_end, val_end):
        x = np.full((2, 7, 1), config.vmd_alpha, dtype=float)
        raw_y = np.ones((2, 3))
        # Deliberately different scaled targets verify that the probe uses raw WS.
        block = (x, np.full((2, 3), 99.0), raw_y, np.array([7, 8]))
        return [block, block, block], None, {}, ["WS"]

    def test_all_k_are_scored_before_alpha_is_selected(self):
        rer = {25: {2: .02, 3: .01, 4: .005},
               100: {2: .02, 3: .004, 4: .009},
               500: {2: .002, 3: .002, 4: .008},
               2000: {2: .01, 3: .006, 4: .003}}
        validation_errors = {25: .2, 100: .1, 500: .3, 2000: .4}

        def modes(ws, k, config):
            result = np.zeros((k, len(ws)))
            result[0] = ws * (1 - rer[int(config.vmd_alpha)][k])
            return result

        windows = Mock(side_effect=self.windows)
        with tempfile.TemporaryDirectory() as directory, \
                patch("run.vmd_modes", side_effect=modes) as vmd, patch("run.Ridge") as ridge:
            probe = ridge.return_value
            probe.fit.return_value = probe
            probe.predict.side_effect = lambda x: np.ones((len(x), 3)) + validation_errors[int(x[0, 0])]
            k, alpha, report = select_vmd_parameters(
                self.frame, ["WS"], self.config, 48, 56, windows, Path(directory), "spring")
            self.assertEqual((k, alpha), (3, 100.0))
            self.assertEqual(vmd.call_count, len(ALPHA_GRID) * 3)
            self.assertEqual([call.args[2] for call in windows.call_args_list], [4, 3, 2, 4])
            self.assertAlmostEqual(report["selected_val_rmse"], .1)
            self.assertTrue(report["search_complete"])
            self.assertFalse(report["three_percent_used_for_selection"])
            for call in probe.fit.call_args_list:
                np.testing.assert_array_equal(call.args[1], np.ones((2, 3)))
            table = pd.read_csv(Path(directory) / "spring_k_rer_search.csv")
            self.assertEqual(len(table), 12)
            self.assertEqual(int(table["selected_pair"].sum()), 1)
            self.assertTrue((table["status"] == "ok").all())
            alpha_table = pd.read_csv(Path(directory) / "spring_alpha_validation_search.csv")
            self.assertEqual(len(alpha_table), 4)
            self.assertEqual(int(alpha_table["selected_pair"].sum()), 1)
            json.dumps(report, allow_nan=False)

    def test_exact_ties_prefer_smaller_k_then_smaller_alpha(self):
        def modes(ws, k, config):
            result = np.zeros((k, len(ws)))
            result[0] = ws * .99
            return result

        with tempfile.TemporaryDirectory() as directory, \
                patch("run.vmd_modes", side_effect=modes), patch("run.Ridge") as ridge:
            probe = ridge.return_value
            probe.fit.return_value = probe
            probe.predict.return_value = np.ones((2, 3))
            k, alpha, report = select_vmd_parameters(
                self.frame, ["WS"], self.config, 48, 56, self.windows, Path(directory), "spring")
            self.assertEqual((k, alpha), (2, 25.0))
            self.assertEqual(len(report["k_trials"]), 12)

    def test_nonfinite_candidate_aborts_instead_of_skipping_it(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch("run.vmd_modes", return_value=np.full((2, 64), np.nan)), patch("run.Ridge") as ridge:
            with self.assertRaises(FloatingPointError):
                select_vmd_parameters(self.frame, ["WS"], self.config, 48, 56,
                                      self.windows, Path(directory), "spring")
            ridge.assert_not_called()
            table = pd.read_csv(Path(directory) / "spring_k_rer_search.csv")
            self.assertEqual(table.iloc[0]["status"], "failed")
            self.assertFalse(table["selected_pair"].any())

    def test_fixed_alpha_still_searches_every_k(self):
        config = replace(self.config, tune_alpha=False, vmd_alpha=75.0)

        def modes(ws, k, config):
            result = np.zeros((k, len(ws)))
            result[0] = ws * (1 - .01 / k)
            return result

        with tempfile.TemporaryDirectory() as directory, \
                patch("run.vmd_modes", side_effect=modes) as vmd, patch("run.Ridge") as ridge:
            probe = ridge.return_value
            probe.fit.return_value = probe
            probe.predict.return_value = np.ones((2, 3))
            k, alpha, report = select_vmd_parameters(
                self.frame, ["WS"], config, 48, 56, self.windows, Path(directory), "spring")
            self.assertEqual((k, alpha), (4, 75.0))
            self.assertEqual(vmd.call_count, 3)
            self.assertEqual(report["alpha_candidates"], [75.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
