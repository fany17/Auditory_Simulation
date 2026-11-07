"""Synthetic M4/M5 contracts. Run only on server2203; no optimizer updates."""

import unittest

import torch

from m6a_public.continuity.alignment_innovation import (
    AdaptiveAlignmentDecoder,
    InnovationStateDecoder,
)
from m6a_public.continuity.common import detach_state


def clone_tree(tree):
    if isinstance(tree, torch.Tensor):
        return tree.clone()
    if isinstance(tree, dict):
        return {key: clone_tree(value) for key, value in tree.items()}
    return tree


def assert_tree_equal(case, first, second):
    if isinstance(first, torch.Tensor):
        case.assertTrue(torch.equal(first, second))
    elif isinstance(first, dict):
        case.assertEqual(set(first), set(second))
        for key in first:
            assert_tree_equal(case, first[key], second[key])
    else:
        case.assertEqual(first, second)


class AlignmentInnovationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2306)

    def models(self):
        return [
            AdaptiveAlignmentDecoder(4, 6, lags=(0, 2, 7), mode=mode)
            for mode in ("adaptive", "fixed")
        ] + [InnovationStateDecoder(4, 6, mode=mode) for mode in ("full", "prior_only", "eeg_only")]

    def test_shapes_and_exact_streaming_contract(self):
        x = torch.randn(2, 31, 4)
        for model in self.models():
            with self.subTest(model=type(model).__name__, mode=model.mode), torch.no_grad():
                full = model(x)
                state, predictions = None, []
                for start, stop in ((0, 1), (1, 9), (9, 21), (21, 31)):
                    chunk = model(x[:, start:stop], state)
                    predictions.append(chunk.prediction)
                    state = chunk.state
                self.assertEqual(tuple(full.prediction.shape), (2, 31, 1))
                self.assertTrue(torch.isfinite(full.prediction).all())
                torch.testing.assert_close(torch.cat(predictions, dim=1), full.prediction, atol=1e-6, rtol=1e-5)

    def test_future_suffix_does_not_change_prefix(self):
        x = torch.randn(2, 31, 4)
        changed = x.clone()
        changed[:, 17:] += 100
        for model in self.models():
            with self.subTest(model=type(model).__name__, mode=model.mode), torch.no_grad():
                first, second = model(x), model(changed)
                torch.testing.assert_close(first.prediction[:, :17], second.prediction[:, :17], atol=0, rtol=0)

    def test_state_not_mutated_and_detach_does_not_reset(self):
        history = torch.randn(2, 20, 4)
        current = torch.randn(2, 9, 4)
        for model in self.models():
            with self.subTest(model=type(model).__name__, mode=model.mode):
                state = model(history).state
                snapshot = clone_tree(state)
                out = model(current, state)
                assert_tree_equal(self, state, snapshot)
                detached = model(current, detach_state(state))
                torch.testing.assert_close(out.prediction, detached.prediction, atol=0, rtol=0)
                reset = model(current, None)
                again = model(current, None)
                torch.testing.assert_close(reset.prediction, again.prediction, atol=0, rtol=0)

    def test_finite_gradients_for_used_parameters_only(self):
        for model in self.models():
            with self.subTest(model=type(model).__name__, mode=model.mode):
                x = torch.randn(2, 27, 4, requires_grad=True)
                result = model(x)
                result.prediction.square().mean().backward()
                if isinstance(model, AdaptiveAlignmentDecoder):
                    excluded = ("lag_logits.", "slow_decay_logit") if model.mode == "fixed" else ()
                elif model.mode == "prior_only":
                    excluded = ("encoder.", "observation_predictor.", "gain_net.")
                elif model.mode == "eeg_only":
                    excluded = ("prior_dynamics.", "observation_predictor.", "gain_net.")
                else:
                    excluded = ()
                for name, parameter in model.named_parameters():
                    if name.startswith(excluded):
                        self.assertIsNone(parameter.grad, name)
                    else:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                if isinstance(model, InnovationStateDecoder) and model.mode == "prior_only":
                    self.assertIsNone(x.grad)
                else:
                    self.assertIsNotNone(x.grad)
                    self.assertTrue(torch.isfinite(x.grad).all())

    def test_alignment_fixed_and_adaptive_share_lag_bank(self):
        adaptive = AdaptiveAlignmentDecoder(4, 6, lags=(0, 2, 7), slow_decay=0.9)
        fixed = AdaptiveAlignmentDecoder(4, 6, lags=(0, 2, 7), mode="fixed")
        fixed.load_state_dict(adaptive.state_dict())
        x = torch.randn(2, 31, 4)
        a, f = adaptive(x), fixed(x)
        torch.testing.assert_close(f.aux["weights"], torch.full((2, 31, 3), 1 / 3))
        torch.testing.assert_close(a.aux["weights"].sum(dim=-1), torch.ones(2, 31))
        self.assertTrue((a.aux["weights"] > 0).all())
        self.assertGreater(float((a.aux["weights"] - f.aux["weights"]).abs().max().detach()), 1e-7)
        self.assertEqual(tuple(a.state["lag"].shape), tuple(f.state["lag"].shape))
        self.assertTrue(0 < float(adaptive.slow_decay.detach()) < 1)

    def test_single_zero_lag_needs_no_history_cache(self):
        model = AdaptiveAlignmentDecoder(4, 6, lags=(0,), mode="fixed")
        first = model(torch.randn(2, 1, 4))
        second = model(torch.randn(2, 2, 4), first.state)
        self.assertEqual(tuple(first.state["lag"].shape), (2, 0, 6))
        self.assertEqual(tuple(second.prediction.shape), (2, 2, 1))

    def test_prior_only_ignores_current_and_all_prefix_eeg(self):
        model = InnovationStateDecoder(4, 6, mode="prior_only")
        # Fail if this control ever accidentally invokes the EEG encoder.
        def forbidden_encoder(*args, **kwargs):
            raise AssertionError("prior_only must not execute encoder")
        model.encoder.forward = forbidden_encoder
        first, second = torch.randn(2, 39, 4), torch.randn(2, 39, 4) * 200
        a, b = model(first), model(second)
        torch.testing.assert_close(a.prediction, b.prediction, atol=0, rtol=0)
        a_prefix, b_prefix = model(first[:, :30]), model(second[:, :30])
        a_tail = model(first[:, 30:], a_prefix.state)
        b_tail = model(second[:, 30:], b_prefix.state)
        torch.testing.assert_close(a_tail.prediction, b_tail.prediction, atol=0, rtol=0)
        self.assertIsNone(a.state["encoder"])
        self.assertTrue((a.aux["innovation"] == 0).all())
        full_state = InnovationStateDecoder(4, 6, mode="full")(first[:, :30]).state
        with self.assertRaises(ValueError):
            model(first[:, 30:], full_state)

    def test_eeg_only_has_no_latent_recurrence_beyond_local_encoder(self):
        model = InnovationStateDecoder(4, 6, mode="eeg_only")
        x = torch.randn(2, 55, 4)
        modified = x.clone()
        modified[:, :10] += 100
        a, b = model(x), model(modified)
        # Two k9 causal convolutions have 17 samples of local receptive field.
        torch.testing.assert_close(a.prediction[:, 26:], b.prediction[:, 26:], atol=0, rtol=0)
        self.assertIsNone(a.state["latent"])

    def test_full_innovation_and_gain_are_explicit(self):
        model = InnovationStateDecoder(4, 6, mode="full")
        x = torch.randn(2, 19, 4)
        output = model(x)
        features, _ = model.encoder(x)
        torch.testing.assert_close(output.aux["innovation"], features - output.aux["predicted_observation"])
        self.assertTrue(((output.aux["gain"] > 0) & (output.aux["gain"] < 1)).all())
        expected_state = torch.tanh(output.aux["prior"] + output.aux["gain"] * output.aux["innovation"])
        torch.testing.assert_close(output.aux["latent"], expected_state)
        changed = model(x + 30)
        self.assertGreater(float((output.prediction - changed.prediction).abs().max().detach()), 1e-7)

    def test_invalid_arguments_and_states_fail_explicitly(self):
        for kwargs in ({"lags": ()}, {"lags": (-1, 0)}, {"lags": (0, 0)}, {"mode": "target_align"}, {"slow_decay": 1}, {"fixed_lag_weights": (-1, 1, 1, 1)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AdaptiveAlignmentDecoder(**kwargs)
        with self.assertRaises(ValueError):
            InnovationStateDecoder(mode="oracle_audio")
        with self.assertRaises(ValueError):
            InnovationStateDecoder(4, 6)(torch.zeros(2, 0, 4))
        with self.assertRaises(ValueError):
            InnovationStateDecoder(4, 6)(torch.zeros(2, 3, 4), {"latent": torch.zeros(1, 6)})


if __name__ == "__main__":
    unittest.main()
