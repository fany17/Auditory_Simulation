"""Synthetic engineering checks only; no fitting, datasets, or performance claims."""

import unittest

import torch

from m6a_public.continuity.common import CausalConv, LocalEncoder, detach_state
from m6a_public.continuity.dense_state import DenseMultiscaleDecoder, PersistentStateDecoder


def assert_tree_close(test, first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)
    elif isinstance(first, dict):
        test.assertEqual(first.keys(), second.keys())
        for key in first:
            assert_tree_close(test, first[key], second[key])
    elif isinstance(first, (tuple, list)):
        test.assertEqual(len(first), len(second))
        for value, other in zip(first, second):
            assert_tree_close(test, value, other)
    else:
        test.assertEqual(first, second)


def clone_tree(tree):
    if isinstance(tree, torch.Tensor):
        return tree.detach().clone()
    if isinstance(tree, dict):
        return {key: clone_tree(value) for key, value in tree.items()}
    return tree


class SS06BuildingBlockTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(106)

    def test_causal_convolution_arbitrary_chunks_including_large_dilation(self):
        # T crosses the whole largest kernel, so tests exercise every delayed tap.
        x = torch.randn(2, 1103, 4)
        chunks = (1, 7, 1, 83, 256, 1, 754)
        for dilation in (1, 8, 64, 256):
            with self.subTest(dilation=dilation):
                layer = CausalConv(4, 6, 5, dilation=dilation)
                full, full_cache = layer(x)
                cursor, cache, outputs = 0, None, []
                for length in chunks:
                    part, cache = layer(x[:, cursor:cursor + length, :], cache)
                    outputs.append(part)
                    cursor += length
                self.assertEqual(cursor, x.shape[1])
                torch.testing.assert_close(torch.cat(outputs, dim=1), full, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(cache, full_cache)
                self.assertEqual(cache.shape, (2, 4 * dilation, 4))

    def test_kernel_one_and_depthwise_groups(self):
        x = torch.randn(2, 17, 4)
        layer = CausalConv(4, 4, 1, groups=4)
        full, cache = layer(x)
        first, carry = layer(x[:, :1])
        second, carry = layer(x[:, 1:], carry)
        torch.testing.assert_close(torch.cat((first, second), dim=1), full)
        self.assertEqual(cache.shape, (2, 0, 4))
        self.assertEqual(carry.shape, cache.shape)

    def test_local_encoder_chunk_equality_no_cache_mutation(self):
        encoder = LocalEncoder(hidden=4)
        x = torch.randn(2, 57, 64)
        full, expected_cache = encoder(x)
        first, cache = encoder(x[:, :19])
        saved_cache = clone_tree(cache)
        second, final_cache = encoder(x[:, 19:], cache)
        torch.testing.assert_close(torch.cat((first, second), dim=1), full, rtol=1e-5, atol=1e-6)
        assert_tree_close(self, final_cache, expected_cache)
        assert_tree_close(self, cache, saved_cache)

    def test_detach_preserves_values_but_breaks_history_graph(self):
        x = torch.randn(2, 11, 64, requires_grad=True)
        _, cache = LocalEncoder(hidden=4)(x)
        nested = {"cache": cache, "list": [x * 2], "tuple": (x * 3, "metadata")}
        detached = detach_state(nested)
        assert_tree_close(self, detached, nested)
        self.assertIsNot(detached, nested)
        self.assertIsNone(detached["cache"]["depthwise"].grad_fn)
        self.assertFalse(detached["list"][0].requires_grad)
        self.assertIsInstance(detached["tuple"], tuple)
        self.assertIsNotNone(nested["list"][0].grad_fn)

    def test_input_and_cache_validation(self):
        layer = CausalConv(4, 3, 5)
        with self.assertRaises(ValueError):
            layer(torch.randn(2, 5, 6))
        with self.assertRaises(ValueError):
            layer(torch.randn(2, 0, 4))
        with self.assertRaises(TypeError):
            layer(torch.ones(2, 5, 4, dtype=torch.long))
        with self.assertRaises(ValueError):
            layer(torch.randn(2, 5, 4), torch.zeros(1, 4, 4))
        with self.assertRaises(ValueError):
            layer(torch.randn(2, 5, 4), torch.zeros(2, 4, 4, dtype=torch.float64))


class SS06DenseStateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(206)

    def test_default_channel_shapes_and_single_sample_chunks(self):
        for model in (DenseMultiscaleDecoder(hidden=4), PersistentStateDecoder(hidden=4)):
            with self.subTest(model=type(model).__name__):
                output = model(torch.randn(2, 1, 64))
                self.assertEqual(output.prediction.shape, (2, 1, 1))
                self.assertTrue(torch.isfinite(output.prediction).all())
                self.assertIsInstance(output.state, dict)

    def test_dense_chunk_equivalence_in_each_branch_mode(self):
        x = torch.randn(2, 301, 64)
        for mode in ("multi", "short", "long"):
            with self.subTest(mode=mode):
                model = DenseMultiscaleDecoder(hidden=4, branch_mode=mode)
                full = model(x)
                cursor, state, pieces = 0, None, []
                for length in (1, 17, 1, 255, 1, 26):
                    result = model(x[:, cursor:cursor + length], state)
                    pieces.append(result.prediction)
                    state = result.state
                    cursor += length
                torch.testing.assert_close(torch.cat(pieces, dim=1), full.prediction, rtol=1e-5, atol=1e-6)
                assert_tree_close(self, state, full.state)

    def test_persistent_chunk_equivalence_and_state_immutability(self):
        model = PersistentStateDecoder(hidden=4)
        x = torch.randn(2, 81, 64)
        full = model(x)
        cursor, state, pieces = 0, None, []
        for length in (1, 13, 1, 28, 1, 37):
            previous = clone_tree(state)
            result = model(x[:, cursor:cursor + length], state)
            if state is not None:
                assert_tree_close(self, state, previous)
            pieces.append(result.prediction)
            state = result.state
            cursor += length
        torch.testing.assert_close(torch.cat(pieces, dim=1), full.prediction, rtol=1e-5, atol=1e-6)
        assert_tree_close(self, state, full.state)
        torch.testing.assert_close(full.aux["fast"][:, -1], full.state["fast"])
        torch.testing.assert_close(full.aux["slow"][:, -1], full.state["slow"])

    def test_suffix_perturbation_does_not_change_prefix(self):
        x = torch.randn(2, 63, 64)
        altered = x.clone()
        altered[:, 31:] = 20 * torch.randn_like(altered[:, 31:])
        for model in (DenseMultiscaleDecoder(hidden=4), PersistentStateDecoder(hidden=4)):
            with self.subTest(model=type(model).__name__):
                original = model(x).prediction
                changed = model(altered).prediction
                torch.testing.assert_close(original[:, :31], changed[:, :31])

    def test_explicit_reset_reproducibility_and_recurrent_history_control(self):
        model = PersistentStateDecoder(hidden=4)
        current = torch.randn(2, 23, 64)
        baseline = model(current).prediction
        past = model(torch.randn(2, 39, 64))
        # Hold current EEG AND its encoder cache fixed; intervene on recurrent history only.
        carried = model(current, past.state).prediction
        reset_recurrent = dict(past.state)
        reset_recurrent["fast"] = torch.zeros_like(past.state["fast"])
        reset_recurrent["slow"] = torch.zeros_like(past.state["slow"])
        interrupted = model(current, reset_recurrent).prediction
        self.assertGreater((carried - interrupted).abs().max().item(), 1e-7)
        torch.testing.assert_close(model(current, None).prediction, baseline)
        detached = model(current, detach_state(past.state)).prediction
        torch.testing.assert_close(detached, carried)

    def test_backward_reaches_active_parameters_and_input(self):
        cases = (DenseMultiscaleDecoder(hidden=4),
                 DenseMultiscaleDecoder(hidden=4, branch_mode="short"),
                 DenseMultiscaleDecoder(hidden=4, branch_mode="long"),
                 PersistentStateDecoder(hidden=4))
        for model in cases:
            with self.subTest(model=type(model).__name__, mode=getattr(model, "branch_mode", None)):
                x = torch.randn(2, 31, 64, requires_grad=True)
                result = model(x)
                result.prediction.square().mean().backward()
                self.assertIsNotNone(x.grad)
                self.assertTrue(torch.isfinite(x.grad).all())
                for name, parameter in model.named_parameters():
                    active = not name.startswith("branches.") or int(name.split(".")[1]) in model.active_dilations
                    if active:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    else:
                        self.assertIsNone(parameter.grad, name)

    def test_branch_control_uses_declared_branches_and_changes_output(self):
        multi = DenseMultiscaleDecoder(hidden=4)
        x = torch.randn(2, 29, 64)
        predictions = {}
        for mode, expected in (("multi", (1, 8, 64, 256)), ("short", (1,)), ("long", (256,))):
            model = DenseMultiscaleDecoder(hidden=4, branch_mode=mode)
            model.load_state_dict(multi.state_dict())
            result = model(x)
            predictions[mode] = result.prediction
            self.assertEqual(result.aux["active_dilations"], expected)
            self.assertEqual(set(result.state["branches"]), {str(value) for value in expected})
        self.assertGreater((predictions["multi"] - predictions["short"]).abs().max().item(), 1e-7)
        self.assertGreater((predictions["short"] - predictions["long"]).abs().max().item(), 1e-7)

    def test_slow_update_stays_bounded_and_is_correct(self):
        model = PersistentStateDecoder(hidden=4)
        x = torch.randn(2, 1, 64)
        result = model(x)
        local, _ = model.encoder(x)
        alpha = model.slow_coefficient
        self.assertTrue(((alpha > 0) & (alpha < 1)).all())
        torch.testing.assert_close(result.state["slow"], alpha * torch.tanh(model.slow_input(local[:, 0])))
        with torch.no_grad():
            model.slow_logit.copy_(torch.tensor([-1000., 1000., -3., 3.]))
        self.assertTrue(((model.slow_coefficient > 0) & (model.slow_coefficient < 1)).all())


if __name__ == "__main__":
    unittest.main()
