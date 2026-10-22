"""Cross-framework scientific-contract checks on synthetic inputs only."""
import unittest

import torch

from m6a_public.continuity.objectives import causal_smooth, dense_objective
from m6a_public.continuity.registry import MODELS, build_model
from m6a_public.continuity.streaming import StreamIdentity, StreamRunner, compare_histories, compare_state_histories


class ContinuityIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(606)
        self.key = (StreamIdentity("s1", "session1", "r1", "train"),)

    def test_boundary_reset_for_subject_session_record_and_split(self):
        x = torch.randn(1, 31, 4)
        current = torch.randn(1, 17, 4)
        keys = [StreamIdentity("s2", "session1", "r1", "train"),
                StreamIdentity("s1", "session2", "r1", "train"),
                StreamIdentity("s1", "session1", "r2", "train"),
                StreamIdentity("s1", "session1", "r1", "val")]
        for name in MODELS:
            model = build_model(name, in_channels=4, hidden=8)
            for key in keys:
                with self.subTest(model=name, key=key):
                    stream = StreamRunner(model)
                    stream.step(x, self.key, 0)
                    got = stream.step(current, (key,), 0)
                    torch.testing.assert_close(got.prediction, model(current).prediction)

    def test_offsets_fail_closed_and_detach_preserves_predictions(self):
        for name in MODELS:
            with self.subTest(name=name):
                model = build_model(name, in_channels=4, hidden=8)
                stream = StreamRunner(model)
                history = torch.randn(1, 23, 4, requires_grad=True)
                current = torch.randn(1, 11, 4)
                past = stream.step(history, self.key, 0)
                expected = model(current, past.state).prediction
                with self.assertRaisesRegex(ValueError, "noncontiguous"):
                    stream.step(current, self.key, 24)
                stream.detach()
                actual = stream.step(current, self.key, 23).prediction
                torch.testing.assert_close(actual, expected)
                actual.square().mean().backward()
                self.assertIsNone(history.grad)

    def test_same_current_input_across_history_interventions(self):
        current = torch.randn(1, 17, 4)
        unchanged = current.clone()
        real = torch.randn(1, 25, 4)
        wrong = torch.randn_like(real)
        for name in MODELS:
            model = build_model(name, in_channels=4, hidden=8)
            outputs = compare_histories(model, current, real, wrong)
            torch.testing.assert_close(current, unchanged, rtol=0, atol=0)
            torch.testing.assert_close(outputs["reset"].prediction, model(current).prediction)
            torch.testing.assert_close(outputs["real"].prediction, model(current, model(real).state).prediction)
            torch.testing.assert_close(outputs["mismatched"].prediction, model(current, model(wrong).state).prediction)

    def test_dense_objective_is_over_full_continuous_sequence(self):
        p = torch.tensor([[[0.], [1.], [2.], [0.], [1.], [2.]]], requires_grad=True)
        y = torch.tensor([[[0.], [1.], [2.], [100.], [101.], [102.]]])
        whole = dense_objective(p, y, mse_weight=0.)
        chunk_loss = (dense_objective(p[:, :3], y[:, :3], mse_weight=0.)["loss"]
                      + dense_objective(p[:, 3:], y[:, 3:], mse_weight=0.)["loss"]) / 2
        self.assertGreater((whole["loss"] - chunk_loss).item(), .9)
        whole["loss"].backward()
        self.assertTrue(torch.isfinite(p.grad).all())

    def test_state_only_intervention_keeps_current_local_features_identical(self):
        current = torch.randn(1, 23, 4)
        history = torch.randn(1, 31, 4)
        wrong = torch.randn_like(history) * 3
        for name in MODELS:
            with self.subTest(model=name):
                model = build_model(name, in_channels=4, hidden=8)
                encoded = []
                hook = model.encoder.register_forward_hook(lambda module, args, output: encoded.append(output[0]))
                try:
                    result = compare_state_histories(model, current, history, wrong)
                finally:
                    hook.remove()
                # Last three encoder calls are real/reset/mismatched current EEG.
                torch.testing.assert_close(encoded[-3], encoded[-2], rtol=0, atol=0)
                torch.testing.assert_close(encoded[-3], encoded[-1], rtol=0, atol=0)
                self.assertEqual(set(result), {"real", "reset", "mismatched"})

    def test_dense_mask_and_undefined_target(self):
        p = torch.randn(2, 20, 1, requires_grad=True)
        y = torch.randn_like(p)
        mask = torch.ones(2, 20, dtype=torch.bool)
        mask[:, 15:] = False
        padded_y = y.clone()
        padded_y[:, 15:] = float("nan")
        a = dense_objective(p, padded_y, mask)
        b = dense_objective(p[:, :15], y[:, :15])
        torch.testing.assert_close(a["loss"], b["loss"])
        a["loss"].backward()
        self.assertTrue(torch.equal(p.grad[:, 15:], torch.zeros_like(p.grad[:, 15:])))
        with self.assertRaisesRegex(ValueError, "constant"):
            dense_objective(p, torch.ones_like(p))
        with self.assertRaisesRegex(ValueError, "two valid"):
            dense_objective(p, y, torch.zeros_like(mask))

    def test_output_smoother_is_causal_and_chunk_equivalent(self):
        y = torch.randn(2, 31, 1)
        a, state = causal_smooth(y[:, :13])
        b, _ = causal_smooth(y[:, 13:], state=state)
        full, _ = causal_smooth(y)
        torch.testing.assert_close(torch.cat((a, b), dim=1), full)
        altered = y.clone()
        altered[:, 13:] *= 100
        changed, _ = causal_smooth(altered)
        torch.testing.assert_close(changed[:, :13], full[:, :13])


if __name__ == "__main__":
    unittest.main()
