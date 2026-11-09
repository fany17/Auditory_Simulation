"""Synthetic engineering checks only; run on server2203, never on EEG data."""

import unittest

import torch

from m6a_public.continuity.common import detach_state
from m6a_public.continuity.predictive import PredictiveContinuityDecoder


class PredictiveContinuityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(603)

    def model(self, **kwargs):
        return PredictiveContinuityDecoder(in_channels=4, hidden=8, horizons=(2, 5), **kwargs)

    def test_shapes_chunk_equivalence_and_input_state_unchanged(self):
        model = self.model().eval()
        x = torch.randn(2, 29, 4)
        with torch.no_grad():
            whole = model(x)
            first = model(x[:, :7])
            retained = first.state["context"].clone()
            second = model(x[:, 7:19], first.state)
            third = model(x[:, 19:], second.state)
        self.assertEqual(tuple(whole.prediction.shape), (2, 29, 1))
        self.assertTrue(torch.equal(retained, first.state["context"]))
        torch.testing.assert_close(
            whole.prediction,
            torch.cat([first.prediction, second.prediction, third.prediction], dim=1),
            rtol=1e-5, atol=1e-6,
        )
        torch.testing.assert_close(whole.state["context"], third.state["context"])
        self.assertEqual(whole.aux["latent_predictions"], {})

    def test_causality_includes_predictive_heads(self):
        model = self.model(stage="pretrain").eval()
        x = torch.randn(2, 25, 4)
        changed = x.clone()
        changed[:, 13:] = 20 * torch.randn_like(changed[:, 13:])
        with torch.no_grad():
            before = model(x)
            after = model(changed)
        torch.testing.assert_close(before.prediction[:, :13], after.prediction[:, :13])
        for horizon in model.horizons:
            torch.testing.assert_close(
                before.aux["latent_predictions"][horizon][:, :13],
                after.aux["latent_predictions"][horizon][:, :13],
            )

    def test_reset_and_detach_are_distinct(self):
        model = self.model()
        history = torch.randn(2, 20, 4, requires_grad=True)
        current = torch.randn(2, 12, 4)
        state = model(history).state
        retained = model(current, state)
        detached = model(current, detach_state(state))
        reset = model(current)
        torch.testing.assert_close(retained.prediction, detached.prediction)
        self.assertGreater((reset.prediction - retained.prediction).abs().max().item(), 1e-6)
        detached.prediction.square().mean().backward()
        self.assertIsNone(history.grad)
        torch.testing.assert_close(reset.prediction, model(current).prediction)

    def test_predictive_gradients_and_future_target_isolation(self):
        model = self.model(stage="pretrain")
        x = torch.randn(2, 25, 4, requires_grad=True)
        target = x.detach().clone().requires_grad_(True)
        output = model(x)
        losses = model.predictive_loss(output, target)
        self.assertEqual(losses["valid_pairs"], {2: 46, 5: 40})
        losses["loss"].backward()
        self.assertIsNone(target.grad)
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            else:
                self.assertIsNone(parameter.grad, name)

    def test_teacher_is_static_until_explicit_ema(self):
        model = self.model(stage="pretrain").train()
        self.assertFalse(model.teacher.training)
        before = {name: value.clone() for name, value in model.teacher.state_dict().items()}
        x = torch.randn(1, 15, 4)
        losses = model.predictive_loss(model(x), x)
        losses["loss"].backward()
        for name, value in model.teacher.state_dict().items():
            self.assertTrue(torch.equal(before[name], value), name)
        # Deliberate synthetic perturbation tests the EMA operation. This is not
        # an optimizer update, fitting step, or a data-dependent parameter update.
        with torch.no_grad():
            next(model.encoder.parameters()).add_(0.25)
        model.update_teacher(decay=1.0)
        for name, value in model.teacher.state_dict().items():
            self.assertTrue(torch.equal(before[name], value), name)
        model.update_teacher(decay=0.0)
        for teacher, online in zip(model.teacher.parameters(), model.encoder.parameters()):
            torch.testing.assert_close(teacher, online)
            self.assertFalse(teacher.requires_grad)
        with self.assertRaises(ValueError):
            model.update_teacher(decay=1.1)

    def test_collapse_penalty_is_positive_and_finite(self):
        model = self.model(stage="pretrain")
        x = torch.zeros(2, 20, 4)
        output = model(x)
        # An exactly constant synthetic representation should incur the explicit
        # variance penalty rather than silently satisfy the predictive objective.
        output.aux["features"] = torch.zeros_like(output.aux["features"], requires_grad=True)
        output.aux["context"] = torch.zeros_like(output.aux["context"], requires_grad=True)
        losses = model.predictive_loss(output, x)
        self.assertGreater(losses["variance_loss"].item(), 0.9)
        self.assertTrue(torch.isfinite(losses["loss"]))

    def test_downstream_modes_and_active_parameter_accounting(self):
        x = torch.randn(2, 17, 4)
        for mode in ("no_pretrain", "frozen", "finetune"):
            model = self.model(mode=mode)
            output = model(x)
            output.prediction.square().mean().backward()
            self.assertEqual(output.aux["latent_predictions"], {})
            for module in (model.encoder, model.context):
                for parameter in module.parameters():
                    self.assertEqual(parameter.requires_grad, mode != "frozen")
                    self.assertEqual(parameter.grad is None, mode == "frozen")
            for parameter in model.envelope_head.parameters():
                self.assertIsNotNone(parameter.grad)
            for module in (model.teacher, model.predictors):
                for parameter in module.parameters():
                    self.assertFalse(parameter.requires_grad)
                    self.assertIsNone(parameter.grad)
            counts = model.parameter_counts()
            self.assertEqual(
                counts["registered_total"],
                counts["inference_active"] + counts["teacher_frozen"] + counts["auxiliary_heads"],
            )
            if mode == "frozen":
                self.assertEqual(counts["trainable"], sum(p.numel() for p in model.envelope_head.parameters()))
            else:
                self.assertEqual(counts["trainable"], counts["inference_active"])
            with self.assertRaises(ValueError):
                model.predictive_loss(output, x)
            if mode != "finetune":
                with self.assertRaises(ValueError):
                    model.set_stage("pretrain")

    def test_mask_support_and_discontinuity_rejections(self):
        model = self.model(stage="pretrain")
        x = torch.randn(2, 14, 4)
        output = model(x)
        mask = torch.ones(2, 14, dtype=torch.bool)
        mask[1, 10:] = False
        losses = model.predictive_loss(output, x, mask)
        self.assertEqual(losses["valid_pairs"], {2: 20, 5: 14})
        hole = mask.clone()
        hole[0, 3] = False
        with self.assertRaisesRegex(ValueError, "trailing padding"):
            model.predictive_loss(output, x, hole)
        with self.assertRaisesRegex(ValueError, "insufficient valid support"):
            model.predictive_loss(model(x[:, :5]), x[:, :5])
        with self.assertRaisesRegex(ValueError, "complete boundary-started"):
            model.predictive_loss(model(x, output.state), x)
        with self.assertRaises(ValueError):
            model.predictive_loss(output, x, mask.float())


if __name__ == "__main__":
    unittest.main()
