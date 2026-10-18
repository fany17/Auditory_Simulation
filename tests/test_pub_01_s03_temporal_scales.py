"""CPU-only shape, gradient, and matched-parameter tests; no patient data."""
import unittest
import torch

from scripts.pub_01_s03_temporal_scales import (
    DILATION_PRESETS,
    TemporalScaleDecoder,
    effective_receptive_field,
    model_parameters,
)


class TemporalScaleTests(unittest.TestCase):
    def test_matching_parameter_counts_and_shapes(self):
        counts = set()
        for name in DILATION_PRESETS:
            decoder = TemporalScaleDecoder(variant=name)
            counts.add(model_parameters(decoder))
            y = decoder(torch.randn(2, 192, 64))
            self.assertEqual(tuple(y.shape), (2, 192, 1))
            self.assertTrue(torch.isfinite(y).all())
        self.assertEqual(len(counts), 1)

    def test_receptive_fields_at_64_hz(self):
        self.assertEqual(
            tuple(effective_receptive_field(9, d) for d in (1, 4, 16)),
            (9, 33, 129),
        )

    def test_gradients_backpropagate(self):
        torch.manual_seed(1024)
        m = TemporalScaleDecoder(variant="multiscale")
        x = torch.randn(2, 192, 64, requires_grad=True)
        loss = m(x).square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad.abs().sum()), 0)
        for p in m.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())

    def test_temporal_support_only(self):
        # All modules are pointwise apart from the temporal depthwise filters.
        # Thus a perturbation >64 time steps away from center is outside
        # the longest 129-sample support even in the multiscale variant.
        torch.manual_seed(23)
        m = TemporalScaleDecoder(variant="multiscale").eval()
        x = torch.randn(1, 192, 64)
        edited = x.clone()
        edited[:, 8, :] += 10.0
        with torch.no_grad():
            y1, y2 = m(x), m(edited)
        self.assertTrue(torch.allclose(y1[:, 100], y2[:, 100], atol=1e-6, rtol=1e-6))

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            TemporalScaleDecoder(width=47)
        with self.assertRaises(ValueError):
            TemporalScaleDecoder(kernel_size=8)
        with self.assertRaises(ValueError):
            TemporalScaleDecoder(dilations=(1, -1, 4))
        with self.assertRaises(ValueError):
            TemporalScaleDecoder()(torch.randn(2, 64, 192))


if __name__ == "__main__":
    unittest.main()
