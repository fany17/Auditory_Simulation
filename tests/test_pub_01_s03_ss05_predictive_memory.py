"""Tests for stateful causality, memory boundaries and gradient validity."""
import unittest
import torch
from scripts.pub_01_s03_ss05_predictive_memory import (
    PredictiveEventMemory, parameter_count,
)


class PredictiveMemoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1024)

    def test_shapes_finite_backward_and_parameter_inventory(self):
        counts = set()
        for variant in PredictiveEventMemory.ALLOWED_VARIANTS:
            m = PredictiveEventMemory(input_channels=6, hidden=8, key_dim=4, variant=variant)
            counts.add(parameter_count(m))
            x = torch.randn(2, 13, 6, requires_grad=True)
            output = m(x)
            self.assertEqual(tuple(output.shape), (2, 13, 1))
            self.assertTrue(torch.isfinite(output).all())
            output.square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
        self.assertEqual(len(counts), 1)

    def test_streaming_equals_unbroken_sequence(self):
        m = PredictiveEventMemory(input_channels=6, hidden=8, key_dim=4, variant="full").eval()
        x = torch.randn(2, 23, 6)
        with torch.no_grad():
            full = m(x)
            a, state = m(x[:, :7], return_state=True)
            b, state = m(x[:, 7:15], state, return_state=True)
            c = m(x[:, 15:], state)
        self.assertTrue(torch.allclose(full, torch.cat([a, b, c], 1), atol=1e-6))

    def test_future_input_never_changes_past_prediction(self):
        m = PredictiveEventMemory(input_channels=6, hidden=8, key_dim=4).eval()
        x = torch.randn(2, 24, 6)
        edited = x.clone()
        edited[:, 14:] += 42
        with torch.no_grad():
            a, b = m(x), m(edited)
        self.assertTrue(torch.equal(a[:, :14], b[:, :14]))

    def test_local_only_depends_on_three_frames(self):
        m = PredictiveEventMemory(input_channels=6, hidden=8, key_dim=4, variant="local").eval()
        x = torch.randn(2, 24, 6)
        edited = x.clone()
        edited[:, :8] += 30
        with torch.no_grad():
            a, b = m(x), m(edited)
        self.assertTrue(torch.allclose(a[:, 12:], b[:, 12:], atol=1e-7))

    def test_record_reset_changes_state_and_eliminates_cross_record_leak(self):
        m = PredictiveEventMemory(input_channels=6, hidden=8, key_dim=4, variant="full").eval()
        history = torch.randn(2, 16, 6)
        current = torch.randn(2, 8, 6)
        with torch.no_grad():
            _, from_history = m(history, return_state=True)
            y_contiguous = m(current, from_history)
            y_reset = m(current)
            y_independent_again = m(current)
        self.assertGreater(float((y_contiguous - y_reset).abs().max()), 1e-5)
        self.assertTrue(torch.equal(y_reset, y_independent_again))

    def test_history_reaches_output_beyond_local_window(self):
        m = PredictiveEventMemory(input_channels=6, hidden=8, key_dim=4).eval()
        x = torch.randn(1, 24, 6)
        modified = x.clone()
        modified[:, 0, 0] += 5.0
        with torch.no_grad():
            a = m(x)
            b = m(modified)
        self.assertGreater(float((a[:, -1] - b[:, -1]).abs().max()), 1e-8)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            PredictiveEventMemory(variant="arbitrary")
        with self.assertRaises(ValueError):
            PredictiveEventMemory(memory_decay=1.1)
        with self.assertRaises(ValueError):
            PredictiveEventMemory()(torch.zeros(3, 64))


if __name__ == "__main__":
    unittest.main()


class WindowDecoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)

    def test_window_shapes_gradients_and_chunk_equivalence(self):
        from scripts.pub_01_s03_ss05_window_decoder import WindowPredictiveDecoder
        m=WindowPredictiveDecoder(eeg_channels=6,width=8,window_samples=16)
        x=torch.randn(2,5,16,6,requires_grad=True)
        full=m(x)
        self.assertEqual(tuple(full.shape),(2,5,16,1))
        self.assertTrue(torch.isfinite(full).all())
        full.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        with torch.no_grad():
            first,state=m(x[:,:2],return_state=True)
            rest=m(x[:,2:],state)
        self.assertTrue(torch.allclose(full.detach(),torch.cat((first,rest),dim=1),atol=1e-6))

    def test_only_past_windows_affect_current(self):
        from scripts.pub_01_s03_ss05_window_decoder import WindowPredictiveDecoder
        m=WindowPredictiveDecoder(eeg_channels=6,width=8,window_samples=16).eval()
        x=torch.randn(1,6,16,6);edited=x.clone()
        edited[:,5] += 20
        with torch.no_grad():
            a=m(x);b=m(edited)
        self.assertTrue(torch.equal(a[:,:5],b[:,:5]))


class GRUWindowControlTests(unittest.TestCase):
    def test_gru_window_outputs_and_stateful_chunks(self):
        from scripts.pub_01_s03_ss05_window_decoder import WindowGRUDecoder
        m=WindowGRUDecoder(eeg_channels=6,width=8,window_samples=16,gru_hidden=24).eval()
        x=torch.randn(2,5,16,6)
        with torch.no_grad():
            full=m(x)
            first,state=m(x[:,:2],return_state=True)
            remaining=m(x[:,2:],state)
        self.assertEqual(tuple(full.shape),(2,5,16,1))
        self.assertTrue(torch.allclose(full,torch.cat((first,remaining),1),atol=1e-6))
