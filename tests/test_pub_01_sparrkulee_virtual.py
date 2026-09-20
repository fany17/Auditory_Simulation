import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('virtual', Path(__file__).parents[1] / 'scripts/pub_01_sparrkulee_virtual.py')
virtual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(virtual)


class CommonOverlapTests(unittest.TestCase):
    def test_explicit_common_prefix_and_bounds(self):
        root = Path(__file__).parents[1] / 'test_fixtures'
        root.mkdir(exist_ok=True)
        folder = Path(tempfile.mkdtemp(prefix='overlap_', dir=root))
        eeg = np.arange(64 * 12, dtype=float).reshape(64, 12)
        env = np.arange(10, dtype=float).reshape(10, 1)
        np.save(folder / 'eeg.npy', eeg)
        np.save(folder / 'env.npy', env)
        loader = virtual.CommonOverlap(folder / 'eeg.npy', folder / 'env.npy')
        self.assertEqual(loader.length, 10)
        a, b = loader.read(0, 10)
        np.testing.assert_array_equal(a, eeg[:, :10].T)
        np.testing.assert_array_equal(b, env)
        with self.assertRaises(ValueError):
            loader.read(0, 11)


if __name__ == '__main__':
    unittest.main()
