import gzip
import importlib.util
import io
from pathlib import Path
import pickle
import tempfile
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('format_qc', Path(__file__).parents[1] / 'scripts/pub_01_s06_format_qc.py')
qc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qc)


class FormatTests(unittest.TestCase):
    def test_numpy_roundtrip(self):
        x = {'a': np.arange(5, dtype=np.float64), 'sampling_rate': 64}
        y = qc.NumericUnpickler(io.BytesIO(pickle.dumps(x))).load()
        np.testing.assert_array_equal(x['a'], y['a'])

    def test_arbitrary_global_refused(self):
        with self.assertRaises(pickle.UnpicklingError):
            qc.NumericUnpickler(io.BytesIO(b'cos\nsystem\n.')).load()

    def test_publisher_reference_inert(self):
        value = qc.NumericUnpickler(io.BytesIO(b'c__main__\ntemp_stimulus_load_fn\n.')).load()
        self.assertIs(value, qc.inert_publisher_loader)
        with self.assertRaises(pickle.UnpicklingError):
            value()

    def test_reduce_cannot_execute_publisher_reference(self):
        with self.assertRaises(pickle.UnpicklingError):
            qc.NumericUnpickler(io.BytesIO(b'c__main__\ntemp_stimulus_load_fn\n)R.')).load()

    def test_object_array_refused(self):
        with self.assertRaises(ValueError):
            qc.array_summary(np.array([None], dtype=object))

    def test_bdf_calibration_and_samples(self):
        root = Path(__file__).parents[1] / 'test_fixtures'
        root.mkdir(exist_ok=True)
        path = Path(tempfile.mkdtemp(prefix='bdf_', dir=root)) / 'tiny.bdf.gz'
        header = bytearray(b' ' * 512)
        header[:8] = b'\xffBIOSEMI'
        def field(start, width, value):
            header[start:start + width] = str(value).encode().ljust(width)
        for start, width, value in [(184,8,512),(236,8,1),(244,8,1),(252,4,1),
                                    (256,16,'A1'),(256+96,8,'uV'),(256+104,8,-100),
                                    (256+112,8,100),(256+120,8,-8388608),(256+128,8,8388607),
                                    (256+216,8,2)]:
            field(start,width,value)
        with gzip.open(path,'wb') as f:
            f.write(header + b'\x00\x00\x00\x01\x00\x00')
        result = qc.bdf_qc(path)
        self.assertEqual(result['physical_dimensions'], ['uV'])
        self.assertEqual(result['calibration_valid'], [True])
        self.assertEqual(result['sample_rates_hz'], [2])
        self.assertEqual(result['flat_channels'], [])


if __name__ == '__main__':
    unittest.main()
