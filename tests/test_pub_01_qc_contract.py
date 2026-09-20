import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('qc', Path(__file__).parents[1] / 'scripts/pub_01_s06_qc_ds004703.py')
qc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qc)


class QCContractTests(unittest.TestCase):
    def test_empty_eligible_table_retains_schema(self):
        root = Path(__file__).parents[1] / 'test_fixtures'
        root.mkdir(exist_ok=True)
        path = Path(tempfile.mkdtemp(prefix='empty_', dir=root)) / 'empty.csv'
        qc.save_csv(path, [], ['recording', 'channel', 'eligible'])
        with path.open() as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, ['recording', 'channel', 'eligible'])
            self.assertEqual(list(reader), [])

    def test_empty_without_schema_fails_explicitly(self):
        with self.assertRaises(ValueError):
            qc.save_csv(Path('unused.csv'), [])


if __name__ == '__main__':
    unittest.main()
