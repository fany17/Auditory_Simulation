"""Network-free tests of resume refusal, reuse, exclusion-safe path handling."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('pub_download', Path(__file__).parents[1] / 'scripts/pub_01_s06_download.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        # Fixtures persist in temp; no deletion of user or temporary files.
        fixture_root = Path(__file__).parents[1] / 'test_fixtures'
        fixture_root.mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix='pub01_test_', dir=fixture_root)).resolve()
        self.row = dict(relative_path='a.txt', bytes='3', source_url='https://example.invalid')

    def test_reuse_size_match_without_network(self):
        (self.root / 'a.txt').write_bytes(b'abc')
        with patch.object(module.urllib.request, 'urlopen', side_effect=AssertionError('network')):
            result = module.download(self.row, self.root)
        self.assertEqual(result['status'], 'EXISTING_SIZE_MATCH')

    def test_existing_conflict_preserved(self):
        (self.root / 'a.txt').write_bytes(b'ab')
        result = module.download(self.row, self.root)
        self.assertEqual(result['status'], 'CONFLICT_PRESERVED')
        self.assertEqual((self.root / 'a.txt').read_bytes(), b'ab')

    def test_path_traversal_refused(self):
        with self.assertRaises(ValueError):
            module.download(dict(self.row, relative_path='../outside'), self.root)

    def test_complete_partial_promoted(self):
        (self.root / 'a.txt.partial').write_bytes(b'abc')
        result = module.download(self.row, self.root)
        self.assertEqual(result['status'], 'DOWNLOADED_SIZE_MATCH')
        self.assertEqual((self.root / 'a.txt').read_bytes(), b'abc')

    def test_resume_ignored_by_server_is_preserved(self):
        (self.root / 'a.txt.partial').write_bytes(b'a')
        class Response:
            status = 200
            headers = {}
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
        with patch.object(module.urllib.request, 'urlopen', return_value=Response()), patch.object(module.time, 'sleep'):
            result = module.download(self.row, self.root)
        self.assertEqual(result['status'], 'FAILED_PARTIAL_RETAINED')
        self.assertEqual((self.root / 'a.txt.partial').read_bytes(), b'a')


if __name__ == '__main__':
    unittest.main()
