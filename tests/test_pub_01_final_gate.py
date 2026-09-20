import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from pub_01_s06_final_handoff import require_complete, require_known_hold, decoded_format, tag_dataset


class GateTests(unittest.TestCase):
    def fixture(self):
        return dict(expected_public_files=4142,downloaded_size_match=4142,format_checked=4142,
                    expected_public_bytes=135967030049,downloaded_bytes=135967030049,
                    missing_or_partial=0,restricted_excluded=196,format_hold=1)

    def test_completed_scope_keeps_known_exclusion(self):
        require_complete(self.fixture())

    def test_partial_download_or_qc_rejected(self):
        for field in ['downloaded_size_match','format_checked','downloaded_bytes']:
            status=self.fixture();status[field]-=1
            with self.assertRaises(AssertionError):require_complete(status)

    def test_hidden_or_extra_hold_rejected(self):
        for holds in [0,2]:
            status=self.fixture();status['format_hold']=holds
            with self.assertRaises(AssertionError):require_complete(status)

    def test_real_failed_record_shape(self):
        row=dict(relative_path='derivatives/preprocessed_stimuli/podcast_35-1.data_dict',
                 expected_bytes=116391936,status='HOLD',error='UnpicklingError: pickle data was truncated')
        self.assertEqual(decoded_format(row),'FAILED_FORMAT')
        require_known_hold([dict(row,format_readability=row['status'])])

    def test_equal_hold_count_different_path_refused(self):
        with self.assertRaises(AssertionError):
            require_known_hold([dict(relative_path='different/podcast_35-1.data_dict',format_readability='HOLD')])

    def test_existing_dataset_column(self):
        self.assertEqual(tag_dataset(dict(dataset='SparrKULee',bytes='123'),'SparrKULee')['bytes'],'123')
        with self.assertRaises(AssertionError):tag_dataset(dict(dataset='wrong'),'SparrKULee')


if __name__=='__main__':unittest.main()
