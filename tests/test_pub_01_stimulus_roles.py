import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from pub_01_sparrkulee_raw_stimulus_audit import classify_role


class RoleTests(unittest.TestCase):
    def test_exact_does_not_claim_speech(self):
        role, basis = classify_role('audiobook_1.npz.gz', 'EXACT')
        self.assertEqual(role, 'PUBLISHED_STIMULUS_AUDIO')
        self.assertIn('does not prove speech', basis)

    def test_trigger_names_provisional(self):
        for name in ['triggers.npz.gz', 't_book.npz.gz']:
            self.assertEqual(classify_role(name, 'NOT_APPLICABLE')[0], 'TRIGGER_BY_FILENAME')

    def test_unmatched_auxiliaries_not_audio(self):
        for name in ['audiobook_3_noise.npz.gz', 'audiobook_6_1_swn.npz.gz', 'noise_book.npz.gz', 'unknown.npz.gz']:
            self.assertEqual(classify_role(name, 'NO_MATCHING_PUBLISHED_DATA_DICT')[0], 'AUXILIARY_ROLE_UNVERIFIED')


if __name__ == '__main__':
    unittest.main()
