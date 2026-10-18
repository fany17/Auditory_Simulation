"""Verify incomplete author inputs cannot accidentally start subset training.

2203-only JSON fixture; not research data. Fixture is retained, never deleted.
"""
import json
from pathlib import Path
import tempfile
import unittest
from pub_01_s03_hdps_train_paper import main


class AuthorCohortGuard(unittest.TestCase):
    def test_incomplete_inputs_fail_before_dataset_or_run_creation(self):
        base = Path(__file__).resolve().parent
        if not str(base).startswith('/home/fanyu/auditory_simulation_m6a/'):
            self.skipTest('run only on2203')
        fixture=Path(tempfile.mkdtemp(prefix='guard_fixture_',dir=base))
        prepared=fixture/'prepared_paper_m01';prepared.mkdir()
        cfg={'work':str(fixture),'patient_data':'FORBIDDEN'}
        config=fixture/'config.json';config.write_text(json.dumps(cfg))
        (prepared/'completed.json').write_text('{}')
        (prepared/'config_frozen.json').write_text(json.dumps(cfg))
        (prepared/'manifest_frozen.json').write_text(json.dumps({'failures':[{'subject':'sub-019','type':'raw_unavailable'}]}))
        with self.assertRaisesRegex(RuntimeError,'strict_complete'):
            main(config)
        self.assertFalse((fixture/'train_paper_m01_seed1024').exists())
        # candidate_groups and array files deliberately absent: refusal must
        # happen before reading them or creating a scientific run directory.


if __name__=='__main__':
    unittest.main(verbosity=2)
