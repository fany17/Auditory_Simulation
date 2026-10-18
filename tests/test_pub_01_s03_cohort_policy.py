"""Input-only selection and explicit authorization guard, no signal fixtures."""
import unittest
from pub_01_s03_cohort_policy import select_subjects, validate_training_cohort


class CohortPolicy(unittest.TestCase):
    def setUp(self):
        self.split={s:{'train':['a'],'val':['b'],'test':['c']} for s in ('s1','s2')}
        self.records=[{'subject':s,'phase':p,'story':story} for s,phases in self.split.items()
            for p,stories in phases.items() for story in stories if (s,p)!=('s2','val')]
        self.failures=[{'subject':'s2','phase':'val','story':'b','type':'missing_input'}]
        self.policy={'mode':'available_complete_subjects','user_authorization':'2026-10-15 explicit human instruction'}

    def test_available_mode_preserves_complete_subject_and_excludes_missing(self):
        subjects,excluded=select_subjects(self.split,self.records,self.failures,self.policy)
        self.assertEqual(subjects,['s1'])
        self.assertEqual(excluded[0]['subject'],'s2')
        self.assertEqual(excluded[0]['missing_phase_story_pairs'],[('val','b')])

    def test_strict_mode_remains_strict(self):
        with self.assertRaisesRegex(RuntimeError,'strict_complete'):
            select_subjects(self.split,self.records,self.failures,{'mode':'strict_complete'})

    def test_available_mode_requires_recorded_authorization(self):
        with self.assertRaisesRegex(RuntimeError,'authorization'):
            select_subjects(self.split,self.records,self.failures,{'mode':'available_complete_subjects'})

    def test_excluded_source_failures_do_not_block_authorized_selected_cohort(self):
        manifest={'cohort_policy':self.policy,'subjects':['s1'],'selected_subjects_complete':True,
            'selected_input_failures':[],'failures':self.failures,'excluded_subjects':[{'subject':'s2'}]}
        validate_training_cohort(manifest,{'cohort_policy':self.policy})
        manifest['selected_input_failures']=[{'subject':'s1','type':'bad_input'}]
        with self.assertRaisesRegex(RuntimeError,'selected cohort'):
            validate_training_cohort(manifest,{'cohort_policy':self.policy})


if __name__=='__main__':
    unittest.main(verbosity=2)
