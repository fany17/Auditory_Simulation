"""Input-only cohort selection; strict and user-authorized available modes."""
PHASES = ('train', 'val', 'test')


def select_subjects(split, records, failures, policy):
    mode = policy.get('mode', 'strict_complete')
    if mode not in ('strict_complete', 'available_complete_subjects'):
        raise ValueError('unknown cohort mode')
    if mode == 'available_complete_subjects' and not policy.get('user_authorization'):
        raise RuntimeError('available subset mode requires recorded user authorization')
    selected, excluded = [], []
    for subject, phases in split.items():
        expected = {(p, story) for p, stories in phases.items() for story in stories}
        actual = {(r['phase'], r['story']) for r in records if r['subject']==subject}
        missing = sorted(expected-actual)
        failed = [r for r in failures if r.get('subject')==subject]
        if missing or failed or any(not phases.get(p) for p in PHASES):
            excluded.append({'subject':subject,'reason':'required_input_missing_or_invalid',
                'missing_phase_story_pairs':missing,'input_failures':failed})
        else:
            selected.append(subject)
    if mode=='strict_complete' and (excluded or failures):
        raise RuntimeError('Published author cohort incomplete in strict_complete mode')
    if not selected:
        raise RuntimeError('no subjects with complete required train/val/test inputs')
    return selected, excluded


def validate_training_cohort(manifest, cfg):
    policy = cfg.get('cohort_policy', {'mode':'strict_complete'})
    mode = policy.get('mode','strict_complete')
    if mode=='strict_complete':
        if manifest.get('failures') or manifest.get('excluded_subjects'):
            raise RuntimeError('Published author cohort incomplete in strict_complete mode')
    elif mode=='available_complete_subjects':
        if not policy.get('user_authorization') or manifest.get('cohort_policy')!=policy:
            raise RuntimeError('authorized available cohort freeze missing or differs')
        if manifest.get('selected_input_failures'):
            raise RuntimeError('selected cohort still has input failures')
        if not manifest.get('subjects') or not manifest.get('selected_subjects_complete'):
            raise RuntimeError('selected subjects do not have complete required phases')
    else:
        raise ValueError('unknown cohort mode')
