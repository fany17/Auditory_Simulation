"""Collection/audio-identity metadata comparison only. No old neural/audio payload reads."""
import hashlib,json
from pathlib import Path
from pub_01_s03_ss07_openneuro_transport import ROOT


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    base=ROOT.parent
    source=base/'s03_strong_baselines_20261015/transformer_available01/prepared/manifest_frozen.json'
    old=json.loads(source.read_text());rows=old['records'];train=[r for r in rows if r['phase']=='train']
    stories=sorted({r['story'] for r in rows});train_stories=sorted({r['story'] for r in train})
    new=json.loads((ROOT/'AUDIO_PREPARE_COMPLETED.json').read_text())['audios']
    dtu=base/'s03_incremental_continuity_20261101/external_dtu'
    pins=[source,dtu/'INTAKE_AUTHORIZATION.json',dtu/'DOWNLOAD_EXTRACT_COMPLETED.json',ROOT/'AUDIO_PREPARE_COMPLETED.json']
    value=dict(scope='Study/language/book and available identity metadata only; no old NN or old signal/audio values',
        new=dict(dataset='OpenNeuro ds004408 v1.0.8',commit='b2d91784a90c18e1d76bd77dcb2b29b0ce27a0be',
            study='Di Liberto2015 plus Broderick2018; Trinity College Dublin',people=19,records=380,
            language='English',book='The Old Man and the Sea',narratives=1,
            audio_identities=[dict(path=r['path'],target_sha256=r['target_sha256']) for r in new]),
        source_Sparr=dict(dataset='SparrKULee',study='KU Leuven',language='Dutch/Flemish',
            existing_manifest_records=len(rows),training_records=len(train),
            training_audio_story_IDs=train_stories,all_available_story_IDs=stories,
            raw_audio_digest_comparison_available=False,
            reason='Existing model manifest names processed NPZ audio objects and numeric story IDs; these are not a directly comparable set of original WAV fingerprints'),
        prior_exposed_DTU=dict(dataset='Zenodo1199011 v1',study='Technical University of Denmark',
            language='Danish',task='Two competing speakers / reverberation; existing selected preprocessed trials',
            only_existing_DATA_preproc_and_script=True,raw_AUDIO_archive_downloaded_for_this_check=False,
            original_audio_identity_complete=False),
        references=[
            dict(url='https://github.com/OpenNeuroDatasets/ds004408/blob/b2d91784a90c18e1d76bd77dcb2b29b0ce27a0be/README',
                supports='Pinned dataset links its acquisition to DiLiberto2015/Broderick2018 and one OldManAndSea audiobook'),
            dict(url='https://mickcrosse.com/assets/pubs/Broderick_etal_CurrBiol_2018.pdf',
                supports='Original paper:19 native-English single-audiobook experiment; key resources identify OldManAndSea. Indexed primary text read; full fetch timed out.'),
            dict(url='https://assets-eu.researchsquare.com/files/rs-3397581/v1_covered_853cbd8c-9086-43e4-874b-adf373f85270.pdf?c=1718765782',
                supports='Authors SparrKULee data-descriptor preprint labels SparrKULee Dutch and contrasts English Broderick/Danish DTU collections'),
            dict(url='https://doi.org/10.3389/fnins.2018.00531',
                supports='DTU primary methods section2.4: professionally recorded Danish male/female fictional stories'),
            dict(url='https://zenodo.org/records/1199011',supports='DTU18 participant study/data packaging and available original audio metadata')],
        interpretation='Metadata supports a distinct collection and stimulus language/narrative source, not a complete cross-corpus waveform or participant identity proof',
        residual_limits=[
            'No full original-waveform identity comparison across Sparr/DTU/ds004408 was possible from available manifests',
            'No complete audit can exclude exposure of this narration/audio during large XLSR pretraining; do not claim guaranteed unseen audio to every pretraining stage',
            'At most19 independent participant units conditional on one shared narrative/speaker;20 segments are not20 independent stories',
            'All new predictive evaluation occurs after frozen shared offline/transductive EEG gain calibration, not an uncalibrated zero-shot confirmation'],
        metadata_pins={str(p):sha(p) for p in pins},source_sha256=sha(__file__),new_NN=0,old_NN=0)
    with (ROOT/'SOURCE_INDEPENDENCE.json').open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    print(json.dumps(dict(source_training_records=len(train),source_training_story_IDs=len(train_stories),
        source_all_story_IDs=len(stories),new_people=19,new_narratives=1,identity_caveats=True,new_or_old_NN=0)))


if __name__=='__main__':main()
