"""Static descriptive summaries; no payload export or predictive results."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pub_01_s03_ss07_openneuro_transport import ROOT


def main():
    audio=json.loads((ROOT/'AUDIO_PREPARE_COMPLETED.json').read_text())['audios']
    assert len(audio)==20
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained');x=np.arange(1,21)
    axes[0].bar(x,[r['duration_seconds'] for r in audio],color='#457b9d')
    axes[0].set(xlabel='Audio segment',ylabel='Duration (s)',ylim=(0,220))
    axes[1].bar(x,[100*r['different_sample_fraction'] for r in audio],color='#e9c46a')
    axes[1].set(xlabel='Audio segment',ylabel='Samples with L != R (%)',ylim=(0,.3))
    fig.suptitle('One narrative, 20 segments; sparse left/right differences shown in percent')
    target=ROOT/'figures/audio_structure_v2.png';assert not target.exists()
    fig.savefig(target,dpi=150);plt.close(fig)
    headers=json.loads((ROOT/'HEADERS_AND_SIZES_V2.json').read_text())
    count=np.empty((19,20),int)
    for r in headers['channel_table_mapping']:
        subject=int(r['path'].split('/')[0].split('-')[1])-1
        run=int(r['path'].split('_run-')[1].split('_')[0])-1
        table=ROOT/'metadata/channel_tables'/f"{r['git_blob_sha']}.tsv"
        rows=list(csv.DictReader(table.open(),delimiter='\t'))
        assert len(rows)==128 and all(x['status'] in ('good','bad') for x in rows)
        count[subject,run]=sum(x['status']=='bad' for x in rows)
    layout=json.loads((ROOT/'PROVISIONAL_DATA_MASK.json').read_text())
    raw_seconds=np.array([r['raw_frames']/512 for r in layout['rows']])
    audio_seconds=np.array([audio[int(r['run'])-1]['duration_seconds'] for r in layout['rows']])
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    im=axes[0].imshow(count,aspect='auto',origin='lower',cmap='magma',vmin=0,vmax=34,
        extent=(.5,20.5,.5,19.5))
    axes[0].set(xlabel='Run',ylabel='Subject',title='Curator-declared bad channels / 128')
    fig.colorbar(im,ax=axes[0],label='Bad channels')
    axes[1].scatter(audio_seconds,raw_seconds,s=14,alpha=.5,color='#457b9d')
    axes[1].plot([160,210],[160,210],'--',lw=1,color='black')
    axes[1].set(xlabel='Actual audio duration (s)',ylabel='EEG binary length / 128 / 512 (s)',
        title='Pinned payload sizes, not full EEG readability or sync proof')
    target=ROOT/'figures/header_layout_overview.png';assert not target.exists()
    fig.savefig(target,dpi=150);plt.close(fig)
    j=dict(records=380,declared_bad_records=int((count>0).sum()),bad_min=int(count.min()),bad_max=int(count.max()),
        EEG_length_seconds_min_max=[float(raw_seconds.min()),float(raw_seconds.max())],
        EEG_minus_audio_seconds_min_max=[float((raw_seconds-audio_seconds).min()),float((raw_seconds-audio_seconds).max())],
        full_EEG_payloads_read=1,full_EEG_payloads_NOT_read=379,
        figures=['audio_structure_v2.png','header_layout_overview.png'],
        corrected_display='v1 audio fraction axis0..1 hid sparse differences; v1 retained, v2 shows percentage0..0.3',NN_forwards=0)
    with (ROOT/'OVERVIEW_FIGURES.json').open('x') as f:json.dump(j,f,indent=2)
    print(json.dumps(j))


if __name__=='__main__':main()
