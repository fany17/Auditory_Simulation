"""Read-only D18 seed1024 incumbent metric audit; server2203 JSON only."""
from pathlib import Path
import json,gzip,hashlib,time,math,resource
ROOT=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_preprojection_memory_20261103')
OLD=ROOT.parent/'s03_full_ft_token_normalization_20261102'
def read(path):
    path=Path(path)
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:
        return json.load(f)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def run():
    start=time.time();cpu=time.process_time()
    old_path=OLD/'fits/H_full_seed1024/INCUMBENT.json'
    paths=[ROOT/f'fits/{arm}_seed1024/INCUMBENT.json.gz' for arm in ('H_history','C_current')]
    old=read(old_path);new=[read(p) for p in paths]
    assert len(old['people'])==49 and len(old['segments'])==752 and len(old['records'])==49
    count={'numeric':0,'metadata':0};max_abs=0.
    excluded={'source_family','family_mapping_basis'}
    def compare(a,b,path):
        nonlocal max_abs
        if isinstance(a,dict):
            assert isinstance(b,dict),path
            for key in a:
                if key not in excluded:
                    assert key in b,(path,key)
                    compare(a[key],b[key],path+'/'+key)
        elif isinstance(a,list):
            assert isinstance(b,list) and len(a)==len(b),path
            for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i))
        elif isinstance(a,(int,float)) and not isinstance(a,bool):
            assert isinstance(b,(int,float)) and math.isfinite(a) and math.isfinite(b),path
            d=abs(a-b);max_abs=max(max_abs,d);count['numeric']+=1
            assert d==0,(path,a,b)
        else:
            assert a==b,(path,a,b);count['metadata']+=1
    roots=('phases','weighted','people','segments','records','Hann_identity_max_abs','boundary_tail_records')
    per_arm=[]
    family_rows=read(ROOT.parent/'s03_incremental_continuity_20261101/SOURCE_FAMILY_JOIN.json')['records']
    family={(x['phase'],x['record']):x for x in family_rows}
    for path,current in zip(paths,new):
        before=dict(count)
        for key in roots:compare(old[key],current[key],key)
        for record in current['records']:
            expected=family[('val',record['record'])]
            assert record['source_family']==expected['source_family']
            assert record['family_mapping_basis']==expected['family_mapping_basis']
            for segment in record['segments']:
                assert segment['source_family']==expected['source_family']
                assert segment['family_mapping_basis']==expected['family_mapping_basis']
        family_error=0.
        for row in current['source_family_descriptive']:
            blocks=[b for b in current['segments'] if b['source_family']==row['source_family']]
            subjects=sorted({b['subject'] for b in blocks})
            means=[math.fsum(b['r'] for b in blocks if b['subject']==s)/sum(b['subject']==s for b in blocks) for s in subjects]
            value=math.fsum(means)/len(means)
            assert row['people']==len(subjects) and row['blocks']==len(blocks)
            family_error=max(family_error,abs(value-row['r']))
        assert len(current['source_family_descriptive'])==5 and family_error<=1e-12
        per_arm.append(dict(path=str(path),sha256=sha(path),counts={k:count[k]-before[k] for k in count},source_family_descriptive_max_abs=family_error))
    assert new[0]==new[1],'The two initial incumbent JSON objects must match in every field.'
    result=dict(passed=True,scope='Seed1024 H/C initial frozen-Ccap metrics only; not trained predictions, not all seeds, and not an independent neural replay.',
        old_path=str(old_path),old_sha256=sha(old_path),new=per_arm,
        covered_roots=list(roots),numeric_scalar_comparisons=count['numeric'],metadata_scalar_comparisons=count['metadata'],max_abs_metric_difference=max_abs,
        records=49,people=49,blocks=752,H_C_all_fields_exact=True,
        intentional_exclusions={'source_family':'Old story label replaced with previously frozen source-family join; independently checked.',
                                'family_mapping_basis':'New provenance annotation, independently checked against frozen join.'},
        source_family_join_sha256=sha(ROOT.parent/'s03_incremental_continuity_20261101/SOURCE_FAMILY_JOIN.json'),
        audit_source_sha256=sha(Path(__file__)),new_NN=0,new_backward=0,new_optimizer_updates=0,new_scientific_array_values_read=False,
        CPU_seconds=time.process_time()-cpu,wall_seconds=time.time()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        no_training_or_selection_change=True,goal_complete=False)
    output=ROOT/'COORDINATOR_INCUMBENT_METRIC_AUDIT.json'
    previous=read(output)
    assert 'boundary_tail' in previous['intentional_exclusions'] and previous['numeric_scalar_comparisons']==20238
    result['superseded_draft_audit']=dict(sha256=sha(output),CPU_seconds=previous['CPU_seconds'],wall_seconds=previous['wall_seconds'],new_NN=0,new_backward=0,new_optimizer_updates=0,correction='The initial draft excluded boundary diagnostics based on an unverified assumption that their coordinates differed. Inspection showed matching definitions; this revision actually compares all boundary values as well. The original compared metrics and their zero differences remain valid.')
    with output.open('w') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('passed','numeric_scalar_comparisons','metadata_scalar_comparisons','max_abs_metric_difference','H_C_all_fields_exact','CPU_seconds','wall_seconds')}))
if __name__=='__main__':run()
