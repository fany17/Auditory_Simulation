"""Metadata fixtures ONLY. They are NOT experimental scores/results."""
import copy
import json
from pathlib import Path
import tempfile
import pub_01_s03_ss07_select as select
from pub_01_s03_ss07_baseline import ROOT,read,dump,sha


def cell(root,stage,config,seed,value=.1):
    folder=root/f'{stage}/{config["id"]}_seed{seed}';folder.mkdir(parents=True)
    dump(folder/'completed.json',{'condition':config,'seed':seed,'stage':stage,'test_read':False,
        'protocol_sha256':sha(root/'INCREMENTAL_PROTOCOL.json'),'updates':3945,'epochs':1,
        'validation_macro':{'macro_r':value,'subjects':49,'blocks':752},
        'best_validation_negative_Pearson':-.1,'selected_epoch':0,'seconds':1,'supervised_target_samples':1})
    dump(folder/'validation_macro.json',{'checkpoint_selection':'original fullval3s fixture',
        'people':[{}]*49,'segments':[{}]*752})


def main():
    actual=select.ROOT;protocol=read(ROOT/'INCREMENTAL_PROTOCOL.json');tests=[]
    # Retain fixtures under engineering provenance, never the scientific runs.
    root=Path(tempfile.mkdtemp(prefix='selection_fixture_',dir=ROOT/'engineering_history'))
    select.ROOT=root;dump(root/'INCREMENTAL_PROTOCOL.json',protocol)
    configs=protocol['conditions'];seeds=protocol['screening_seeds']
    first=configs[0];cell(root,'screen',first,seeds[0])
    _,means,missing,failed=select.collect('screen',[first],seeds)
    assert not means and len(missing)==2 and not failed;tests.append('missing seeds never averaged')
    with (root/'SCREEN_FAILURES.jsonl').open('x') as f:
        f.write(json.dumps({'condition':first['id'],'seed':seeds[1]})+'\n')
    _,means,missing,failed=select.collect('screen',[first],seeds)
    assert not means and len(missing)==1 and len(failed)==1;tests.append('failed seed explicitly detected')
    # Separate complete fixture so failed-run history is not erased.
    root2=Path(tempfile.mkdtemp(prefix='selection_complete_fixture_',dir=ROOT/'engineering_history'))
    select.ROOT=root2;dump(root2/'INCREMENTAL_PROTOCOL.json',protocol)
    for c in configs:
        for seed in seeds:cell(root2,'screen',c,seed)
    select.screen_selection();chosen=read(root2/'FULL_SELECTION.json')
    assert chosen['selected_family_condition_ids']==[f'm{i}_h16' for i in range(1,6)]
    tests.append('all54 required; exact tie selects smaller width')
    catalog={c['id']:c for c in protocol['full_conditions']}
    for id_ in chosen['selected_family_condition_ids']:
        for seed in seeds:cell(root2,'full',catalog[id_],seed)
    select.primary_selection();final=read(root2/'FULL_PRIMARY_SELECTION.json')
    assert final['primary']=='m1_h16' and final['final_seeds']==[1024,2026,31415,42,12345]
    assert final['same_full_budget_control_ids']==['m1_current_h16','static_m1_h16','output_prior_h16']
    tests.append('all15 full cells; family tie rule; fixed five seeds and deterministic controls')
    original=(root2/'FULL_PRIMARY_SELECTION.json').read_bytes();select.primary_selection()
    assert (root2/'FULL_PRIMARY_SELECTION.json').read_bytes()==original;tests.append('frozen child idempotent')
    root3=Path(tempfile.mkdtemp(prefix='selection_bad_fixture_',dir=ROOT/'engineering_history'))
    select.ROOT=root3;dump(root3/'INCREMENTAL_PROTOCOL.json',protocol)
    cell(root3,'screen',first,seeds[0],float('nan'))
    try:select.collect('screen',[first],seeds[:1])
    except AssertionError:tests.append('nonfinite macro rejects condition')
    else:raise AssertionError('nonfinite accepted')
    select.ROOT=actual
    dump(ROOT/'SELECTION_ENGINEERING_CHECK.json',{'status':'PASS','synthetic_metadata_only':True,
        'tests':tests,'test_read':False,'fixture_paths':[str(root),str(root2),str(root3)]})
    print(json.dumps({'status':'PASS','tests':tests}))


if __name__=='__main__':main()
