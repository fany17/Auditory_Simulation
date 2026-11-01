"""Complete-matrix warm train/val reporting on2203, never fits/selects/tests.

Separate post-freeze reporter; does not mutate any frozen training source.
All 18 cells are required, every seed and paired difference is retained.
"""
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pub_01_s03_ss07_warm import ROOT,SEEDS,CONDITIONS,read,dump,sha

def table(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def verify_continuous(value):
    c=value['continuous'];assert c['subjects']==49 and c['blocks']==752
    segments=c['segments'];assert len(segments)==752
    assert len({(r['record'],r['start']) for r in segments})==752
    assert all(np.isfinite(r['r']) for r in segments)
    people={r['subject']:r['r'] for r in c['people']};assert len(people)==49
    assert set(people)=={r['subject'] for r in segments}
    maxdiff=0.
    for person,r in people.items():
        fresh=float(np.mean([s['r'] for s in segments if s['subject']==person]))
        maxdiff=max(maxdiff,abs(fresh-r))
    assert maxdiff<=1e-12
    assert abs(float(np.mean(list(people.values())))-c['macro_r'])<=1e-12
    return maxdiff

def main():
    finished=read(ROOT/'TRAIN_VAL_COMPLETED.json');assert finished['new_fits']==18 and not finished['test_read']
    protocol=read(ROOT/'WARM_PROTOCOL.json');assert protocol['new_fits']==18 and not protocol['test_operations']
    for p,d in protocol['sources'].items():assert sha(p)==d,p
    out=ROOT/'results01';out.mkdir(exist_ok=False)
    fits=[];people=[];epoch_rows=[];lookup={};inputs={};maxperson=0.
    for condition in CONDITIONS:
        for seed in SEEDS:
            folder=ROOT/f'fits/{condition}_seed{seed}'
            for name in ['completed.json','history.json','INCUMBENT_VALIDATION.json','SELECTED_VALIDATION.json','resolved_config.json']:
                inputs[str(folder/name)]=sha(folder/name)
            done=read(folder/'completed.json');history=read(folder/'history.json')
            initial=read(folder/'INCUMBENT_VALIDATION.json');selected=read(folder/'SELECTED_VALIDATION.json')
            assert done['condition']==condition and done['seed']==seed and done['test_read'] is False
            assert done['epochs']==5 and done['updates']==19725 and len(history)==5
            assert done['supervised_window_presentations']==1262270 and done['validation_full_passes']==7
            assert initial['windows']==selected['windows']==30500
            best=initial['negative_Pearson'];bestepoch=-1
            for e,r in enumerate(history):
                assert r['epoch']==e and r['updates']==(e+1)*3945 and r['record_starts']==426
                if r['validation_negative_Pearson']<best:best=r['validation_negative_Pearson'];bestepoch=e
                assert r['selected_epoch_so_far']==bestepoch
                epoch_rows.append({'condition':condition,'seed':seed,**r})
            assert done['selected_epoch']==bestepoch and abs(done['best_validation_negative_Pearson']-best)<=1e-12
            assert abs(selected['negative_Pearson']-best)<=1e-7
            assert sha(folder/'best_validation.pt')==done['checkpoint_sha256']
            cp=torch.load(folder/'best_validation.pt',map_location='cpu',weights_only=True)
            assert cp['condition']==condition and cp['seed']==seed and cp['epoch']==bestepoch
            del cp
            for name,value in [('incumbent',initial),('selected',selected)]:
                maxperson=max(maxperson,verify_continuous(value))
                for person in value['continuous']['people']:
                    people.append({'condition':condition,'seed':seed,'checkpoint':name,**person})
            assert abs(done['selected_val60s_macro_r']-selected['continuous']['macro_r'])<=1e-12
            row={'condition':condition,'seed':seed,'encoder_mode':condition.split('_')[0],'selected_epoch':bestepoch,
                'incumbent_val3s_r':-initial['negative_Pearson'],'selected_val3s_r':-selected['negative_Pearson'],
                'incumbent_val60s_r':initial['continuous']['macro_r'],'selected_val60s_r':selected['continuous']['macro_r'],
                'gain_val60s':selected['continuous']['macro_r']-initial['continuous']['macro_r'],
                'updates':19725,'window_presentations':1262270,'seconds':done['seconds'],
                'trainable_parameters':done['trainable_parameters'],'validation_full_passes':7}
            fits.append(row);lookup[(condition,seed)]=row
    assert len(fits)==18 and len(epoch_rows)==90
    pairs=[];averages=[]
    for mode in ('frozen','joint'):
        for seed in SEEDS:
            p=lookup[(mode+'_p',seed)];past=lookup[(mode+'_past',seed)];current=lookup[(mode+'_current',seed)]
            for comp,base3,base60 in [('continued_P',p['selected_val3s_r'],p['selected_val60s_r']),
                ('current',current['selected_val3s_r'],current['selected_val60s_r']),
                ('original_P_incumbent',past['incumbent_val3s_r'],past['incumbent_val60s_r'])]:
                pairs.append({'encoder_mode':mode,'seed':seed,'contrast':'past_minus_'+comp,
                    'VAL3s_difference':past['selected_val3s_r']-base3,'VAL60s_difference':past['selected_val60s_r']-base60})
        for kind in ('p','current','past'):
            group=[lookup[(mode+'_'+kind,s)] for s in SEEDS]
            averages.append({'condition':mode+'_'+kind,'VAL3s_mean':float(np.mean([r['selected_val3s_r'] for r in group])),
                'VAL60s_mean':float(np.mean([r['selected_val60s_r'] for r in group])),
                'selected_epochs':[r['selected_epoch'] for r in group]})
    summary={'time':time.time(),'protocol_sha256':sha(ROOT/'WARM_PROTOCOL.json'),'report_source_sha256':sha(Path(__file__)),
        'inputs':inputs,'cells':18,'seeds':list(SEEDS),'condition_means':averages,'paired_differences':pairs,
        'costs':{'epochs':90,'updates':sum(r['updates'] for r in fits),'supervised_window_presentations':sum(r['window_presentations'] for r in fits),
            'target_time_sample_presentations':sum(r['window_presentations']*192 for r in fits),
            'elapsed_seconds_sum':sum(r['seconds'] for r in fits),'validation_full_passes':126,
            'per_fit_elapsed_excludes_model_and_data_construction':True,
            'queue_wall_seconds':finished['time']-read(ROOT/'QUEUE_LEASE.json')['time'],
            'shared_upstream_training_reused_not_recounted_as_new':True,'engineering_costs_separate':True},
        'review':{'all_budgets_and_earliest_incumbent_selection_pass':True,'saved_segment_to_person_max_difference':maxperson,
            'selected_checkpoint_roles_and_hashes_verified':True,'independent_raw_neural_replay':False},
        'scope':'source TRAIN_VAL_DEVELOPMENT only, no tests, no calibrated significance or generalization claim; row mode also changes dropout',
        'numerical_boundary':'old/new P approx1e-8 differences are implementation rounding, not scientific gain',
        'scientific_goal_achieved_claimed':False,'new_fits_requested':False}
    assert summary['costs']['updates']==355050 and summary['costs']['supervised_window_presentations']==22720860
    dump(out/'SUMMARY.json',summary);table(out/'FIT_RESULTS.csv',fits);table(out/'VAL_COMPARISONS.csv',pairs)
    table(out/'VAL_PER_PERSON.csv',people);table(out/'EPOCH_HISTORY.csv',epoch_rows)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#0072B2','#D55E00','#009E73'];fig,axes=plt.subplots(2,2,figsize=(12,8))
    for col,mode in enumerate(('frozen','joint')):
        ax=axes[0,col];labels=['Original P\nincumbent','Continued P','M4 current','M4 past']
        for idx,seed in enumerate(SEEDS):
            y=[lookup[(mode+'_p',seed)]['incumbent_val60s_r']]+[lookup[(mode+'_'+kind,seed)]['selected_val60s_r'] for kind in ('p','current','past')]
            ax.plot(range(4),y,'o-',color=colors[idx],alpha=.85,label=str(seed))
        ax.set_xticks(range(4),labels);ax.set_ylabel('VAL 60s/person r');ax.set_title(mode+' encoder: all paired seeds');ax.legend(title='seed')
        ax=axes[1,col];keys=['past_minus_continued_P','past_minus_current','past_minus_original_P_incumbent']
        for idx,seed in enumerate(SEEDS):
            y=[next(r['VAL60s_difference'] for r in pairs if r['encoder_mode']==mode and r['seed']==seed and r['contrast']==key) for key in keys]
            ax.scatter(np.arange(3)+(idx-1)*.08,y,color=colors[idx],label=str(seed))
        means=[float(np.mean([r['VAL60s_difference'] for r in pairs if r['encoder_mode']==mode and r['contrast']==k])) for k in keys]
        ax.scatter(range(3),means,color='black',marker='_',s=250,label='3-seed mean')
        ax.axhline(0,color='grey',lw=1);ax.set_xticks(range(3),['Past - continued P','Past - current','Past - original P']);ax.set_ylabel('Paired VAL r difference');ax.set_title('Descriptive only; no confidence intervals')
    fig.suptitle('Warm round02: fixed 18-cell source validation (no test evaluation)');fig.tight_layout();fig.savefig(out/'01_VAL_ABSOLUTE_PAIRED.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,8),sharex=True)
    for i,mode in enumerate(('frozen','joint')):
        for j,kind in enumerate(('p','current','past')):
            ax=axes[i,j];condition=mode+'_'+kind
            for idx,seed in enumerate(SEEDS):
                rs=[r for r in epoch_rows if r['condition']==condition and r['seed']==seed];r=lookup[(condition,seed)]
                vals=[r['incumbent_val3s_r']]+[-x['validation_negative_Pearson'] for x in rs]
                ax.plot(range(6),vals,'o-',ms=3,color=colors[idx],label=str(seed))
                ax.plot(range(1,6),[-x['train_negative_Pearson'] for x in rs],'--',lw=1,color=colors[idx],alpha=.6)
                selected_index=r['selected_epoch']+1;ax.scatter([selected_index],[vals[selected_index]],s=75,facecolors='none',edgecolors=colors[idx],marker='s')
            ax.set_title(condition);ax.set_ylabel('3s Pearson r');ax.set_xlabel('0 = trained-P incumbent; 1-5 = new epochs');ax.set_xticks(range(6))
    axes[0,0].legend(title='seed; solid VAL / dashed online train');fig.suptitle('Actual complete histories; square = original 3s checkpoint choice');fig.tight_layout();fig.savefig(out/'02_ALL_TRAIN_VAL_CURVES.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    xs=np.arange(6);labels=list(CONDITIONS)
    for i,seed in enumerate(SEEDS):
        axes[0].bar(xs+(i-1)*.22,[lookup[(c,seed)]['updates'] for c in labels],.22,color=colors[i],label=str(seed))
        axes[1].bar(xs+(i-1)*.22,[lookup[(c,seed)]['seconds']/60 for c in labels],.22,color=colors[i])
    for ax in axes:ax.set_xticks(xs,labels,rotation=25,ha='right')
    axes[0].set_ylabel('New optimizer updates per fit');axes[0].legend(title='seed')
    axes[1].set_ylabel('Per-fit elapsed minutes (shared GPUs)')
    fig.suptitle('All 18 new fits; shared upstream and engineering separate; elapsed is not isolated latency');fig.tight_layout();fig.savefig(out/'03_COMPLETE_COSTS.png',dpi=170);plt.close(fig)
    lines=['# Warm round02全格源验证数值（非测试）','','全部18fit、90epoch、355050更新；以下均为train/val开发，不能作确认性结果。',
        '','| condition | 三seed VAL3s均值 | VAL60s/person均值 | selected 0-based epoch（−1为原P） |','|---|---:|---:|---|']
    for r in averages:lines.append(f"| {r['condition']} | {r['VAL3s_mean']:.9f} | {r['VAL60s_mean']:.9f} | {r['selected_epochs']} |")
    lines+=['','所有seed与配对差见CSV；不按单seed或途中分数重选，不计算新测试或显著性。原/新P约1e−8差属于实现数值边界。',
        '','frozen/joint同时改变train/eval随机行为，不是纯encoder梯度效应；warm与旧随机起点预算/上游不同，不是初始化因果消融。',
        '','![全部源VAL](01_VAL_ABSOLUTE_PAIRED.png)','','![实际完整曲线](02_ALL_TRAIN_VAL_CURVES.png)','','![完整新增成本](03_COMPLETE_COSTS.png)']
    (out/'NUMERICAL_SUMMARY.md').write_text('\n'.join(lines)+'\n')
    dump(out/'COMPLETED.json',{'time':time.time(),'all18_reported':True,'figures':3,'test_read':False,'scientific_goal_achieved_claimed':False})
    print('WARM_FULL_MATRIX_REPORTED',flush=True)

if __name__=='__main__':main()
