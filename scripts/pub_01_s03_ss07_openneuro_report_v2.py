"""Actual-result CSV/static figures/Chinese draft. No new NN or parameter fitting."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pub_01_s03_ss07_openneuro_transport import ROOT

SEEDS=(1024,2026,31415);ROLES=('P_FT_star','Ccap','W','F');COLORS=['#6c757d','#457b9d','#e9a23b','#b23a48']


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def number(x):return 'undefined' if x is None else f'{x:.8f}'
def write_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def cost_ledger():
    base=ROOT.parent;prior=base/'s03_incremental_continuity_20261101';entries=[]
    def add(stage,path,inherited=False,wall_key='wall_seconds',note=None):
        value=read(path)
        entry=dict(stage=stage,inherited=inherited,manifest=str(path),manifest_sha256=sha(path),
            wall_seconds=value.get(wall_key),CPU_seconds=value.get('CPU_seconds'),
            peak_RSS_KiB=value.get('peak_RSS_KiB'),peak_GPU_bytes=value.get('peak_GPU_bytes',value.get('peak_GPU_allocated_bytes')),note=note,
            wall_measurement_key=wall_key)
        for key in ['epochs','updates','selected_epoch','primary_encoder_head_windows','extra_bridge_encoder_head_windows',
                    'new_ridge_fits','NN_updates','solves','reused_v1_records','new_v2_records',
                    'unlabeled_record_gain_statistics','total_elapsed_since_v1_seconds','main_encoder_head_windows',
                    'extra_bridge_windows','new_model_fits','input_record_gain_statistics_reused']:
            if key in value:entry[key]=value[key]
        entry['missing_measurements']=[k for k in ['wall_seconds','CPU_seconds','peak_RSS_KiB','peak_GPU_bytes'] if entry[k] is None]
        entries.append(entry);return value
    for label,start,end in [('Header attempt01','HEADER_ACQUISITION_STARTED.json','HEADER_ATTEMPT01_TRANSPORT_SWITCH.json'),
                            ('Audio/EEG prefetch attempt01','ACQUISITION_PREFLIGHT_STARTED.json','PREFLIGHT_ATTEMPT01_TRANSPORT_SWITCH.json'),
                            ('EEG preprocessing retained v1','EEG_PREPARE_STARTED.json','PREPROCESSING_TRANSPORT_HANDOFF.json')]:
        a,b=ROOT/start,ROOT/end
        entries.append(dict(stage=label,inherited=False,wall_seconds=read(b)['time']-read(a)['time'],
            measurement='Elapsed start-to-voluntary-interruption timestamps; not CPU or active compute time',
            references=[dict(path=str(p),sha256=sha(p)) for p in (a,b)],
            CPU_seconds=None,peak_RSS_KiB=None,peak_GPU_bytes=None,
            missing_measurements=['CPU_seconds','peak_RSS_KiB','peak_GPU_bytes']))
    for label,name in [('Header acquisition v2','HEADER_ACQUISITION_V2_COMPLETED.json'),
                       ('Audio/EEG prefetch v2 including five reused verified WAVs','ACQUISITION_PREFLIGHT_V2_COMPLETED.json'),
                       ('Twenty fixed gammatone envelopes','AUDIO_PREPARE_COMPLETED.json'),
                       ('EEG acquisition/preprocessing v2; sixteen v1 outputs reused','EEG_PREPARE_COMPLETED_V2.json')]:
        add(label,ROOT/name,note='total_elapsed_since_v1_seconds is an enclosing interval, not additive to either preprocessing phase' if name.startswith('EEG_') else None)
    for seed in SEEDS:
        add(f'External seed{seed} inference/readout/scoring',ROOT/f'evaluation_seed{seed}/COMPLETED.json')
        for phase,folder,key in [('Native pretraining','baseline/pretrain_seed','seconds'),
                                 ('Native supervised head including cache','baseline/head_seed','seconds_including_cache'),
                                 ('Native strong finetune','finetune/native_P_seed','seconds')]:
            add(f'Inherited {phase} seed{seed}',prior/f'{folder}{seed}/completed.json',True,key)
    add('Inherited D8 source FT cache/calibration/Ccap selection and 27 ridge fits',base/'s03_finetuned_readout_transfer_20261101/COST.json',True,
        note='Full stage-level numeric costs are in referenced manifest; enclosing wall time, not sum of stages')
    d9=add('Inherited D9 fixed physical W/F readouts, six solves',base/'s03_physical_objective_20261101/COMPLETED.json',True)
    for worker in d9['workers']:
        entries.append(dict(stage=f"Inherited D9 worker seed{worker['seed']}",inherited=True,
            included_in='Inherited D9 fixed physical W/F readouts, six solves',wall_seconds=worker['wall_seconds'],
            CPU_seconds=worker.get('CPU_seconds'),peak_RSS_bytes=worker.get('peak_RSS_bytes'),
            peak_GPU_allocated_bytes=worker.get('peak_GPU_allocated_bytes'),solves=worker['solves']))
    entries.append(dict(stage='Inherited upstream XLSR pretraining and original TRAIN normalization preparation',inherited=True,
        wall_seconds=None,CPU_seconds=None,note='Costs not measured in these run manifests; unknown, not zero. Source/model identity pinned in SOURCE_ASSETS_V2.json and EVALUATION_PROTOCOL.json',
        missing_measurements=['wall_seconds','CPU_seconds','peak_RSS','peak_GPU'],
        references=[dict(path=str(ROOT/p),sha256=sha(ROOT/p)) for p in ['SOURCE_ASSETS_V2.json','EVALUATION_PROTOCOL.json']]))
    return dict(entries=entries,parallel_wall_times_not_summed=True,missing_is_not_zero=True,
        inherited_training_not_new_work=True,scope='Three inherited seeds used here; not a complete all-SS07 research cost ledger',
        additional_engineering_and_review_overhead='Not fully instrumented; no total elapsed-compute claim')


def freeze():
    assert not any((ROOT/f'evaluation_seed{s}/STARTED.json').exists() for s in SEEDS)
    value=dict(source_sha256=sha(__file__),frozen_before_any_real_NN=True,
        figures=['person_and_seed_performance','two_primary_differences','window_and_fused_error',
            'full_FP32_sensitivity','preselected_example'],example=dict(seed=2026,canonical_record_index=0,seconds=[0,60]),
        no_gain_or_axis_normalization_for_example=True,example_y_axis='Shared original numeric min/max across target and all four predictions; no clipping',
        cost_ledger='Acquisition attempts/audio/preprocessing/inference plus separately inherited three-seed training and D8/D9; missing measurements explicit; no summed parallel wall time',
        supersedes_presentation_only='PRESENTATION_PROTOCOL.json',all_seed_and_person_rows=True,new_NN=0,
        interpretation='Offline/transductive unlabeled whole-record scalar calibration, not SI repair or uncalibrated confirmation')
    with (ROOT/'PRESENTATION_PROTOCOL_V2.json').open('x') as f:json.dump(value,f,indent=2)
    print('PRESENTATION_FROZEN_NO_NN')


def run():
    presentation=read(ROOT/'PRESENTATION_PROTOCOL_V2.json');assert presentation['source_sha256']==sha(__file__)
    summary=read(ROOT/'EXTERNAL_SUMMARY.json');protocol=read(ROOT/'EVALUATION_PROTOCOL.json')
    output=ROOT/'report01';output.mkdir(exist_ok=False);figures=output/'figures';figures.mkdir()
    people=[];seed_rows=[];metric_files={};precision=[]
    expected_subjects=sorted({r['subject'] for r in protocol['rows']})
    for role in ROLES:
        for seed in SEEDS:
            j=read(ROOT/f'evaluation_seed{seed}/{role}_METRICS.json');metric_files[role,seed]=j
            blocks=[b for r in j['records'] for b in r['blocks']]
            subject_values=[]
            for subject in expected_subjects:
                selected=[b for b in blocks if b['subject']==subject]
                valid=len(selected)==52 and all(b['r'] is not None for b in selected)
                r=float(np.mean([b['r'] for b in selected])) if valid else None
                person=dict(subject=subject,seed=seed,role=role,r=r,blocks=len(selected),MSE=float(np.mean([b['MSE'] for b in selected])) if selected else None)
                people.append(person);subject_values.append(r)
            windows=sum(r['windows'] for r in j['records']);undefined=sum(r['undefined_window_r'] for r in j['records'])
            seed_rows.append(dict(role=role,seed=seed,r=float(np.mean(subject_values)) if all(v is not None for v in subject_values) else None,
                MSE=float(np.mean([p['MSE'] for p in people if p['role']==role and p['seed']==seed])) if j['passed'] else None,
                window_MSE=sum(r['window_MSE']*r['windows'] for r in j['records'])/windows if windows else None,
                r3s=sum(r['r3s_sum_defined'] for r in j['records'])/windows if windows and not undefined else None,
                undefined_3s_windows=undefined,primary_complete=j['passed']))
            if role in ('W','F'):
                for item in j['precision_diagnostics']:
                    precision.append(dict(seed=seed,role=role,record=item['record'],wave_max=item['whole_record_waveform_max_abs'],
                        whole_record_MSE32=item['fused_MSE32'],whole_record_MSE64=item['fused_MSE64']))
    write_csv(output/'PERSON_BY_SEED.csv',people);write_csv(output/'SEED_SUMMARY.csv',seed_rows)
    if precision:write_csv(output/'PRECISION_RECORDS.csv',precision)
    model={r['role']:r for r in summary['models']}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):fig.savefig(figures/name,dpi=160,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for i,role in enumerate(ROLES):
        values=[]
        for subject in expected_subjects:
            rr=[p['r'] for p in people if p['role']==role and p['subject']==subject]
            if all(v is not None for v in rr):values.append(np.mean(rr))
        ax.scatter(i+np.linspace(-.13,.13,len(values)),values,color=COLORS[i],s=20,alpha=.45)
        seeds=[r['r'] for r in seed_rows if r['role']==role and r['r'] is not None]
        ax.scatter(i+np.linspace(-.08,.08,len(seeds)),seeds,color='black',marker='^',s=36)
        if model[role]['r'] is not None:ax.scatter([i],[model[role]['r']],color=COLORS[i],marker='D',s=90,zorder=5)
    ax.set(xticks=range(4),xticklabels=ROLES,ylabel='Person-macro block Pearson r',
        title='19 people (dots), 3 frozen-seed means (triangles), overall mean (diamond)')
    save(fig,'person_and_seed_performance.png')
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for i,item in enumerate(summary['primary']):
        test=item['inference']
        if test:
            mean=test['mean_difference'];low,high=test['Bonferroni_family95_approximate_CI']
            axes[0].errorbar(mean,i,xerr=np.array([[mean-low],[high-mean]]),fmt='o',capsize=4,color=COLORS[i+1])
            axes[0].annotate(f"Holm2 p={test['p_Holm2']:.5g}",(mean,i),xytext=(0,13),textcoords='offset points',ha='center',fontsize=9)
        else:axes[0].text(0,i,'Undefined',ha='center')
    axes[0].axvline(0,color='black',lw=.8);axes[0].set(yticks=[0,1],yticklabels=['Ccap - P_FT*','F - W'],
        xlabel='Difference in person-macro r',ylim=(-.5,1.6),title='Two fixed primary contrasts; family95 approximate CIs')
    if all(item['person_differences'] is not None for item in summary['primary']):
        a={r['subject']:r['difference'] for r in summary['primary'][0]['person_differences']}
        b={r['subject']:r['difference'] for r in summary['primary'][1]['person_differences']}
        axes[1].scatter([a[s] for s in expected_subjects],[b[s] for s in expected_subjects],s=35,color='#457b9d')
    axes[1].axhline(0,color='black',lw=.8);axes[1].axvline(0,color='black',lw=.8)
    axes[1].set(xlabel='Ccap - P_FT*: person difference',ylabel='F - W: person difference',title='Each dot is one person, averaged over three seeds')
    save(fig,'two_primary_differences.png')
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained');x=np.arange(4)
    for offset,key,label in [(-.22,'window_MSE','Unweighted window MSE'),(0,'weighted_single_MSE','Weighted-single physical MSE'),(.22,'MSE','Fused physical block/person MSE')]:
        values=[model[r].get(key) for r in ROLES]
        axes[0].bar(x+offset,[np.nan if v is None else v for v in values],width=.21,label=label)
    axes[0].set(xticks=x,xticklabels=ROLES,yscale='log',ylabel='MSE (fixed envelope numeric units squared)',title='Window and fused errors have different coordinates');axes[0].legend(fontsize=7)
    axes[1].bar(x,[np.nan if model[r]['r3s'] is None else model[r]['r3s'] for r in ROLES],color=COLORS)
    axes[1].set(xticks=x,xticklabels=ROLES,ylabel='Mean 3-second window r',title='Undefined windows remain counted, not silently removed')
    save(fig,'window_and_fused_error.png')
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    precision_blocks=[]
    for i,role in enumerate(('W','F')):
        rr=[p for p in precision if p['role']==role]
        if rr:axes[0].scatter(i+np.linspace(-.13,.13,len(rr)),[p['wave_max'] for p in rr],s=10,alpha=.25,color=COLORS[i+2])
        differences=[]
        for subject in expected_subjects:
            values=[]
            for seed in SEEDS:
                j=metric_files[role,seed];lookup={r['record']:r['subject'] for r in j['records']}
                bb=[b for p in j['precision_diagnostics'] if lookup[p['record']]==subject for b in p['blocks']]
                if len(bb)==52 and all(b['r32'] is not None and b['r64'] is not None for b in bb):values.append(np.mean([b['r32']-b['r64'] for b in bb]))
            if len(values)==3:differences.append(np.mean(values));precision_blocks.append(dict(role=role,subject=subject,FP32_minus_FP64_r=float(np.mean(values))))
        axes[1].scatter(i+np.linspace(-.13,.13,len(differences)),differences,s=25,color=COLORS[i+2])
    axes[0].set(xticks=[0,1],xticklabels=['W','F'],yscale='log',ylabel='Max |fused32 - fused64| per record',title='Full FP32 readout + FP32 Hann; same frozen Z/P')
    axes[1].axhline(0,color='black',lw=.8);axes[1].set(xticks=[0,1],xticklabels=['W','F'],ylabel='Person r32 - r64, averaged over 3 seeds',title='Precision sensitivity is descriptive; FP64 remains primary')
    save(fig,'full_FP32_sensitivity.png')
    if precision_blocks:write_csv(output/'PRECISION_PERSON.csv',precision_blocks)
    # Preselected metadata example, never chosen for an attractive score.
    import pub_01_s03_ss07_openneuro_evaluate_v3 as evaluator
    import pub_01_s03_ss07_openneuro_scoring as scoring
    row=protocol['rows'][0];seed=2026;cache=read(ROOT/f'evaluation_seed{seed}/CACHE_PATHS.json')
    z=np.load(cache['Z'],mmap_mode='r');p=np.load(cache['P'],mmap_mode='r');a=row['offset'];b=a+row['windows']
    cal,stats,co=evaluator.frozen(seed);target=np.load(row['target_path']);timeaxis=np.arange(3840)/64
    fig,axes=plt.subplots(4,1,figsize=(11,7),sharex=True,sharey=True,layout='constrained')
    example_values=[target[:3840]]
    for i,role in enumerate(ROLES):
        pred=scoring.readout(z[a:b],p[a:b],cal,stats,co.get(role));wave,keep=scoring.fuse(pred,row['length'])
        view=wave[:3840].copy();view[~keep[:3840]]=np.nan
        example_values.append(view)
        axes[i].plot(timeaxis,target[:3840],color='black',lw=.7,label='Target')
        axes[i].plot(timeaxis,view,color=COLORS[i],lw=.8,label=role)
        axes[i].set(ylabel='Envelope a.u.');axes[i].legend(loc='upper right',fontsize=8)
    finite=np.concatenate([v[np.isfinite(v)] for v in example_values]);assert finite.size
    low,high=float(finite.min()),float(finite.max());pad=(high-low)*.03 if high>low else max(abs(high)*.03,1e-9)
    axes[0].set_ylim(low-pad,high+pad)
    with (output/'EXAMPLE_AXES.json').open('x') as f:json.dump(dict(record=row['record'],seed=seed,seconds=[0,60],original_min=low,original_max=high,shared_ylim=[low-pad,high+pad],no_clipping=True),f,indent=2)
    axes[-1].set(xlabel='Time (s)');fig.suptitle(f"Preselected {row['record']}, seed2026: actual unrescaled first60s")
    save(fig,'preselected_example.png')
    table='\n'.join(f"|{r['role']}|{r['seed']}|{number(r['r'])}|{number(r['MSE'])}|{number(r['r3s'])}|{number(r['window_MSE'])}|" for r in seed_rows)
    conclusions=[];directions=[]
    for a,b in [('Ccap','P_FT_star'),('F','W')]:
        values=[]
        for seed in SEEDS:
            aa=next(r['r'] for r in seed_rows if r['seed']==seed and r['role']==a);bb=next(r['r'] for r in seed_rows if r['seed']==seed and r['role']==b)
            values.append(None if aa is None or bb is None else aa-bb)
        directions.append(dict(comparison=f'{a}-{b}',seed_differences=dict(zip(map(str,SEEDS),values)),positive_seeds=sum(v is not None and v>0 for v in values)))
    for item in summary['primary']:
        t=item['inference']
        if t:
            low,high=t['Bonferroni_family95_approximate_CI']
            label='达到预定Holm2阈值的正向差异' if t['p_Holm2']<.05 and t['mean_difference']>0 else ('达到预定Holm2阈值的负向差异' if t['p_Holm2']<.05 else '未达到预定Holm2阈值，不能据此认定等效')
            conclusions.append(f"{item['comparison']}：Δr={t['mean_difference']:.8f}，family95近似CI[{low:.8f}, {high:.8f}]，exact sign-flip p={t['p']:.8g}，Holm2 p={t['p_Holm2']:.8g}；{label}。")
        else:conclusions.append(f"{item['comparison']}：固定支持上的预测未全部可评估，结果undefined，没有删块后重算。")
    direction_text='\n'.join(f"- {r['comparison']}，seed1024/2026/31415差值依次为"+', '.join(number(r['seed_differences'][str(s)]) for s in SEEDS)+f"；正向{r['positive_seeds']}/3。" for r in directions)
    secondary=[]
    for a,b in [('F','Ccap'),('W','Ccap')]:
        secondary.append(dict(comparison=f'{a}-{b}',descriptive_only=True,
            delta_r=None if model[a]['r'] is None or model[b]['r'] is None else model[a]['r']-model[b]['r'],
            delta_equal_clip_r=None if model[a]['equal_clip_r'] is None or model[b]['equal_clip_r'] is None else model[a]['equal_clip_r']-model[b]['equal_clip_r']))
    write_csv(output/'SECONDARY_DESCRIPTIVE.csv',secondary)
    secondary_text='；'.join(f"{r['comparison']} Δr={number(r['delta_r'])}、equal-clip Δr={number(r['delta_equal_clip_r'])}" for r in secondary)
    precision_failed=sum(len(j.get('precision_failures',[])) for j in metric_files.values())
    costs=cost_ledger()
    with (output/'COST_LEDGER.json').open('x') as f:json.dump(costs,f,indent=2,allow_nan=False)
    text=f'''# ds004408外部迁移结果：无标签全record整体增益校准

状态：实际三seed四角色结果已汇总，独立最终数值审阅仍待；不是最终验收或goal完成。固定19人/380条/988完整60秒块，三seed只在人内平均，不能算57个独立人。所有推断条件于同一叙述/说话人及固定offline/transductive输入适配。

{chr(10).join(conclusions)}

## 图1：逐人及逐seed绝对表现

![逐人和逐seed表现](figures/person_and_seed_performance.png)

每个半透明点为一个人在三seed内平均后的块r；黑三角是每seed的被试宏平均，菱形是最终平均。点数19代表推断单位，seed三角不是独立样本。原TRAIN mean/std未重拟合；每record整体gain是新增无标签输入统计，共380次，使用未来EEG，不能称纯零校准确认。

|角色|seed|60s/person r|物理MSE|3s r|窗口MSE|
|---|---:|---:|---:|---:|---:|
{table}

{direction_text}

预定secondary仅描述：{secondary_text}。不据这些结果改变两个primary或模型。

完整228个逐seed逐人行见PERSON_BY_SEED.csv；每seed/角色表见SEED_SUMMARY.csv，全部固定块见ROOT的BLOCK_RESULTS.csv。不能只凭某个seed或最高均值下结论。

## 图2：两个预定主比较

![两主比较及逐人差异](figures/two_primary_differences.png)

区间是共同person paired bootstrap的97.5% percentile区间（两比较Bonferroni family95近似覆盖），不是精确区间；p来自全部2^19符号的abs(mean差值)枚举，仍假定人独立和零假设下符号对称/可交换。Holm2另行调整；不能把20段共享故事当20个故事重复验证，也不能把相关改善解释成历史机制或因果机制。

## 图3：窗口误差与融合误差

![不同坐标下的误差](figures/window_and_fused_error.png)

窗口MSE、Hann加权single MSE与最终融合MSE在不同坐标下度量，不能互换。F已知可能靠低Hann权重处的大窗口值与重叠相消形成较小融合误差；单窗病态不自动证明融合输出无效，较好融合分数也不证明窗口表示稳健。四角色均保留窗口r/MSE、overlap、weightedSingle/fused/disagreement恒等式与边界/tail记录。包络来自当前固定gammatone处理，旧发布revision未知，MSE的幅度域限制保留。

## 图4：全FP32敏感性

![FP32诊断](figures/full_FP32_sensitivity.png)

W/F诊断使用同一份FP32 Z/P，但读出、原TRAIN统计转换、Hann及累积均用FP32；主结果始终FP64读出/Hann。左图每点是一个record/seed的波形最大差，右图是完整块/人/seed层级的r32−r64，不依据这些差选择precision或删模型。本轮成功诊断{len(precision)}/2280个record-seed-role，记录到的诊断异常{precision_failed}；异常单列，不能使有效FP64主结果被静默排除。逐record与逐人诊断见PRECISION_RECORDS/PRECISION_PERSON。

## 图5：事先选定的实际示例

![预定记录前60秒](figures/preselected_example.png)

固定首条canonical record和seed2026，显示前60秒；四面板共用由目标与全部预测实际最小/最大值确定的同一纵轴，没有看成绩选例子、重新缩放预测、裁掉负值或目标驱动平移。示例只解释输出形态，不代替全部19人的比较，也不证明物理同步。完整目标/预测数组只保留2203。

## 来源、限制与执行范围

物理增益UNVERIFIED；唯一标量适配来自输入EEG，不是SI修复。BioSemi128→64用标准模板，缺个体实测位置；README零偏移是来源假设，marker为空。Sparr源训练为另一个collection，当前英语叙述与原荷兰语/DTU丹麦语来源不同，但缺完整跨库原音频指纹，不能排除XLSR大规模预训练音频接触。source/model/mask/statistics、实际payload SHA和跨两盘总量均有记录。

本轮没有新学习模型拟合、新权重、患者资料、旧test/DTU新推理或成绩驱动适配。所有训练模型/读出均在新预测前冻结；3seed主encoder/head各45429窗一次，工程原Decoder桥另计，实际wall/CPU/RSS/GPU及传输接续总耗时见COST_LEDGER.json，继承训练成本不能当零。此前D11及更早结果不改。本外部结果只检验固定读出在此校准条件下的迁移，不建立连续历史机制、方法新颖性或ICML/NeurIPS充分性；SS07科学goal不由本报告自动关闭。
'''
    with (output/'REPORT_DRAFT.md').open('x') as f:f.write(text)
    with (output/'DIRECTIONS.json').open('x') as f:json.dump(directions,f,indent=2,allow_nan=False)
    files=[p for p in output.rglob('*') if p.is_file()]
    with (output/'PRESENTATION_COMPLETED.json').open('x') as f:json.dump(dict(artifacts={str(p.relative_to(output)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files},
        figures=5,new_NN=0,actual_results_only=True,review_pending=True),f,indent=2)
    print('ACTUAL_FIVE_FIGURES_AND_FULL_PERSON_SEED_TABLES_READY_FOR_REVIEW')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','run']);args=p.parse_args()
    freeze() if args.mode=='freeze' else run()
