"""Metadata/schema audit only; no models or performance. Server2203."""
import json
from pathlib import Path
import time
import numpy as np
from scipy.io import whosmat,loadmat

ROOT=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_incremental_continuity_20261101/external_dtu')


def structure(value,path='',depth=0):
    if depth>7:return {'type':type(value).__name__,'truncated_depth':True}
    if isinstance(value,dict):
        return {str(k):structure(v,path+'/'+str(k),depth+1) for k,v in value.items() if not str(k).startswith('__')}
    if isinstance(value,(list,tuple)):
        return {'type':type(value).__name__,'length':len(value),
            'first_two_schemas':[structure(v,path+f'/{i}',depth+1) for i,v in enumerate(value[:2])]}
    if isinstance(value,np.ndarray):
        result={'shape':list(value.shape),'dtype':str(value.dtype)}
        if value.dtype.kind in 'US' and value.size<=256:result['metadata_labels']=value.tolist()
        if value.dtype.kind=='O':
            result['first_two_schemas']=[structure(v,path+'/object',depth+1) for v in value.flat[:2]]
        return result
    if isinstance(value,(str,np.str_)):return {'type':'string','metadata':str(value)[:500]}
    if isinstance(value,(int,float,np.number)):
        result={'type':type(value).__name__}
        if any(k in path.lower() for k in ['fs','samplerate','sampling_rate','nchan']):result['metadata_value']=float(value)
        return result
    return {'type':type(value).__name__}


def main():
    deadline=time.time()+4*3600
    while not (ROOT/'DOWNLOAD_EXTRACT_COMPLETED.json').exists():
        if (ROOT/'download_status.json').exists():
            status=json.loads((ROOT/'download_status.json').read_text())
            if status['state']=='FAILED_PRESERVED':raise RuntimeError('download failed; no automatic retry')
        if time.time()>deadline:raise RuntimeError('download wait expired')
        time.sleep(30)
    files=sorted((ROOT/'extracted').glob('S*_data_preproc.mat'));assert len(files)==18
    headers=[]
    for path in files:
        headers.append({'file':path.name,'bytes':path.stat().st_size,
            'MATLAB_variables':[{'name':name,'shape':list(shape),'MATLAB_class':kind} for name,shape,kind in whosmat(path)]})
    first=ROOT/'extracted/S1_data_preproc.mat';data=loadmat(first,simplify_cells=True)
    result={'metadata_schema_only':True,'model_performance_computed':False,'all18_MAT_headers_readable':True,
        'headers':headers,'first_subject':first.name,'first_subject_structure':structure(data),
        'scalp_channel_mapping_or_trial_eligibility_not_yet_claimed':True,'time':time.time()}
    with (ROOT/'SCHEMA_HEADER_AUDIT.json').open('x') as output:json.dump(result,output,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
