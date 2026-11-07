"""Check new read-only model inventory against historical frozen artifact identities."""
import hashlib,json
from pathlib import Path
from pub_01_s03_ss07_openneuro_transport import ROOT


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    base=ROOT.parent;assets=json.loads((ROOT/'SOURCE_ASSETS.json').read_text())
    historical=[base/'s03_finetuned_internal_audit_20261101/INPUT_INVENTORY.json',
        base/'s03_finetuned_internal_audit_20261101/PROTOCOL.json',
        base/'s03_finetuned_readout_transfer_20261101/INPUT_INVENTORY.json',
        base/'s03_physical_objective_20261101/PROTOCOL.json']
    index={}
    for p in historical:
        value=json.loads(p.read_text())
        for key,digest in value['source_pins'].items():index.setdefault(key,[]).append(dict(manifest=str(p),sha256=digest))
    checked=[]
    for path,digest in assets['source_pins'].items():
        if not (path.endswith(('.npz','.pt')) or path.endswith('P_FT_BASELINE.json')):continue
        assert path in index and index[path]
        assert all(row['sha256']==digest for row in index[path]) and sha(path)==digest
        checked.append(dict(path=path,sha256=digest,historical_references=index[path]))
    assert len(checked)==19
    value=dict(passed=True,scope='Original TRAIN norm,3 checkpoints,3 TRAIN transforms,3 calibrations,9 frozen Ccap/W/F readouts',
        assets_checked=checked,historical_manifests={str(p):sha(p) for p in historical},
        SOURCE_ASSETS_sha256=sha(ROOT/'SOURCE_ASSETS.json'),source_sha256=sha(__file__),real_NN_forwards=0,new_model_fits=0)
    with (ROOT/'LEGACY_ASSET_IDENTITY_CHECK.json').open('x') as f:json.dump(value,f,indent=2)
    print('HISTORICAL_ARTIFACT_IDENTITIES_PASS',len(checked))


if __name__=='__main__':main()
