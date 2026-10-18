"""Verify actual train→validation→train LSTM/DataParallel execution; no test."""
import argparse
import json
from pathlib import Path
import torch
from pub_01_s03_transformer_train_v2 import TransformerCLIP,StoryBatches,contrastive


def main(config):
    cfg=json.loads(config.read_text());work=Path(cfg['work']);prepared=work/'prepared';torch.set_num_threads(4);torch.manual_seed(1024)
    features=json.loads((work/'features/manifest.json').read_text());model=torch.nn.DataParallel(TransformerCLIP(cfg).cuda(),device_ids=[0,1]);opt=torch.optim.AdamW(model.parameters(),lr=.001)
    train=StoryBatches(prepared,features,'train',128);val=StoryBatches(prepared,features,'val',128);outcomes=[]
    for phase in ('train','val','train'):
        model.train(phase=='train');batch=next((train if phase=='train' else val).batches(1024,False));eeg,audio=(x.cuda() for x in batch)
        if phase=='train':
            opt.zero_grad(set_to_none=True);loss=contrastive(model,eeg,audio);loss.backward();opt.step()
        else:
            with torch.no_grad():loss=contrastive(model,eeg,audio)
        if not torch.isfinite(loss):raise RuntimeError('cycle nonfinite')
        outcomes.append({'phase':phase,'loss':loss.item(),'finite':True})
    with (work/'train_val_cycle_smoke.json').open('x') as stream:json.dump({'phases':outcomes,'no_test_read':True,'production_weights_not_reused':True},stream,indent=2)
    print(json.dumps(outcomes),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);main(parser.parse_args().config)
