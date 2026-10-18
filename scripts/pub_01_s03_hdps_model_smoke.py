"""Original100Hz/M real-input forward/backward/resource smoke on2203."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from pub_01_s03_hdps_paper_model import PaperHDPS, normalize, similarity


def main(config):
    cfg = json.loads(config.read_text())
    base = Path(cfg['work'])
    cfg['model']['target_dim'] = 80
    torch.set_num_threads(4)
    torch.manual_seed(cfg['training']['seed'])
    eeg_path = next((base / 'preprocessing100_smoke06/eeg').rglob('*.npy'))
    eeg = np.load(eeg_path).T.astype(np.float32)
    target = np.load(base / 'mel_smoke01/audiobook_5_1.npy')
    ids = json.loads((base / 'valid_segments_metadata.json').read_text())['audiobook_5_1.wav']
    valid = [i for i in ids if (i + 1) * 300 <= min(len(eeg), len(target))]
    x = normalize(np.stack([eeg[i*300:(i+1)*300] for i in valid]))
    y = np.stack([target[i*300:(i+1)*300] for i in valid])
    model = PaperHDPS(cfg, 81).to('cuda:0')
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['lr'])
    records = []
    for batch in (16, 256):
        pick = np.arange(batch) % len(x)
        real_x = torch.from_numpy(x[pick]).to('cuda:0')
        real_y = torch.from_numpy(y[pick]).to('cuda:0')
        subject = torch.zeros(batch, dtype=torch.long, device='cuda:0')
        for gpu in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(gpu)
        optimizer.zero_grad(set_to_none=True)
        started = time.time()
        predicted, fused = model(real_x, real_y, subject)
        score = similarity(predicted, fused)
        loss = torch.nn.functional.cross_entropy(score / .015, torch.arange(batch, device='cuda:0'))
        loss.backward()
        finite_gradient = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        if not torch.isfinite(loss) or not finite_gradient:
            raise RuntimeError('nonfinite loss or gradient')
        optimizer.step()
        torch.cuda.synchronize()
        records.append({'batch': batch, 'seconds': time.time()-started, 'loss': loss.item(),
            'finite_gradient': finite_gradient, 'max_memory_GiB': [torch.cuda.max_memory_allocated(i)/2**30 for i in range(torch.cuda.device_count())],
            'note': 'resource smoke; repeat actual windows if fewer than batch; not a performance result'})
        print(json.dumps(records[-1]), flush=True)
    (base / 'model_smoke01.json').write_text(json.dumps({'records': records, 'parameter_count': sum(p.numel() for p in model.parameters()),
        'source_eeg': str(eeg_path), 'real_valid_segments': len(valid), 'sampling_rate': 100}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    main(parser.parse_args().config)
