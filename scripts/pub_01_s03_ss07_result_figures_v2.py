"""V2 reviewed-result entry; pure rendering implementation stays unchanged."""
import json
import time
from pub_01_s03_ss07_baseline import ROOT, read, dump, sha
from pub_01_s03_ss07_review_metrics_v2 import gated
from pub_01_s03_ss07_result_figures import render, FILES, FIGURES


def reviewed_gate():
    done = read(ROOT/'INDEPENDENT_REVIEW_COMPLETED_V2.json')
    assert done['technical_review_passed'] and done['review_version'] == 2
    assert not done['scientific_goal_complete']
    manifest, ready = gated()
    assert done['protocol_sha256'] == sha(ROOT/'INDEPENDENT_REVIEW_PROTOCOL_V2.json')
    assert done['old_raw_protocol_sha256'] == sha(ROOT/'INDEPENDENT_REVIEW_PROTOCOL.json')
    raw = read(ROOT/'independent_raw_review/RAW_INPUT_REVIEW.json')
    metrics = read(ROOT/'independent_review/METRICS_REVIEW.json')
    assert raw['passed'] and metrics['passed'] and metrics.get('review_version') == 2
    assert raw['review_protocol_sha256'] == sha(ROOT/'INDEPENDENT_REVIEW_PROTOCOL.json')
    assert metrics['review_protocol_sha256'] == sha(ROOT/'INDEPENDENT_REVIEW_PROTOCOL_V2.json')
    for proof in [raw, metrics]:
        assert proof['checkpoint_manifest_sha256'] == sha(ROOT/'FINAL_CHECKPOINT_MANIFEST.json')
    return manifest


def main():
    manifest = reviewed_gate()  # before reading any actual outcome summary
    documents = {name: read(ROOT/'final_summary'/name) for name in FILES}
    destination = ROOT/'final_presentation01'
    result = render(destination, manifest, documents)
    dump(destination/'PRESENTATION_MANIFEST.json', dict(result, time=time.time(),
        sources={name: sha(ROOT/'final_summary'/name) for name in FILES},
        source_sha256=sha(ROOT/'input/pub_01_s03_ss07_result_figures_v2.py'),
        reused_rendering_source_sha256=sha(ROOT/'input/pub_01_s03_ss07_result_figures.py'),
        independent_review_complete_sha256=sha(ROOT/'INDEPENDENT_REVIEW_COMPLETED_V2.json'),
        raw_protocol_version=1, metric_protocol_version=2,
        scientific_goal_complete=False, original_data_and_summaries_unchanged=True))
    print(json.dumps({'presentation_generated': True, 'review_version': 2,
        'scientific_goal_complete': False, 'figures': FIGURES}))


if __name__ == '__main__': main()
