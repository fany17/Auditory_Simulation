"""Read-only D18 within-seed budget/control audit on 2203; no neural work."""
import argparse
import gzip
import hashlib
import json
import math
import time
from pathlib import Path

ROOT = Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_preprojection_memory_20261103')
PROTOCOL = '255ce2b95457d807e41775eaa208d40024dc5d1637c3a37b24edfec189c90632'
ARMS = ('H_history', 'C_current', 'T_continue', 'P_prediction_history')


def audit(seed):
    wall, cpu = time.time(), time.process_time()
    assert hashlib.sha256((ROOT/'PROTOCOL.json').read_bytes()).hexdigest() == PROTOCOL
    protocol = json.loads((ROOT/'PROTOCOL.json').read_text())
    rows, hashes, orders, anchors, expected_counts = {}, {}, {}, {}, {}
    for arm in ARMS:
        folder = ROOT/'fits'/f'{arm}_seed{seed}'
        assert not (folder/'FAILURE.json').exists()
        data = (folder/'COMPLETED.json').read_bytes()
        row = json.loads(data)
        assert row['seed'] == seed and row['arm'] == arm and row['protocol_sha256'] == PROTOCOL
        assert row['epochs'] == 20 and row['optimizer_updates'] == row['update_positions'] == 78900
        assert row['empty_positions'] == 0 and row['target_window_presentations'] == 5049080
        assert row['effective_TRAIN_presentations'] == 4972400
        assert row['registered_trainable_parameters'] == protocol['registered_parameters'][arm]
        assert math.isfinite(row['selected_validation_r'])
        orders[arm] = hashlib.sha256((folder/'PERMUTATIONS.jsonl').read_bytes()).hexdigest()
        anchors[arm] = hashlib.sha256((folder/'INCUMBENT.json.gz').read_bytes()).hexdigest()
        hashes[arm] = hashlib.sha256(data).hexdigest()
        active = int(row['selected_epoch'] >= 0)
        current = 24*252454 + (20+active)*30500
        attention = 0 if arm == 'T_continue' else 24*248620 + (20+active*(2 if arm == 'H_history' else 1))*30059
        extra_head = active*30500 if arm == 'H_history' else 0
        frozen = 2*attention if arm == 'H_history' else attention if arm == 'C_current' else 0
        expected = dict(current_encoder_prefix_windows=current,
                        current_projection_windows=current+extra_head,
                        current_head_windows=current+extra_head,
                        frozen_encoder_prefix_windows=frozen,
                        attention_query_windows=attention,
                        prior_projection_windows=2*attention if arm == 'P_prediction_history' else 0)
        assert row['component_forward_counts'] == expected
        expected_counts[arm] = expected
        rows[arm] = row
    assert len(set(orders.values())) == len(set(anchors.values())) == 1
    folder = ROOT/'fits'/f'H_history_seed{seed}'
    incumbent = json.loads(gzip.decompress((folder/'INCUMBENT.json.gz').read_bytes()))['phases']['val']['r']
    older_bytes = (folder/'OLDER_METRICS.json.gz').read_bytes()
    older = json.loads(gzip.decompress(older_bytes))['phases']['val']['r']
    value = rows['H_history']['selected_validation_r']
    differences = {arm: value-rows[arm]['selected_validation_r'] for arm in ARMS[1:]}
    directions = {arm: delta > 0 for arm, delta in differences.items()}
    directions['incumbent'] = value > incumbent
    directions['near_gt_older'] = value > older
    cost_fields = ('CPU_seconds', 'wall_seconds', 'peak_RSS_KiB', 'peak_GPU_allocated_bytes', 'registered_trainable_parameters')
    return dict(seed=seed, passed=True, protocol_sha256=PROTOCOL,
                same_protocol_supervision_order_and_incumbent_passed=True,
                all_four_component_formulas_passed=True, expected_counts=expected_counts,
                completed_sha256=hashes, permutation_sha256=orders, incumbent_sha256=anchors,
                H_older_metric_sha256=hashlib.sha256(older_bytes).hexdigest(),
                selected_validation_r={arm: row['selected_validation_r'] for arm, row in rows.items()},
                selected_epochs={arm: row['selected_epoch'] for arm, row in rows.items()},
                incumbent_r=incumbent, H_minus_controls_r=differences, H_near_minus_older_r=value-older,
                per_seed_gate_components=directions, per_seed_all_required_directions=all(directions.values()),
                criterion_source='COORDINATOR_DIRECTION.md:44 (commit63e77496cacad6174cbdf900efac56fb6211e3bf); PLAN.md:61',
                training_costs={arm: {key: row[key] for key in cost_fields} for arm, row in rows.items()},
                audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                audit_CPU_seconds=time.process_time()-cpu, audit_wall_seconds=time.time()-wall,
                audit_NN_forwards=0, audit_backwards=0, audit_optimizer_updates=0,
                scope='One completed seed; compare all three seeds under the unchanged original direction rule. No significance, independent-generalization, or global history-null claim.',
                cost_interpretation='Invoked counts are not unique data or FLOPs. P prior cache creation is previously incurred upstream work, not free deployment history.',
                goal_complete=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=(1024, 2026, 31415), required=True)
    print(json.dumps(audit(parser.parse_args().seed), allow_nan=False))
