from types import SimpleNamespace

import numpy as np
import torch

from src.mylonas_ijepa_wm import experience_curriculum as exp
from src.mylonas_ijepa_wm import experience_half_square as half


def test_repeated_rounds_keep_the_first_start_batch_and_sample_fresh_starts():
    points = [[50, 42], [44, 40], [53, 48], [44, 20], [50, 12], [55, 14]]
    design = dict(points=points, train_ids=list(range(len(points))))
    original = half.make_manifest(design, bands=half.WIDE_BANDS)
    repeat = half.make_repeated_manifest(design, 6, bands=half.WIDE_BANDS, rounds=3)
    assert repeat == half.make_repeated_manifest(design, 6, bands=half.WIDE_BANDS, rounds=3)
    assert repeat['bands'] == original['bands']
    assert len(repeat['stages']) == 3 * len(original['stages'])
    case_ids = [case['case_id'] for stage in repeat['stages'] for case in stage]
    assert len(case_ids) == len(set(case_ids))
    for i, original_cases in enumerate(original['stages']):
        first, second, third = repeat['stages'][i*3:i*3+3]
        assert [(c['start'], c['seed']) for c in first] == [(c['start'], c['seed']) for c in original_cases]
        assert all(c['band'] == i+1 for c in first+second+third)
        starts = [tuple(c['start']) for c in first+second+third]
        assert len(starts) == len(set(starts))


def test_empty_rounds_carry_cost_and_one_pair_triggers_cumulative_fit(tmp_path):
    goal = [50, 42]
    design = dict(points=[goal], heldout_mask=[False], goal_ids=[0], frame_sha256=['reference'])
    cases = [dict(case_id=f'case_{i}', goal=goal, goal_index=0, band=1 if i < 2 else 2)
             for i in range(4)]
    manifest = dict(bands=[[0, 6], [6, 12]], rounds_per_band=2, goal_ids=[0],
                    stages=[[case] for case in cases], pair_cap=8)
    calls = []

    class Runtime:
        root = tmp_path
        runtime_id = 'fixture'
        device = 'cpu'
        config = SimpleNamespace(actions_per_model_step=5)

        def encode_frames(self, frames):
            return torch.as_tensor(frames, dtype=torch.float32)[:, None, :] / 10

        def run_case(self, case, *, capture, latent_cost_fn, journal_stage, capture_failures):
            assert capture and not capture_failures
            i = int(case['case_id'][-1])
            calls.append(i)
            success = i in (1, 3)
            xy = np.array([[50, 48], [50, 45]], dtype=np.float32)
            record = dict(success=success, raw_path=None, micro_steps=1, primitive_positions=xy.tolist())
            if success:
                path = f"raw/{case['case_id']}.pt"
                exp.save_tensors(tmp_path / path, dict(case=case, runtime_id=self.runtime_id,
                    success=True, frames=torch.as_tensor(xy), states=torch.as_tensor(xy),
                    primitive_steps=torch.arange(2)))
                record['raw_path'] = path
            return record

    runtime = Runtime()
    settings = dict(updates=1, batch_size=2, bottleneck=2)
    opts = dict(filter_steps=half.filter_steps,
                encode_positions=lambda rt, d, ids: rt.encode_frames([d['points'][i] for i in ids]))
    reports = [exp.collect_and_fit(runtime, manifest, design, settings, .1, i, **opts) for i in range(4)]
    assert [r['updates'] for r in reports] == [0, 1, 0, 1]
    assert [r['retained_pairs'] for r in reports] == [0, 1, 0, 1]
    assert [(r['band'], r['round']) for r in reports] == [(1, 1), (1, 2), (2, 1), (2, 2)]
    assert reports[0]['checkpoint_sha256'] is None
    assert reports[1]['collector'] == 'raw'
    assert reports[3]['collector'] == reports[2]['checkpoint_sha256']
    train = exp.read(tmp_path / 'band_2_round_2' / 'training.json')
    assert train['eligible_triples'] == 2 and train['initialization'] == 'previous_adapter'
    assert reports[2]['collection'][0]['boundary']['first_entry_step'] == 1
    assert [exp.collect_and_fit(runtime, manifest, design, settings, .1, i, **opts) for i in range(4)] == reports
    assert calls == [0, 1, 2, 3]
