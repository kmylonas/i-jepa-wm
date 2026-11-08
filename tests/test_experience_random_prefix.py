from types import SimpleNamespace

import numpy as np
import torch

from src.mylonas_ijepa_wm import experience_curriculum as exp
from src.mylonas_ijepa_wm import plan_proprio
from src.mylonas_ijepa_wm.curriculum_protocol import load_stage, run_stage
from src.mylonas_ijepa_wm.environment_adapters import StepResult


def test_temporal_replay_skips_the_random_transition():
    frames = torch.arange(3)[:, None]
    raw = dict(states=torch.tensor([[50, 51], [50, 52], [50, 49]]),
               frames=frames, primitive_steps=torch.arange(3), success=True, prefix_steps=1)
    design = dict(points=[[50, 45]], heldout_mask=[False], goal_ids=[0], frame_sha256=['reference'])
    starts, _ = exp._temporal_filter(raw, dict(goal=[50, 45], goal_id=0), design, 8)
    assert starts == [1]


def test_stage_resumes_and_stops_each_goal_at_quota(tmp_path):
    cases = [dict(case_id=str(i), goal_index=i % 2) for i in range(8)]
    seen = []

    def execute(case):
        seen.append(case['case_id'])
        return dict(success=case['case_id'] != '0', micro_steps=1)

    records = run_stage(tmp_path, 'band', cases, 'runtime', execute, success_quota=2)
    assert [r['case_id'] for r in records] == ['0', '1', '2', '3', '4']
    assert run_stage(tmp_path, 'band', cases, 'runtime', execute, success_quota=2) == records
    assert load_stage(tmp_path, 'band', cases, 'runtime') == records
    assert seen == ['0', '1', '2', '3', '4']


def test_mpc_starts_from_post_prefix_frame_and_shares_budget(monkeypatch):
    class Env:
        action_low = np.array([-1., -1.])
        action_high = np.array([1., 1.])
        goal = np.array([50., 50.])
        def step(self, action):
            self.position += action
            return StepResult(self.position.copy(), self.position.copy(), False)

    env = Env()
    env.position = np.array([0., 0.])
    initial = StepResult(env.position.copy(), env.position.copy(), False)
    monkeypatch.setattr(plan_proprio.visual, 'encode_frame',
                        lambda encoder, frame, device, precision: torch.tensor(frame)[None])
    calls = []
    def plan(**kwargs):
        calls.append(kwargs['latent_history'].clone())
        return SimpleNamespace(action_sequence=torch.ones(1, 10), cost=0.)
    monkeypatch.setattr(plan_proprio.planner, 'plan_action', plan)
    args = SimpleNamespace(cost='position', max_macro_steps=1, planning_horizon=1, actions_per_plan=1,
                           population=2, num_elites=1, cem_iterations=1, candidate_batch_size=2, min_std=.1)
    config = SimpleNamespace(env='pointmaze', actions_per_model_step=5, macro_action_dim=10, micro_action_dim=2)
    metadata = dict(state_mean=torch.zeros(2), state_std=torch.ones(2))
    observed = []
    result = plan_proprio.run_episode(env, config, None, SimpleNamespace(num_hist=1), metadata, args,
                                      torch.device('cpu'), 'fp32', 0, None, initial=initial,
                                      initial_action=[1., 0.], observation_observer=lambda i, *_: observed.append(i))
    assert calls[0].flatten().tolist() == [1., 0.]
    assert observed == list(range(6))
    assert result['micro_steps'] == 5 and result['prefix_steps'] == 1
    assert result['macro_durations'] == [4]



def test_boundary_entry_is_counted_on_realized_right_room_positions():
    from src.mylonas_ijepa_wm import experience_exploration as explore

    case = dict(goal=[44, 30], random_first_action=[-1, 0])
    xy = [[36, 41], [34, 40], [36, 39], [37, 39]]
    row = explore.measure(dict(case=case, primitive_positions=xy, prefix_steps=1), 10, 5)
    assert row['arm'] == 'random first'
    assert row['first_entry_step'] == 2
    assert not row['entered_by_first_action']
    assert row['entered_by_five_actions']
    assert row['first_distance_change'] > 0 and row['final_distance_change'] < 0
