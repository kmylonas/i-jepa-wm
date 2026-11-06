"""Tests catch false claims of sampled-pair exposure and spatial monotonicity."""
import importlib
import importlib.util
from collections import Counter

import numpy as np
import pytest
import torch


def geometry():
    name = 'src.mylonas_ijepa_wm.curriculum_local_geometry'
    assert importlib.util.find_spec(name), 'local diagnostic geometry module missing'
    return importlib.import_module(name)


def test_reconstructed_pairs_and_inputs_match_actual_fitter(monkeypatch):
    from src.mylonas_ijepa_wm import curriculum_replay as replay
    episodes = []
    for k in range(2):
        episodes.append(dict(case_id=f'train_{k}', split='train', success=True,
            case=dict(goal=[50,42],band='near'),
            states=torch.tensor([[44.,30.],[45.,32.],[46.,35.],[48.,39.],[50.,42.]]),
            primitive_steps=torch.arange(5)*2,
            latents=(torch.arange(5)+k*10+1).float()[:,None,None].expand(-1,1,4)))
    observed=[]
    original=replay.token_goal_cost
    def watch(adapter, tokens, goal):
        observed.append(tokens[:,0,0].tolist())
        return original(adapter,tokens,goal)
    monkeypatch.setattr(replay,'token_goal_cost',watch)
    _, training=replay.fit_adapter(episodes,torch.zeros(1,4),seed=13,updates=7,batch_size=3,
        bottleneck=2,anchor_weight=0,min_goal_progress=.5)
    result=geometry().sampled_pairs(episodes,training)
    actual=Counter((a,b) for row in observed for a,b in zip(row[:3],row[3:6]))
    predicted=Counter({(float(episodes[p['episode']]['latents'][p['i'],0,0]),
                        float(episodes[p['episode']]['latents'][p['j'],0,0])):p['draws']
                       for p in result['pairs']})
    assert actual==predicted and sum(actual.values())==21
    expected=set(float(v) for row in observed for v in row)
    assert {float(episodes[e]['latents'][i,0,0]) for e,i in result['inputs']}==expected


def test_profiles_exclude_overshoot_and_out_of_domain_offsets():
    mod=geometry()
    result=mod.profiles([50,30],[50,40],[50,42],(4.001,59.999))
    assert len(result['valid'])==7 and result['rejected']==0
    center=next(p for p in result['valid'] if p['offset']==0)
    assert np.allclose(np.asarray(center['points'])[:,1],np.arange(30,41))
    assert not mod.profiles([50,40],[50,43],[50,42],(4.001,59.999))['valid']
    near_edge=mod.profiles([59,30],[59,40],[50,42],(4.001,59.999))
    assert near_edge['rejected']>0
    assert all(np.asarray(p['points']).max()<=59.999 for p in near_edge['valid'])


@pytest.mark.parametrize('costs,up,down,ties,informative',[
    ([3,2,1],0,2,0,True),([1,2,3],2,0,0,False),
    ([3,4,1],1,1,0,False),([2,2,2],0,0,2,False)])
def test_curve_metrics_distinguish_ridges_and_flatness(costs,up,down,ties,informative):
    result=geometry().curve_metrics(costs)
    assert (result['up_steps'],result['down_steps'],result['tie_steps'])==(up,down,ties)
    assert result['informative_monotone']==informative
    assert result['flat']==(ties==2)
    if costs==[3,4,1]: assert result['uphill_normalized']==.5


def test_coverage_checks_other_fitted_positions_not_only_selected_endpoints():
    result=geometry().coverage([[45,35],[45,35.2],[46,35]],[[40,30],[45,35]],[[46,35]])
    assert np.allclose(result['nearest_fitting_distance'],[0,.2,1])
    assert result['fitted_position']==[True,False,False]
    assert result['replay_position']==[False,False,True]


def test_pair_selection_spreads_across_episodes_without_discarding_rank_failures():
    mod=geometry()
    episodes=[dict(states=torch.tensor([[50.,30.],[50.,34.],[50.,38.]])) for _ in range(2)]
    pairs=[dict(episode=e,i=i,j=j,draws=1) for e in range(2) for i,j in ((0,1),(0,2),(1,2))]
    result=mod.select_pairs(pairs,episodes,[50,42],(4.001,59.999),count=2)
    assert len(result['selected'])==2
    assert {p['episode'] for p in result['selected']}=={0,1}
    assert result['geometrically_eligible']==6
