"""Heldout references and matched single/multi-goal distance supervision."""
import importlib
import importlib.util

import numpy as np
import pytest
import torch


def module(name='radial_multigoal'):
    full='src.mylonas_ijepa_wm.'+name
    assert importlib.util.find_spec(full), f'{name} missing'
    return importlib.import_module(full)


def test_split_excludes_reserved_coordinates_images_and_identical_training_images():
    # Node4 aliases withheld goal3; nodes1/5 share pixels; node6 exceeds pair range.
    points=[[50,42],[50,40],[48,38],[50,18],[50,18.1],[50,39.9],[50,8],[51,20]]
    hashes=['a','b','c','goal','goal','b','f','query']
    split=module().make_split(points,[0,1,2,3,4,5,6],[0,3],hashes)
    assert split['train_ids']==[0,1,2,6]
    assert split['removed_candidate_ids']==[3,4,5]
    assert split['heldout_mask']==[False]*7+[True]
    single=split['single_pairs'];multi=split['multi_pairs']
    assert {tuple(p) for p in single}=={(1,0),(2,0),(6,0)}
    assert len({tuple(p) for p in multi})==len(multi)
    for a,b in multi:
        assert a<b and a in split['train_ids'] and b in split['train_ids']
        assert 0<np.linalg.norm(np.subtract(points[a],points[b]))<=34+1e-5
    assert (0,6) in map(tuple,multi)


def test_pair_costs_use_both_endpoints_and_target_linear_distance():
    mod=module()
    tokens=torch.tensor([[[0.,0.]],[[1.,0.]],[[2.,1.]]])
    pairs=[[1,0],[2,1]]
    adapter=torch.nn.Linear(2,2,bias=False)
    with torch.no_grad():adapter.weight.copy_(torch.eye(2))
    values=mod.pair_costs(adapter,tokens,pairs)
    assert torch.allclose(values,torch.tensor([.5,1.]))
    values.sum().backward()
    assert torch.isfinite(adapter.weight.grad).all() and adapter.weight.grad.abs().sum()>0
    assert torch.equal(mod.pair_costs(None,tokens,pairs),values.detach())
    positions=torch.tensor([[0.,0.],[17.,0.]])
    _,t=mod.fit(tokens[:2],positions,[[0,1]],updates=1,batch_size=4,bottleneck=2,lr=1e-30)
    assert t['probe_before']['regression_loss']<1e-12
    assert sum(t['endpoint_counts'])==8


def test_pair_fit_improves_fixed_training_probe_and_has_matched_budget():
    mod=module()
    tokens=torch.tensor([[[0.,0,1,0]],[[2.,1,0,0]],[[1.,0,0,1]]])
    positions=[[0,0],[2,0],[0,6]]
    _,t=mod.fit(tokens,positions,[[0,1],[0,2],[1,2]],seed=7,
        updates=200,batch_size=8,bottleneck=3,lr=.01)
    assert t['probe_after']['regression_loss']<.1*t['probe_before']['regression_loss']
    assert sum(t['pair_sample_counts'])==1600
    assert sum(t['endpoint_counts'])==3200
    assert t['initialization']=='identity' and t['anchor_weight']==0
    assert t['history'][0]['update']==0 and t['history'][-1]['update']==200
    assert t['distance_scale']==34 and t['probe_pair_ids']==[0,1,2]


@pytest.mark.parametrize('pairs',[[],[[0,0]],[[0,8]],[[0,1],[0,1]],[[1,0],[0,1]]])
def test_invalid_pair_populations_reject(pairs):
    with pytest.raises(ValueError,match='pair'):
        module().fit(torch.zeros(2,1,3),[[0,0],[2,0]],pairs,updates=1)
