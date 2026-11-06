"""Manual ordinal supervision with unseen goals and no distance regression."""
import importlib
import importlib.util

import numpy as np
import pytest
import torch


def module(name='multigoal_ranking'):
    full='src.mylonas_ijepa_wm.'+name
    assert importlib.util.find_spec(full), f'{name} missing'
    return importlib.import_module(full)


def test_triplets_use_distinct_roles_range_and_half_unit_gap():
    positions=np.array([[50,42],[50,40],[48,38],[50,18],[50,8],[50,42.2]])
    population=module().triples(positions,[0,1,3])
    assert population.ndim==2 and population.shape[1]==3 and len(population)>0
    assert set(population[:,0].tolist())=={0,1,3}
    for g,f,n in population.tolist():
        assert len({g,f,n})==3
        df,dn=np.linalg.norm(positions[[f,n]]-positions[g],axis=1)
        assert df<=34+1e-5 and dn<=34+1e-5 and df-dn>=.5-1e-5
    assert (0,4,1) in map(tuple,population.tolist())
    assert (0,1,5) in map(tuple,population.tolist())
    assert len({tuple(t) for t in population.tolist()})==len(population)


def test_cost_gradients_include_reference_and_both_queries():
    tokens=torch.tensor([[[1.,1.]],[[3.,0.]],[[2.,1.]]])
    adapter=torch.nn.Linear(2,2,bias=False)
    with torch.no_grad():adapter.weight.copy_(torch.eye(2))
    far,near=module().triplet_costs(adapter,tokens,[[0,1,2]])
    assert far.item()==2.5 and near.item()==.5
    (far-near).sum().backward()
    assert torch.allclose(adapter.weight.grad,torch.tensor([[3.,-2.],[-2.,1.]]))


def test_fit_improves_ranking_balances_references_and_audits_exposure():
    tokens=torch.tensor([[[0.,0,1,0]],[[2.,1,0,0]],[[1.,0,0,1]],[[3.,1,1,0]],[[2.,3,0,1]]])
    population=module().triples([[0,0],[1,0],[2,0],[3,0],[4,0]],[0]).tolist()+[[1,4,0]]
    _,t=module().fit(tokens,population,.5,seed=7,updates=200,batch_size=8,bottleneck=4,lr=.01)
    assert t['probe_after']['rank_loss']<.5*t['probe_before']['rank_loss']
    assert sum(t['endpoint_counts'])==4800 and len(t['sampled_triplet_ids'])==1600
    assert sum(t['reference_counts'])==1600 and abs(t['reference_counts'][0]-800)<100
    assert t['reference_counts'][2:]==[0,0,0]
    assert t['initialization']=='identity' and t['anchor_weight']==0 and t['scale']==.5
    assert t['probe_after']['rank_accuracy']>=t['probe_before']['rank_accuracy']
    assert 'distance_scale' not in t and 'distance_mae' not in t['probe_after']


def test_ordinal_metrics_accept_any_monotonic_cost_and_keep_ties_empty_groups():
    mod=module();points=np.array([[50,42],[50,18],[50,20],[50,24],[50,30]],float)
    goals=np.array([[50,42],[50,18]],float)
    truth=np.linalg.norm(points[None]-goals[:,None],axis=2)
    heldout=[False,False,True,True,True];edges=[(2,3),(3,4)]
    rows=mod.ordinal_rows(points,goals,truth,truth**2,heldout,edges)
    assert all(g['heldout_count']==3 for g in rows)
    assert all(g['adapted_rank']['correct_fraction']==1 for g in rows)
    assert all(g['heldout_directional']['adapted_up_fraction']==0 for g in rows)
    assert all('heldout_common' not in g for g in rows)
    for values in (np.zeros_like(truth),-truth):
        r=mod.ordinal_rows(points,goals,truth,values,heldout,edges)
        key='tie_fraction' if (values==0).all() else 'reversed_fraction'
        assert all(g['adapted_rank'][key]==1 for g in r)
    empty=mod.ordinal_rows(points,goals,truth,truth,[False]*5,edges)
    assert all(g['adapted_rank']['comparisons']==0 and g['adapted_rank']['correct_fraction'] is None for g in empty)


@pytest.mark.parametrize('population,scale',[([],1),([[0,0,1]],1),([[0,1,8]],1),([[0,1,2]],0)])
def test_invalid_triplets_or_scale_reject(population,scale):
    with pytest.raises(ValueError,match='triple|scale'):
        module().fit(torch.zeros(3,1,2),population,scale,updates=1)
