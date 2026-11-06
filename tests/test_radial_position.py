"""Coordinate supervision, spatial order, extrapolation and artifact boundaries."""
import importlib
import importlib.util

import numpy as np
import pytest
import torch

from test_radial_ranking import prepared_radial


def module():
    name='src.mylonas_ijepa_wm.radial_position'
    assert importlib.util.find_spec(name), 'position model missing'
    return importlib.import_module(name)


def test_readout_preserves_spatial_order_and_has_unbounded_output():
    model=module().PositionReadout(patches=2,dim=1,projection=1,hidden=1)
    with torch.no_grad():
        model.project.weight.fill_(1);model.project.bias.zero_()
        model.hidden.weight.copy_(torch.tensor([[1.,0.]]));model.hidden.bias.zero_()
        model.output.weight.copy_(torch.tensor([[1.],[2.]]));model.output.bias.zero_()
    tokens=torch.tensor([[[2.],[0.]],[[0.],[2.]]])
    y=model(tokens)
    assert y.shape==(2,2) and y[0,0]>1
    assert y[1,0]==0 and y[0,1]==2*y[0,0]


def test_coordinate_fit_uses_fixed_xy_targets_and_warm_starts_without_mutation():
    tokens=torch.tensor([[[1.,0],[0,1]],[[0.,1],[1,0]],[[1.,1],[-1,0]],[[0.,-1],[1,1]]])
    xy=torch.tensor([[52.,42.],[50.,48.],[46.,40.],[54.,36.]])
    model,t=module().fit(tokens,xy,seed=7,updates=250,batch_size=8,projection=3,hidden=8,lr=.01)
    # Zero final layer initially predicts (50,42); squared normalized error
    # is (4+36+16+4+16+36)/(8*34^2)=112/9248.
    assert t['full_position_loss_before']==pytest.approx(112/9248)
    assert t['position_error_after']<.1
    assert sum(t['sample_counts'])==2000 and t['position_scale']==34
    assert t['initialization']=='fresh' and t['optimizer_reset']
    before={k:v.clone() for k,v in model.state_dict().items()}
    updated,more=module().fit(tokens,xy,seed=8,updates=2,batch_size=2,
        projection=3,hidden=8,initial_model=model)
    assert updated is not model and more['initialization']=='previous_checkpoint'
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())


def test_position_metrics_exclude_exposure_deduplicate_and_measure_true_xy_error():
    design=dict(goal=[50,42],points=[[50,42],[52,42],[56,42],[54,42],[58,42],[50,46]],curves=[
        dict(kind='trained_direction',point_ids=[4,2,3,1]),
        dict(kind='withheld_direction',point_ids=[4,3,5]),
        dict(kind='withheld_direction',point_ids=[3,4])])
    truth=np.asarray(design['points'],dtype=float);prediction=truth.copy()
    prediction[4]=[50,50]  # Same distance8 to goal, but wrong ball position.
    hashes=['g','t1','t2','u1','u2','t1']
    report=module().summarize_positions(design,prediction,hashes,[1,2],[1],6)
    assert report['exposure']['heldout_from_offered_training'][0]  # Goal is not a fitting input.
    assert not report['exposure']['heldout_from_offered_training'][5]
    assert report['training']['count']==2
    outside=report['withheld_direction']['outside']
    assert outside['count']==1
    assert outside['model']['mean_position_error']==pytest.approx(128**.5)
    assert outside['model']['distance_mae']==pytest.approx(0)
    assert outside['model']['xy_mae']==pytest.approx([8,8])
    assert report['fixed_near_withheld_direction']['count']==1
    assert report['outside_shells'][0]['metrics']['count']==1
    final=module().summarize_positions(design,prediction,hashes,[1,2],[1],34)
    assert final['directional']['outside']['count']==0
    assert final['directional']['outside']['model']['mean_position_error'] is None
