"""Physical distance targets, cumulative fitting and genuinely heldout errors."""
import importlib
import importlib.util

import numpy as np
import pytest
import torch


def module():
    name='src.mylonas_ijepa_wm.radial_distance'
    assert importlib.util.find_spec(name), 'distance fitter missing'
    return importlib.import_module(name)


def test_target_is_linear_distance_with_one_fixed_scale():
    # Identity MSE is exactly r/34 at all three radii, not (r/34)^2.
    radii=torch.tensor([2.,6.,10.])
    tokens=torch.zeros(3,1,2)
    tokens[:,0,0]=(2*radii/34).sqrt()
    _,training=module().fit(tokens,radii,torch.zeros(1,2),seed=0,
        updates=1,batch_size=4,bottleneck=2,lr=1e-30)
    assert training['distance_scale']==34
    assert training['full_regression_loss_before']<1e-12
    assert training['history'][-1]['full_regression_loss']<1e-12
    assert training['distance_rmse_before']<1e-5
    assert sum(training['sample_counts'])==4


def test_distance_fit_reduces_error_and_warm_start_preserves_incoming():
    tokens=torch.tensor([[[3.,0,1,0]],[[2.,1,0,0]],[[1.,0,0,1]]])
    radii=torch.tensor([2.,6.,10.]);goal=torch.zeros(1,4)
    adapter,t=module().fit(tokens,radii,goal,seed=7,updates=200,
        batch_size=8,bottleneck=3,lr=.01)
    assert t['history'][-1]['full_regression_loss']<.05*t['full_regression_loss_before']
    assert t['distance_rmse_after']<t['distance_rmse_before']
    assert sum(t['sample_counts'])==1600
    assert t['initialization']=='identity' and t['anchor_weight']==0
    before={k:v.clone() for k,v in adapter.state_dict().items()}
    updated,more=module().fit(tokens,radii,goal,seed=8,updates=2,batch_size=2,
        bottleneck=3,initial_adapter=adapter)
    assert more['initialization']=='previous_checkpoint' and more['optimizer_reset']
    assert updated is not adapter
    assert all(torch.equal(v,adapter.state_dict()[k]) for k,v in before.items())
    for invalid in (torch.tensor([-1.,6.,10.]),torch.tensor([2.,float('nan'),10.])):
        with pytest.raises(ValueError):module().fit(tokens,invalid,goal,seed=0)


def test_distance_diagnostics_calibrate_on_training_only_and_preserve_empty_regions():
    design=dict(goal=[0,0],points=[[0,0],[2,0],[6,0],[4,0],[8,0],[0,4]],curves=[
        dict(kind='trained_direction',point_ids=[4,2,3,1]),
        dict(kind='withheld_direction',point_ids=[4,3,5]),
        dict(kind='withheld_direction',point_ids=[3,4])])
    report=dict(offered_training_ids=[1,2],outer_radius=6,
        exposure=dict(heldout_from_offered_training=[False,False,False,True,True,False]))
    raw=[0,.2,.6,.4,.8,.4]
    adapted=np.array([0,2,6,4,6,4])/34
    summary=module().distance_summary(design,raw,adapted,report)
    assert summary['raw_calibration']==pytest.approx(10)
    assert summary['training']['count']==2
    assert summary['training']['adapted']['rmse']==pytest.approx(0)
    outside=summary['withheld_direction']['outside']
    assert outside['count']==1 and outside['adapted']['mae']==pytest.approx(2)
    assert outside['raw_calibrated']['rmse']==pytest.approx(0)
    assert summary['fixed_near_withheld_direction']['count']==1
    assert summary['fixed_near_withheld_direction']['adapted']['mae']==pytest.approx(0)
    raw[4]=1000
    assert module().distance_summary(design,raw,adapted,report)['raw_calibration']==pytest.approx(10)
    report['outer_radius']=34
    empty=module().distance_summary(design,raw,adapted,report)['directional']['outside']
    assert empty['count']==0 and empty['adapted']['mae'] is None
