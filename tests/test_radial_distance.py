"""Physical distance targets, cumulative fitting and genuinely heldout errors."""
import importlib
import importlib.util

import numpy as np
import pytest
import torch

from test_radial_ranking import prepared_radial


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


def distance_run(prepared_radial):
    runner,oldout,oldcache,_,env,encode,signature=prepared_radial
    out,cache=oldout.parent/'distance',oldcache.parent/'distance-cache'
    prepared=runner.prepare(out,cache,env,encode,signature,objective='distance',
        fitting=dict(updates=2,batch_size=4,bottleneck=3,lr=1e-3))
    return runner,out,cache,prepared,env,encode,signature


def test_distance_stages_only_fit_cumulative_training_and_reuse_completed_artifacts(prepared_radial,monkeypatch):
    runner,out,cache,prepared,*_=distance_run(prepared_radial)
    mod=module();original=mod.fit;seen=[]
    all_data=torch.load(out/'training_data.pt',weights_only=True)
    for stage_index,n in enumerate((48,90,105,115),1):
        ids=prepared['design']['stages'][stage_index-1]['training_ids']
        take=[all_data['point_ids'].index(i) for i in ids]
        def checked(tokens,radii,goal,**kwargs):
            assert torch.equal(tokens,all_data['tokens'][take])
            assert torch.equal(radii,all_data['radii'][take])
            seen.append(len(tokens));return original(tokens,radii,goal,**kwargs)
        monkeypatch.setattr(mod,'fit',checked)
        report=runner.stage(out,cache,stage_index)
        assert report['training_positions']==n and report['primitive_interactions']==0
        assert report['objective']=='distance' and report['training']['distance_scale']==34
        assert report['initialization']==('identity' if stage_index==1 else 'previous_checkpoint')
        assert sum(report['training']['sample_counts'])==8
        assert set(report['actual_fitting_ids'])<=set(ids)|{prepared['design']['goal_id']}
        assert report['distance']['training']['count']==n
        if stage_index==4:
            empty=report['distance']['withheld_direction']['outside']
            assert empty['count']==0 and empty['adapted']['rmse'] is None
    assert seen==[48,90,105,115]
    monkeypatch.setattr(mod,'fit',lambda *a,**k:(_ for _ in ()).throw(AssertionError('refit')))
    assert runner.stage(out,cache,4)['checkpoint_sha256']==report['checkpoint_sha256']


def test_objective_and_fixed_distance_scale_are_signed_and_mismatches_rejected(prepared_radial):
    import json
    from src.mylonas_ijepa_wm.curriculum_runtime import file_sha256
    runner,out,cache,prepared,env,encode,signature=distance_run(prepared_radial)
    assert prepared['config']['kind']=='radial_distance_v1'
    assert prepared['config']['distance_scale']==34
    with pytest.raises(ValueError,match='config|signature'):
        runner.prepare(out,cache,env,encode,signature,objective='ranking',
            fitting=dict(updates=2,batch_size=4,bottleneck=3,lr=1e-3))
    path=out/'config.json';config=json.loads(path.read_text());config['distance_scale']=6
    path.write_text(json.dumps(config))
    run=json.loads((out/'run.json').read_text());run['files']['config.json']=file_sha256(path)
    (out/'run.json').write_text(json.dumps(run))
    with pytest.raises(ValueError,match='config|signature|scale'):
        runner.stage(out,cache,1,render=False)
