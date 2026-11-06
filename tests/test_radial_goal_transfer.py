"""Frozen historical adapters evaluated against genuinely moved goal references."""
import importlib
import importlib.util
import json

import numpy as np
import pytest
import torch

from test_radial_ranking import prepared_radial


def module():
    name='src.mylonas_ijepa_wm.radial_goal_transfer'
    assert importlib.util.find_spec(name), 'goal-transfer diagnostic missing'
    return importlib.import_module(name)


@pytest.fixture
def transfer_source(prepared_radial,monkeypatch):
    runner,oldout,oldcache,_,env,encode,_=prepared_radial
    source,cache=oldout.parent/'distance-source',oldcache.parent/'old-local-cache'
    signature=dict(encoder_sha256='synthetic',renderer_commit='pinned',precision='fp32',device_type='cpu')
    with monkeypatch.context() as patch:
        patch.setattr(runner,'source_digest',lambda p:'historical-fitting-source')
        runner.prepare(source,cache,env,encode,signature,objective='distance',batch_size=64,
            fitting=dict(updates=2,batch_size=4,bottleneck=3,lr=1e-3))
        runner.stage(source,cache,1,render=False);runner.stage(source,cache,2,render=False)
    # Historical local cache is not needed for this diagnostic.
    import shutil
    shutil.rmtree(cache)
    return source,oldout.parent/'transfer-output',env,encode,signature


def test_fresh_goal_transfer_keeps_adapter_source_and_scale_frozen(transfer_source):
    source,out,env,encode,signature=transfer_source
    before={str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    r=module().evaluate(source,out,2,[46,38],env,encode,signature,render=True)
    assert before=={str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    assert r['source']['radius']==14 and r['adapter_unchanged'] and r['distance_scale']==34
    assert [g['goal'] for g in r['goals']]==[[50.,42.],[46.,38.]]
    assert r['goals'][0]['common']['count']==r['goals'][1]['common']['count']>0
    for g in r['goals']:
        assert g['raw_self_cost']==0 and g['adapted_self_cost']<1e-8
    goals=torch.load(out/'goals.pt',weights_only=True)
    assert not torch.equal(goals['tokens'][0],goals['tokens'][1])
    assert r['original_goal_consistency']['raw_token_mse']<1e-10
    assert len(list(out.glob('*.png')))>=4
    def forbidden(*args):raise AssertionError('reuse encoded again')
    assert module().evaluate(source,out,2,[46,38],env,forbidden,signature)['adapter_unchanged']
    (out/'goal_1_heatmaps.png').unlink()
    module().evaluate(source,out,2,[46,38],env,forbidden,signature)
    assert (out/'goal_1_heatmaps.png').is_file()


def test_source_chain_kind_and_diagnostic_changes_are_rejected(transfer_source):
    source,out,env,encode,signature=transfer_source
    with pytest.raises(ValueError,match='Euclidean-distance'):
        module().load_source(source.parent/'output',1)
    module().evaluate(source,out,2,[46,38],env,encode,signature,render=False)
    with pytest.raises(ValueError,match='config|signature|inputs'):
        module().evaluate(source,out,2,[48,38],env,encode,signature,render=False)
    with pytest.raises(ValueError,match='encoder|precision|signature'):
        module().evaluate(source,out.parent/'other',2,[46,38],env,encode,dict(signature,precision='bf16'))
    with pytest.raises(ValueError,match='region|goal|radius'):
        module().evaluate(source,out.parent/'outside',2,[50,8],env,encode,signature)
    with pytest.raises(ValueError,match='overlap|separate'):
        module().evaluate(source,source/'nested',2,[46,38],env,encode,signature)
    checkpoint=source/'stages/s1/adapter.pt';checkpoint.write_bytes(checkpoint.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='hash|checkpoint|artifact'):
        module().load_source(source,2)


def test_grid_metrics_share_cohort_and_do_not_refit_raw_calibration():
    # A perfect distance cost at each goal must have zero error on the same
    # original-region/known-distance ball positions, regardless of the goal.
    mod=module()
    points=np.array([[50,42],[46,38],[48,40],[58,58]],float)
    goals=np.array([[50,42],[46,38]],float)
    truth=np.linalg.norm(points[None]-goals[:,None],axis=2)
    r=mod.measure(points,goals,truth/10,truth/34,14,10,np.array([False,True,True,True]),[])
    assert [g['common']['count'] for g in r]==[3,3]
    assert [g['heldout_common']['count'] for g in r]==[2,2]
    assert all(g['common']['adapted']['mae']<1e-12 for g in r)
    assert all(g['common']['raw_calibrated']['mae']<1e-12 for g in r)
    assert all(g['directional']['comparisons']==0 and g['directional']['adapted_up_fraction'] is None for g in r)


def test_reference_coordinates_encoded_once_and_reused_for_self_cost(transfer_source):
    source,out,env,encode,signature=transfer_source
    seen=[]
    def batch_sensitive(frames):
        seen.extend(f.tobytes() for f in frames)
        return encode(frames)+len(frames)*1e-4
    r=module().evaluate(source,out,2,[46,38],env,batch_sensitive,signature,render=False)
    assert len(seen)==len(set(seen))
    assert all(g['raw_self_cost']==0 for g in r['goals'])
