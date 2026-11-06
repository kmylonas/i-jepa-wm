"""Controlled radial inputs, genuinely held-out probes and cumulative fitting."""
import importlib
import importlib.util

import numpy as np
import pytest
import torch


def module(name):
    full='src.mylonas_ijepa_wm.'+name
    assert importlib.util.find_spec(full), f'{name} missing'
    return importlib.import_module(full)


def test_design_has_cumulative_populations_and_omits_cross_wall_or_boundary_points():
    design=module('radial_geometry').make_design()
    assert [len(s['training_ids']) for s in design['stages']]==[48,90,105,115]
    assert [s['outer_radius'] for s in design['stages']]==[6,14,22,34]
    points=np.asarray(design['points'])
    assert points[design['goal_id']].tolist()==[50.,42.]
    assert (points[:,0]>35).all() and (points<60).all() and (points[:,1]>4).all()
    # The invalid r10 rightward point must be omitted, never clipped to the wall.
    assert not any(t['radius']==10 and t['angle']==0 for t in design['training'])
    ids=[set(s['training_ids']) for s in design['stages']]
    assert all(a<b for a,b in zip(ids,ids[1:]))
    assert any(c['kind']=='withheld_direction' and c['angle']==7.5 for c in design['curves'])
    train=set(ids[-1])
    midpoint=[i for c in design['curves'] if c['kind']=='trained_direction'
              for i,r in zip(c['point_ids'],c['radii']) if r==4]
    assert len(midpoint)==24 and not train.intersection(midpoint)
    assert len(design['families'])==50
    for f in design['families']:
        center=next(p for p in f['profiles'] if p['offset']==0)
        assert len(center['point_ids'])==11
        assert len(set(center['point_ids'][1:-1])&train)==0


def test_training_pairs_include_cross_direction_examples_and_exclude_equal_radius():
    pairs=module('radial_training').training_pairs(torch.tensor([2.,2.,6.,10.]))
    assert set(map(tuple,pairs.tolist()))=={(2,0),(2,1),(3,0),(3,1),(3,2)}
    with pytest.raises(ValueError,match='pair|radius'):
        module('radial_training').training_pairs(torch.tensor([2.,2.]))


def test_rank_only_fit_improves_order_and_warm_starts_without_mutating_incoming():
    mod=module('radial_training')
    tokens=torch.tensor([[[3.,0,1,0]],[[2.,1,0,0]],[[1.,0,0,1]]])
    radii=torch.tensor([2.,6,10])
    adapter,training=mod.fit(tokens,radii,torch.zeros(1,4),seed=7,updates=100,batch_size=8,
        bottleneck=3,lr=.01)
    assert training['rank_accuracy_after']>training['rank_accuracy_before']
    assert training['rank_accuracy_after']==1
    assert training['history'][-1]['full_rank_loss'] < training['full_rank_loss_before']
    assert training['anchor_weight']==0 and training['initialization']=='identity'
    assert sum(training['sample_counts'])==1600
    before={k:v.clone() for k,v in adapter.state_dict().items()}
    updated,more=mod.fit(tokens,radii,torch.zeros(1,4),seed=8,updates=2,batch_size=2,
        bottleneck=3,initial_adapter=adapter)
    assert more['initialization']=='previous_checkpoint' and more['optimizer_reset']
    assert all(torch.equal(v,adapter.state_dict()[k]) for k,v in before.items())
    assert updated is not adapter


@pytest.fixture
def prepared_radial(tmp_path):
    from types import SimpleNamespace
    mod=module('radial_experiment')
    env=SimpleNamespace(env=SimpleNamespace(img_size=65,border_padding=7.4,
        wall_config=SimpleNamespace(border_wall_loc=5)),reset=lambda *a,**k:None,
        _render_at=lambda xy:np.broadcast_to(np.array([round(xy[0]*3),round(xy[1]*3),13],np.uint8),(2,2,3)).copy())
    def encode(frames):
        xy=torch.as_tensor(frames[:,0,0,:2]).float()/3
        return torch.cat([xy,torch.zeros(len(xy),1),torch.ones(len(xy),1)],dim=1)[:,None,:]
    output,cache=tmp_path/'output',tmp_path/'cache'
    signature=dict(encoder='synthetic',renderer='pinned',precision='fp32')
    result=mod.prepare(output,cache,env,encode,signature,
        fitting=dict(updates=3,batch_size=2,bottleneck=3,lr=1e-3))
    return mod,output,cache,result,env,encode,signature


def test_preparation_uses_fresh_goal_and_reuses_only_identical_cache(prepared_radial):
    mod,out,cache,result,env,encode,signature=prepared_radial
    goal=torch.load(out/'goal.pt',weights_only=True)['tokens']
    assert torch.allclose(goal,torch.tensor([[50.,42.,0.,1.]]))
    assert result['raw_costs'][result['design']['goal_id']]==0
    assert (out/'goal.png').is_file()
    def forbidden(frames):raise AssertionError('cache reuse must not encode again')
    again=mod.prepare(out,cache,env,forbidden,signature,
        fitting=dict(updates=3,batch_size=2,bottleneck=3,lr=1e-3))
    assert again['config']==result['config']
    with pytest.raises(ValueError,match='signature|config'):
        mod.prepare(out,cache,env,forbidden,dict(signature,precision='bf16'))
    with pytest.raises(ValueError,match='separate|overlap'):
        mod.prepare(out,out/'nested',env,forbidden,signature)


def test_four_stages_only_fit_offered_points_and_keep_probe_cohort_fixed(prepared_radial,monkeypatch):
    mod,out,cache,prepared,*_=prepared_radial
    design=prepared['design']
    previous=None
    for index,n in enumerate((48,90,105,115),1):
        report=mod.stage(out,cache,index)
        assert report['training_positions']==n and report['primitive_interactions']==0
        assert set(report['actual_fitting_ids'])<=set(design['stages'][index-1]['training_ids'])|{design['goal_id']}
        assert report['incoming_sha256']==previous
        assert report['initialization']==('identity' if index==1 else 'previous_checkpoint')
        previous=report['checkpoint_sha256']
        assert report['matched_family_ids']==design['matched_family_ids']
        assert report['generalization']['withheld_direction']['inside']['raw']['comparisons']>0
        outside=report['generalization']['withheld_direction']['outside']['adapted']
        assert outside['comparisons']>0 if index<4 else outside['up_fraction'] is None
    # Completed-stage reuse cannot silently retrain.
    monkeypatch.setattr(mod,'fit',lambda *a,**k: (_ for _ in ()).throw(AssertionError('refit')))
    assert mod.stage(out,cache,4)['checkpoint_sha256']==previous


def test_changed_token_shard_or_checkpoint_blocks_fitting(prepared_radial):
    import json
    mod,out,cache,prepared,*_=prepared_radial
    mod.stage(out,cache,1)
    checkpoint=out/'stages/s1/adapter.pt'
    checkpoint.write_bytes(checkpoint.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='checkpoint|hash|artifact'):
        mod.stage(out,cache,2)
    # Cache corruption must be rejected even before trying to reuse stage1.
    shard=next((cache/'shards').glob('*.pt'))
    shard.write_bytes(shard.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='cache|hash|shard'):
        mod.stage(out,cache,1)


def test_changed_source_signature_rejects_cached_features_before_fit(prepared_radial,monkeypatch):
    mod,out,cache,*_=prepared_radial
    monkeypatch.setattr(mod,'source_digest',lambda path:'changed-source',raising=False)
    with pytest.raises(ValueError,match='source|signature|config'):
        mod.stage(out,cache,1)


def test_radial_interpolation_has_comparisons_between_two_unfitted_positions(prepared_radial):
    mod,out,cache,*_=prepared_radial
    report=mod.stage(out,cache,1,render=False)
    # Between fitted radii2 and6, retain multiple unseen interior radii so
    # comparisons of two genuinely unfitted observations have a denominator.
    assert report['generalization']['trained_direction']['inside']['raw']['heldout_comparisons']>0


@pytest.mark.parametrize('alias', ['stages', 'figures', 'cache_file', 'dangling'])
def test_descendant_symlinks_rejected_before_fit_without_external_writes(prepared_radial,monkeypatch,alias):
    mod,out,cache,*_=prepared_radial
    external=out.parent/'unrelated'
    external.mkdir()
    (external/'sentinel.txt').write_text('another experiment')
    if alias=='cache_file':
        link=next((cache/'shards').glob('*.pt'))
        target=external/'original.pt'
        target.write_bytes(link.read_bytes())
        link.unlink()
    else:
        link=out/'stages' if alias=='stages' else out/'stages/s1/figures'
        link.parent.mkdir(parents=True,exist_ok=True)
        target=external/'missing' if alias=='dangling' else external
    link.symlink_to(target,target_is_directory=alias!='cache_file')
    before={str(p.relative_to(external)):p.read_bytes() for p in external.rglob('*') if p.is_file()}
    def forbidden(*args,**kwargs):raise AssertionError('unsafe alias reached fitting')
    monkeypatch.setattr(mod,'fit',forbidden)
    try:
        with pytest.raises(ValueError,match='symlink|alias'):
            mod.stage(out,cache,1)
    finally:
        assert before=={str(p.relative_to(external)):p.read_bytes() for p in external.rglob('*') if p.is_file()}
