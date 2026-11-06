"""Heldout references and matched single/multi-goal distance supervision."""
import importlib
import importlib.util

import numpy as np
import pytest
import torch

from test_radial_ranking import prepared_radial


def module(name='radial_multigoal'):
    full='src.mylonas_ijepa_wm.'+name
    assert importlib.util.find_spec(full), f'{name} missing'
    return importlib.import_module(full)


def test_split_excludes_reserved_coordinates_images_and_identical_training_images():
    import json
    # Node4 aliases goal3; nodes1/5 share pixels; pair6/8 exceeds range.
    points=[[50,42],[50,40],[48,38],[50,18],[50,18.1],[50,39.9],[50,8],[51,20],[50,55]]
    hashes=['a','b','c','goal','goal','b','f','query','h']
    split=module().make_split(points,[0,1,2,3,4,5,6,8],[0,3],hashes)
    assert split['train_ids']==[0,1,2,6,8]
    assert split['removed_candidate_ids']==[3,4,5]
    assert split['heldout_mask']==[False]*7+[True,False]
    single=split['single_pairs'];multi=split['multi_pairs']
    assert {tuple(p) for p in single}=={(1,0),(2,0),(6,0),(8,0)}
    assert len({tuple(p) for p in multi})==len(multi)
    for a,b in multi:
        assert a<b and a in split['train_ids'] and b in split['train_ids']
        assert 0<np.linalg.norm(np.subtract(points[a],points[b]))<=34+1e-5
    assert (0,6) in map(tuple,multi)
    assert (6,8) not in map(tuple,multi)
    assert json.loads(json.dumps(split))==split


def test_pair_costs_use_both_endpoints_and_target_linear_distance():
    mod=module()
    tokens=torch.tensor([[[0.,0.]],[[1.,0.]],[[2.,1.]]])
    pairs=[[1,0],[2,1]]
    adapter=torch.nn.Linear(2,2,bias=False)
    with torch.no_grad():adapter.weight.copy_(torch.eye(2))
    values=mod.pair_costs(adapter,tokens,pairs)
    assert torch.allclose(values,torch.tensor([.5,1.]))
    values.sum().backward()
    assert torch.allclose(adapter.weight.grad,torch.tensor([[2.,1.],[1.,1.]]))
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


@pytest.fixture
def multigoal_prepared(prepared_radial):
    runner,old,_,_,env,encode,_=prepared_radial
    source,source_cache=old.parent/'distance-source',old.parent/'distance-cache'
    signature=dict(encoder_sha256='synthetic',renderer_commit='pinned',precision='fp32',device_type='cpu')
    runner.prepare(source,source_cache,env,encode,signature,objective='distance',batch_size=64,
        fitting=dict(updates=2,batch_size=4,bottleneck=3,lr=1e-3))
    for stage in (1,2,3,4):runner.stage(source,source_cache,stage,render=False)
    mod=module('radial_multigoal_experiment')
    output,cache=old.parent/'multigoal',old.parent/'multi-cache'
    prepared=mod.prepare(source,output,cache,env,encode,signature,
        new_goals=[[46,38],[50,18],[50,8]],batch_size=64,
        fitting=dict(updates=3,batch_size=4,bottleneck=3,lr=1e-3,seed=0))
    return mod,source,output,cache,prepared,env,encode,signature


def test_matched_branches_shared_evaluation_and_training_only_calibration(multigoal_prepared):
    from src.mylonas_ijepa_wm.curriculum_runtime import file_sha256
    mod,source,output,cache,prepared,*_=multigoal_prepared
    before={str(p.relative_to(source)):file_sha256(p) for p in source.rglob('*') if p.is_file()}
    design=prepared['design'];reserved=set(design['goal_ids'][1:])
    data=torch.load(output/'training_data.pt',weights_only=True)
    assert not reserved.intersection(data['point_ids'])
    for branch in ('single','multi'):
        saved=mod.train(output,cache,branch)
        assert sum(saved['endpoint_counts'])==24 and sum(saved['pair_sample_counts'])==12
        assert saved['initialization']=='identity'
    r=mod.evaluate(output,cache,render=False)
    assert r['source_unchanged'] and r['distance_scale']==34
    assert r['original_goal_consistency']['raw_token_mse']==0
    assert set(r['methods'])=={'single','multi','historical'}
    for rows in r['methods'].values():
        assert len(rows)==4 and all(g['raw_self_cost']==0 and g['adapted_self_cost']<1e-7 for g in rows)
        assert all(g['common_mask']==rows[0]['common_mask'] for g in rows)
        assert all(g['heldout_common']['count']==rows[0]['heldout_common']['count']>0 for g in rows)
    assert r['methods']['single'][0]['common_mask']==r['methods']['multi'][0]['common_mask']
    raw=module().pair_costs(None,data['tokens'],data['pairs']['multi']).numpy()
    xy=np.asarray(data['positions']);pairs=np.asarray(data['pairs']['multi'])
    distances=np.linalg.norm(xy[pairs[:,0]]-xy[pairs[:,1]],axis=1)
    assert r['raw_calibration']==pytest.approx(float(raw@distances/(raw@raw)))
    assert before=={str(p.relative_to(source)):file_sha256(p) for p in source.rglob('*') if p.is_file()}


def test_completed_reuse_and_report_survive_local_cache_deletion(multigoal_prepared,monkeypatch):
    import shutil
    mod,source,output,cache,prepared,env,_,signature=multigoal_prepared
    for branch in ('single','multi'):mod.train(output,cache,branch)
    first=mod.evaluate(output,cache,render=False)
    monkeypatch.setattr(mod,'fit',lambda *a,**k:(_ for _ in ()).throw(AssertionError('refit')))
    mod.train(output,cache,'multi')
    def forbidden(frames):raise AssertionError('re-encoding')
    assert mod.prepare(source,output,cache,env,forbidden,signature,
        new_goals=[[46,38],[50,18],[50,8]],batch_size=64,
        fitting=dict(updates=3,batch_size=4,bottleneck=3,lr=1e-3,seed=0))['design']==prepared['design']
    shutil.rmtree(cache)
    assert mod.report(output,render=False)==first
    assert mod.evaluate(output,cache,render=False)==first


def test_changed_preparation_cache_checkpoint_and_report_are_rejected(multigoal_prepared):
    mod,source,output,cache,_,env,encode,signature=multigoal_prepared
    with pytest.raises(ValueError,match='config|inputs|signature'):
        mod.prepare(source,output,cache,env,encode,signature,new_goals=[[48,38]])
    for branch in ('single','multi'):mod.train(output,cache,branch)
    mod.evaluate(output,cache,render=False)
    path=output/'branches/multi/adapter.pt';original=path.read_bytes();path.write_bytes(original+b'changed')
    with pytest.raises(ValueError,match='hash|artifact|checkpoint'):mod.report(output,render=False)
    path.write_bytes(original)
    path=cache/'tokens.pt';original=path.read_bytes();path.write_bytes(original+b'changed')
    with pytest.raises(ValueError,match='hash|cache'):mod.train(output,cache,'single')
    path.write_bytes(original)
    path=output/'report.json';path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='hash|artifact'):mod.report(output,render=False)


def test_plots_regenerate_without_local_cache_and_show_all_goals(multigoal_prepared):
    import shutil
    mod,_,output,cache,*_=multigoal_prepared
    for branch in ('single','multi'):mod.train(output,cache,branch)
    mod.evaluate(output,cache,render=False)
    shutil.rmtree(cache)
    r=mod.report(output,render=True)
    assert len(list(output.glob('*.pdf')))==7
    for name in ('coverage','training_loss','goal_summary','goal_3_heatmaps'):
        assert (output/f'{name}.png').stat().st_size>1000
    assert r['methods']['multi'][0]['reference_status']=='seen'
    assert all(g['reference_status']=='withheld' for g in r['methods']['multi'][1:])


def test_multigoal_notebook_compiles_commands_and_executes_actual_reporting(multigoal_prepared,monkeypatch):
    import ast,json,sys,types,subprocess
    from pathlib import Path
    notebook=Path(__file__).resolve().parents[1]/'notebooks/radial_multigoal_colab.ipynb'
    assert notebook.is_file(), 'multi-goal notebook missing'
    n=json.loads(notebook.read_text())
    for c in n['cells']:
        if c['cell_type']=='code':compile(''.join(c['source']),'<notebook>','exec')
    mod,_,output,cache,*_=multigoal_prepared
    for branch in ('single','multi'):mod.train(output,cache,branch)
    mod.evaluate(output,cache,render=True)
    context=dict(sys=sys,OUTPUT=output,CACHE=cache,REPO=Path.cwd(),json=json,
        SOURCE_RUN=Path('source'),ENCODER=Path('encoder'),DINO_ROOT=Path('renderer'))
    tree=ast.parse(''.join(n['cells'][3]['source']))
    nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='command']
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<command>','exec'),context)
    assert '--cache-dir' not in context['command']('report')
    assert context['command']('train','multi')[-2:]==['--branch','multi']
    displayed=[]
    class Image:
        def __init__(self,filename):assert Path(filename).is_file();displayed.append(filename)
    ip=types.ModuleType('IPython');display=types.ModuleType('IPython.display')
    display.display=lambda image:None;display.Image=Image
    monkeypatch.setitem(sys.modules,'IPython',ip);monkeypatch.setitem(sys.modules,'IPython.display',display)
    context['shell']=lambda command,cwd=None:subprocess.run(list(map(str,command)),cwd=cwd,check=True)
    context['time']=__import__('time')
    for index in (9,11,13):exec(''.join(n['cells'][index]['source']),context)
    assert len(displayed)==7
