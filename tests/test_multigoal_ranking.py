"""Manual ordinal supervision with unseen goals and no distance regression."""
import importlib
import importlib.util

import numpy as np
import pytest
import torch

from test_radial_multigoal import multigoal_prepared,prepared_radial


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
    chosen=np.asarray(population)[t['sampled_triplet_ids']]
    assert t['far_counts']==np.bincount(chosen[:,1],minlength=5).tolist()
    assert t['near_counts']==np.bincount(chosen[:,2],minlength=5).tolist()
    assert (np.asarray(t['reference_counts'])+t['far_counts']+np.asarray(t['near_counts'])).tolist()==t['endpoint_counts']
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
    assert all(g['heldout_directional']['adapted_decreasing_fraction']==1 for g in rows)
    assert all('heldout_common' not in g for g in rows)
    for values in (np.zeros_like(truth),-truth):
        r=mod.ordinal_rows(points,goals,truth,values,heldout,edges)
        key='tie_fraction' if (values==0).all() else 'reversed_fraction'
        assert all(g['adapted_rank'][key]==1 for g in r)
        expected=1 if key=='tie_fraction' else 0
        assert all(g['heldout_directional']['adapted_tie_fraction']==expected for g in r)
    empty=mod.ordinal_rows(points,goals,truth,truth,[False]*5,edges)
    assert all(g['adapted_rank']['comparisons']==0 and g['adapted_rank']['correct_fraction'] is None for g in empty)
    assert all(g['heldout_directional']['adapted_tie_fraction'] is None and g['heldout_directional']['adapted_decreasing_fraction'] is None for g in empty)


@pytest.mark.parametrize('population,scale',[([],1),([[0,0,1]],1),([[0,1,8]],1),([[0,1,2]],0)])
def test_invalid_triplets_or_scale_reject(population,scale):
    with pytest.raises(ValueError,match='triple|scale'):
        module().fit(torch.zeros(3,1,2),population,scale,updates=1)


@pytest.fixture
def ranking_prepared(multigoal_prepared):
    common,source,baseline,_,_,env,encode,signature=multigoal_prepared
    common.train(baseline,baseline.parent/'multi-cache','multi')
    mod=module('multigoal_ranking_experiment')
    out,cache=baseline.parent/'ranking',baseline.parent/'rank-cache'
    prepared=mod.prepare(source,baseline,out,cache,env,encode,signature,
        new_goals=[[46,38],[50,18],[50,8]],batch_size=64,
        fitting=dict(updates=3,batch_size=4,bottleneck=3,lr=1e-3,seed=0))
    return mod,source,baseline,out,cache,prepared,env,encode,signature


def test_complete_ranking_run_preserves_sources_and_evaluates_same_unseen_cohort(ranking_prepared):
    import json
    from src.mylonas_ijepa_wm.curriculum_runtime import file_sha256
    mod,source,baseline,out,cache,prepared,*_=ranking_prepared
    def hashes(root):return {str(p.relative_to(root)):file_sha256(p) for p in root.rglob('*') if p.is_file()}
    before=hashes(source),hashes(baseline)
    design=prepared['design'];data=torch.load(out/'data/training_data.pt',weights_only=True)
    population=torch.load(out/'triplets.pt',weights_only=True)
    assert not set(data['point_ids']).intersection(design['goal_ids'][1:])
    for branch in ('single','multi'):
        assert population[branch].shape[1]==3
        t=mod.train(out,cache,branch)
        assert sum(t['endpoint_counts'])==36 and len(t['sampled_triplet_ids'])==12
        assert t['scale']==prepared['config']['scale']
    r=mod.evaluate(out,cache,render=False)
    assert r['references_unchanged'] and r['original_goal_consistency']['raw_token_mse']==0
    assert set(r['methods'])=={'single','multi','distance'}
    assert 'distance_mae' not in json.dumps(r) and 'distance_scale' not in r
    first=r['methods']['multi'][0]
    assert first['heldout_count']>0
    for rows in r['methods'].values():
        assert len(rows)==4
        assert all(g['common_mask']==first['common_mask'] and g['heldout_count']==first['heldout_count'] for g in rows)
        assert all(g['adapted_self_cost']<1e-7 and g['raw_self_cost']==0 for g in rows)
        assert rows[0]['reference_status']=='seen' and all(g['reference_status']=='withheld' for g in rows[1:])
    assert before==(hashes(source),hashes(baseline))


def test_ranking_reuse_report_cache_loss_and_corruption(ranking_prepared,monkeypatch):
    import shutil
    mod,_,baseline,out,cache,*_=ranking_prepared
    for branch in ('single','multi'):mod.train(out,cache,branch)
    r=mod.evaluate(out,cache,render=False)
    monkeypatch.setattr(mod,'fit',lambda *a,**k:(_ for _ in ()).throw(AssertionError('refit')))
    mod.train(out,cache,'multi')
    shutil.rmtree(cache)
    assert mod.report(out,render=False)==r
    assert mod.evaluate(out,cache,render=False)==r
    p=baseline/'branches/multi/adapter.pt';p.write_bytes(p.read_bytes()+b'corrupt')
    with pytest.raises(ValueError,match='hash|artifact|checkpoint|reference'):mod.report(out,render=False)


def test_frozen_reference_accepts_historical_code_and_rejects_wrong_data(multigoal_prepared,monkeypatch):
    common,source,baseline,basecache,_,env,encode,signature=multigoal_prepared
    common.train(baseline,basecache,'multi')
    mod=module('multigoal_ranking_experiment')
    monkeypatch.setattr(common,'source_digest',lambda _: 'new-ranking-code')
    out,cache=baseline.parent/'new-rank',baseline.parent/'new-cache'
    p=mod.prepare(source,baseline,out,cache,env,encode,signature,
        new_goals=[[46,38],[50,18],[50,8]],batch_size=64,
        fitting=dict(updates=1,batch_size=2,bottleneck=3,lr=1e-3,seed=0))
    assert p['config']['distance_reference']['root']==str(baseline.resolve())
    with pytest.raises(ValueError,match='goal|geometry|fitting|reference'):
        mod.prepare(source,baseline,baseline.parent/'wrong',baseline.parent/'wrong-cache',env,encode,signature,
            new_goals=[[46,38]],batch_size=64)


def test_ranking_plots_and_actual_notebook_reporting_after_cache_loss(ranking_prepared,monkeypatch):
    import ast,json,shutil,subprocess,sys,types
    from pathlib import Path
    notebook=Path(__file__).resolve().parents[1]/'notebooks/radial_multigoal_ranking_colab.ipynb'
    assert notebook.is_file(), 'ranking notebook missing'
    n=json.loads(notebook.read_text())
    for c in n['cells']:
        if c['cell_type']=='code':compile(''.join(c['source']),'<cell>','exec')
    mod,_,_,out,cache,*_=ranking_prepared
    for branch in ('single','multi'):mod.train(out,cache,branch)
    mod.evaluate(out,cache,render=False);shutil.rmtree(cache)
    mod.report(out,render=True)
    assert len(list(out.glob('*.pdf')))==7
    (out/'goal_3_heatmaps.png').unlink();mod.report(out)
    context=dict(sys=sys,OUTPUT=out,CACHE=cache,REPO=Path.cwd(),json=json,time=__import__('time'),
        SOURCE_RUN=Path('source'),DISTANCE_BASELINE=Path('baseline'),ENCODER=Path('encoder'),DINO_ROOT=Path('renderer'))
    nodes=[a for a in ast.parse(''.join(n['cells'][3]['source'])).body if isinstance(a,ast.FunctionDef) and a.name=='command']
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<command>','exec'),context)
    assert '--distance-baseline' in context['command']('prepare')
    assert '--cache-dir' not in context['command']('report')
    context['shell']=lambda cmd,cwd=None:subprocess.run(list(map(str,cmd)),cwd=cwd,check=True)
    images=[]
    class Image:
        def __init__(self,filename):assert Path(filename).is_file();images.append(filename)
    ip=types.ModuleType('IPython');disp=types.ModuleType('IPython.display')
    disp.display=lambda image:None;disp.Image=Image
    monkeypatch.setitem(sys.modules,'IPython',ip);monkeypatch.setitem(sys.modules,'IPython.display',disp)
    for i in (9,11,13):exec(''.join(n['cells'][i]['source']),context)
    assert len(images)==7
