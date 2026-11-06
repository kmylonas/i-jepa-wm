"""Paired world-model planning with a frozen manually fitted ranking cost."""
import importlib
import importlib.util

import numpy as np
import pytest
import torch

from test_multigoal_ranking import ranking_prepared,multigoal_prepared,prepared_radial


def module(name='ranking_planning_protocol'):
    full='src.mylonas_ijepa_wm.'+name
    assert importlib.util.find_spec(full), f'{name} missing'
    return importlib.import_module(full)


def manual_design():
    from src.mylonas_ijepa_wm.radial_geometry import make_design
    d=make_design();points=list(d['points']);ids=[d['goal_id']]
    goals=[[50,42],[46,38],[40,18],[50,18],[56,10],[50,8]]
    for goal in goals[1:]:points.append(goal);ids.append(len(points)-1)
    return dict(points=points,goal_ids=ids,train_ids=d['stages'][3]['training_ids']+[d['goal_id']])


def test_fixed_manifest_has_stratified_new_starts_in_fitted_right_region():
    d=manual_design();mod=module();manifest=mod.make_manifest(d)
    assert manifest==mod.make_manifest(d)
    assert len(manifest['goals'])==6 and len(manifest['evaluation'])==48
    assert manifest['planning']['planning_horizon']==5 and manifest['planning']['max_macro_steps']==20
    offered=np.asarray([d['points'][i] for i in d['train_ids']+d['goal_ids']])
    for gi in range(6):
        rows=[c for c in manifest['evaluation'] if c['goal_id']==gi]
        assert len(rows)==8
        assert [sum(c['band']==b for c in rows) for b in ('near','medium','long')]==[2,3,3]
    for c in manifest['evaluation']:
        start,goal=np.asarray(c['start']),np.asarray(c['goal'])
        for p in (start,goal):
            assert 38<=p[0]<=56 and 8<=p[1]<=55
            assert np.linalg.norm(p-[50,42])<=34+1e-5
        distance=np.linalg.norm(start-goal)
        lo,hi={'near':(6,10),'medium':(10,18),'long':(18,30)}[c['band']]
        assert lo<=distance<=hi and np.linalg.norm(offered-start,axis=1).min()>=.5
    starts=np.asarray([c['start'] for c in manifest['evaluation']]);dist=np.linalg.norm(starts[:,None]-starts[None,:],axis=2)
    assert (dist+np.eye(len(starts))*100).min()>=.5


def test_paired_summary_preserves_failures_and_both_success_efficiency():
    mod=module();cases=mod.make_manifest(manual_design())['evaluation'][:3]
    def row(case,success,steps,distance,path,excursion=False):
        return dict(case=case,case_id=case['case_id'],success=success,micro_steps=steps,
            final_goal_distance=distance,path_length=path,ever_outside_fitting=excursion)
    raw=[row(cases[0],True,8,2,10),row(cases[1],True,10,3,15),row(cases[2],False,100,12,60)]
    adapted=[row(cases[0],True,5,1,8),row(cases[1],False,100,8,50),row(cases[2],True,7,2,12,True)]
    r=mod.summarize(raw,adapted)['overall']
    assert r['wins']==1 and r['losses']==1 and r['both_success']==1 and r['both_fail']==0
    assert r['conditions']['adapted']['successes']==2 and r['conditions']['adapted']['in_region_successes']==1
    assert r['both_success_efficiency']['episodes']==1
    assert r['both_success_efficiency']['mean_steps_delta']==-3
    assert r['both_success_efficiency']['mean_path_length_delta']==-2
    assert mod.summarize([raw[2]],[adapted[1]|dict(case=cases[2],case_id=cases[2]['case_id'])])['overall']['both_success_efficiency']['mean_steps_delta'] is None
    with pytest.raises(ValueError,match='paired|case'):mod.summarize(raw,adapted[:-1])


@pytest.fixture
def frozen_ranking(ranking_prepared):
    runner,_,_,source,cache,_,_,_,_=ranking_prepared
    runner.train(source,cache,'multi')
    return source


def test_readonly_loader_accepts_historical_code_and_rejects_checkpoint_corruption(frozen_ranking,monkeypatch):
    from src.mylonas_ijepa_wm import radial_multigoal_experiment as data_engine
    mod=module();monkeypatch.setattr(data_engine,'source_digest',lambda _: 'new-planning-source')
    adapter,metadata,design=mod.load_ranking_source(frozen_ranking)
    assert not any(p.requires_grad for p in adapter.parameters())
    assert metadata['root']==str(frozen_ranking.resolve()) and len(design['goal_ids'])==4
    path=frozen_ranking/'branches/multi/adapter.pt';path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='hash|artifact|checkpoint'):mod.load_ranking_source(frozen_ranking)
