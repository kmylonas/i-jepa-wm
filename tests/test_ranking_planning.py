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


def test_real_cem_uses_each_goal_identity_matches_raw_and_histories_stay_raw(tmp_path):
    from types import SimpleNamespace
    from src.mylonas_ijepa_wm.cost_adapter import ResidualTokenAdapter
    from src.mylonas_ijepa_wm.curriculum_protocol import PLANNING
    from src.mylonas_ijepa_wm.environment_configs import EnvironmentConfig
    from src.mylonas_ijepa_wm.environment_adapters import StepResult
    from src.mylonas_ijepa_wm.embed_frames import IJEPA_MEAN,IJEPA_STD
    mod=module('ranking_planning');seen=[]
    class Encoder(torch.nn.Module):
        def forward(self,images):
            xy=(images[:,:2,0,0]*IJEPA_STD[0,:2,0,0]+IJEPA_MEAN[0,:2,0,0])*255
            return xy[:,None,:]
    class Model(torch.nn.Module):
        num_hist=1
        def forward(self,latents,states,actions):
            if not seen:seen.append(latents[0,0].clone())
            next_state=states+actions.reshape(*actions.shape[:-1],5,2).sum(dim=-2)
            return next_state.unsqueeze(-2),next_state
    class Env:
        action_low=np.full(2,-1.,np.float32);action_high=-action_low;layout=(32,30)
        def _render_at(self,p):return np.broadcast_to(np.array([*np.rint(p).astype(np.uint8),0],np.uint8),(224,224,3)).copy()
        def reset(self,seed,**kwargs):
            self.xy=np.asarray(kwargs['start'],np.float32);self.goal=np.asarray(kwargs['goal'],np.float32)
            return StepResult(self.xy.copy(),self._render_at(self.xy),False)
        def render_goal(self):return self._render_at(self.goal)
        def step(self,a):
            self.xy+=a
            return StepResult(self.xy.copy(),self._render_at(self.xy),bool(np.linalg.norm(self.xy-self.goal)<4.5))
    runtime=mod.PlanningRuntime.__new__(mod.PlanningRuntime)
    runtime.root=tmp_path;runtime.runtime_id='tiny';runtime.device=torch.device('cpu');runtime.precision='fp32'
    runtime.args=SimpleNamespace(videos=False);runtime.env=Env();runtime.config=EnvironmentConfig(env='wall')
    runtime.encoder=Encoder();runtime.model=Model();runtime.metadata=dict(state_mean=torch.zeros(2),state_std=torch.ones(2))
    runtime.manifest=dict(planning=dict(PLANNING,planning_horizon=2,actions_per_plan=2,max_macro_steps=2,
        population=64,num_elites=8,cem_iterations=4,candidate_batch_size=16))
    runtime.load_encoder=lambda:runtime.encoder;runtime.load_world_model=lambda:runtime.model
    case=dict(case_id='tiny',goal_id=0,seed=7,layout=[32,30],start=[50,30],goal=[50,38])
    target=torch.tensor([[50.,38.]])
    a=runtime.run_case(case,target)
    identity=ResidualTokenAdapter(2,2)
    b=runtime.run_case(case,target,metric=identity)
    assert a['primitive_actions']==b['primitive_actions'] and a['achieved_positions']==b['achieved_positions']
    assert len(a['primitive_positions'])==a['micro_steps']+1 and a['path_length']>0
    assert torch.allclose(seen[0],torch.tensor([[50.,30.]]),atol=1e-4)
    with torch.no_grad():identity.up.bias.fill_(10)
    seen.clear();runtime.run_case(case,target,metric=identity)
    assert torch.allclose(seen[0],torch.tensor([[50.,30.]]),atol=1e-4)
    c=runtime.run_case(case|dict(goal=[50,22]),torch.tensor([[50.,22.]]))
    assert sum(v[1] for v in a['primitive_actions'])>0 and sum(v[1] for v in c['primitive_actions'])<0


class SyntheticRuntime:
    def __init__(self,runner,output,config,manifest,role,env,encode):
        from types import SimpleNamespace
        from src.mylonas_ijepa_wm.curriculum_protocol import digest,write_json
        self.root=output/('reference' if role=='reference' else f'conditions/{role}')
        self.manifest=manifest;self.device=torch.device('cpu');self.precision='fp32';self.role=role
        self.env=env;self.encode_frames=encode;self.args=SimpleNamespace(videos=False);self.visited=[];self.interrupt=None
        signature=dict(manifest=digest(manifest),models=config['models'],source=config['diagnostic_source'],
            dino_source='synthetic',dino_commit=config['source']['fitting_signature']['renderer_commit'],
            torch_version='test',device_type='cpu',precision='fp32',encoder_batch_size=8,
            runtime_variant=runner.runtime_variant(config,role,'cpu'))
        self.runtime_id=digest(signature);write_json(self.root/'runtime.json',dict(runtime_id=self.runtime_id,signature=signature))
    def run_case(self,case,target,metric=None,goal_frame_sha256=None):
        from src.mylonas_ijepa_wm.curriculum_runtime import InteractionJournal
        assert torch.allclose(target[0,:2],torch.tensor(case['goal']),atol=1e-4)
        assert (metric is None)==(self.role=='raw')
        self.visited.append(case['case_id'])
        positions=[(np.asarray(case['start'])*(1-t)+np.asarray(case['goal'])*t).tolist() for t in np.linspace(0,1,5)]
        with InteractionJournal(self.root,self.runtime_id,'evaluate',case['case_id']) as journal:
            for step in range(1,5):
                journal.record(step,[1,0])
                if self.interrupt==case['case_id'] and step==2:raise RuntimeError('interrupted')
        return dict(success=True,micro_steps=4,final_goal_distance=0.,mpc_updates=1,
            macro_steps=1,primitive_actions=[[1,0]]*4,macro_durations=[4],achieved_positions=[positions[0],positions[-1]],
            primitive_positions=positions,path_length=float(np.linalg.norm(np.diff(positions,axis=0),axis=1).sum()),
            ever_outside_fitting=False,attempt_id=journal.identity['attempt_id'],video_path=None)


@pytest.fixture
def planning_prepared(frozen_ranking,ranking_prepared,tmp_path):
    mod=module('ranking_planning');_,_,_,_,_,_,env,encode,_=ranking_prepared
    _,_,design=module().load_ranking_source(frozen_ranking)
    goals=[design['points'][i] for i in design['goal_ids']]
    out=tmp_path/'planning'
    config=mod.prepare(frozen_ranking,out,goals=goals,models=dict(encoder='synthetic',world_model='synthetic-world'))
    manifest=__import__('json').loads((out/'manifest.json').read_text())
    reference=SyntheticRuntime(mod,out,config,manifest,'reference',env,encode)
    mod.prepare_goals(frozen_ranking,out,reference)
    return mod,frozen_ranking,out,config,manifest,env,encode


def test_paired_evaluation_reuses_completed_cases_and_interruption_counts_attempts(planning_prepared):
    mod,source,out,config,manifest,env,encode=planning_prepared
    raw=SyntheticRuntime(mod,out,config,manifest,'raw',env,encode)
    mod.evaluate_condition(source,out,'raw',raw,limit=1)
    raw.interrupt=manifest['evaluation'][1]['case_id']
    with pytest.raises(RuntimeError,match='interrupted'):mod.evaluate_condition(source,out,'raw',raw)
    raw.interrupt=None;mod.evaluate_condition(source,out,'raw',raw)
    assert raw.visited.count(manifest['evaluation'][0]['case_id'])==1
    adapted=SyntheticRuntime(mod,out,config,manifest,'adapted',env,encode)
    mod.evaluate_condition(source,out,'adapted',adapted)
    r=mod.report(source,out,render=False)
    assert r['summary']['overall']['episodes']==32 and r['summary']['overall']['both_success']==32
    assert r['journals']['raw']['interrupted_attempts']==1
    assert r['journals']['raw']['primitive_interactions']==32*4+2
    assert r['journals']['adapted']['primitive_interactions']==32*4


def test_changed_goal_reference_or_episode_cannot_resume(planning_prepared):
    mod,source,out,config,manifest,env,encode=planning_prepared
    raw=SyntheticRuntime(mod,out,config,manifest,'raw',env,encode)
    mod.evaluate_condition(source,out,'raw',raw,limit=1)
    path=next((raw.root/'episodes').glob('*.json'))
    # Select the result rather than its completion marker.
    path=raw.root/'episodes'/f"{manifest['evaluation'][0]['case_id']}.json"
    original=path.read_bytes();path.write_bytes(original+b'changed')
    with pytest.raises(ValueError,match='hash|artifact|episode'):mod.evaluate_condition(source,out,'raw',raw)
    path.write_bytes(original)
    path=out/'reference/goals.pt';path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='hash|artifact|reference'):mod.evaluate_condition(source,out,'raw',raw)
