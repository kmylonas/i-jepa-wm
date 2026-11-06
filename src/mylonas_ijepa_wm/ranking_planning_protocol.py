"""Frozen ranking provenance, deterministic in-region cases and paired outcomes."""
from pathlib import Path

import numpy as np

from src.mylonas_ijepa_wm import radial_multigoal_experiment as data_engine
from src.mylonas_ijepa_wm import multigoal_ranking_experiment as ranking
from src.mylonas_ijepa_wm.curriculum_multiband_protocol import read
from src.mylonas_ijepa_wm.curriculum_protocol import PLANNING,digest
from src.mylonas_ijepa_wm.radial_experiment import locations
from src.mylonas_ijepa_wm.radial_goal_transfer import load_source


GOALS=[[50.,42.],[46.,38.],[40.,18.],[50.,18.],[56.,10.],[50.,8.]]
SEED=20261106


def load_ranking_source(root):
    root=Path(root).resolve();config=read(root/'config.json');data=root/'data'
    dataset=read(data/'config.json');design=read(data/'design.json')
    locations(root,dataset['source']['root'])
    source=load_source(dataset['source']['root'],4)
    _,reference,_,_=ranking.load_distance_reference(config['distance_reference']['root'])
    data_hashes=ranking._hashes(data,data_engine._root_files(dataset))
    root_hashes=ranking._hashes(root,('config.json','triplets.pt'))
    if (config['kind']!='radial_multigoal_ranking_v1' or dataset['kind']!='radial_multigoal_v1'
            or config['dataset_config_digest']!=digest(dataset) or dataset['source']!=source['metadata']
            or config['distance_reference']!=reference or config['fitting']!=dataset['fitting']
            or config['margin']!=.2 or config['min_physical_gap']!=.5
            or not np.isfinite(config['scale']) or config['scale']<=0
            or digest(design)!=dataset['design_digest']
            or read(data/'cache_manifest.json')['config_digest']!=digest(dataset)
            or read(data/'run.json')['files']!=data_hashes or read(root/'run.json')['files']!=root_hashes):
        raise ValueError('frozen ranking input/source/config hashes differ')
    adapter,_=ranking._branch(root,config,'multi')
    metadata=dict(root=str(root),config_digest=digest(config),dataset_config_digest=digest(dataset),
        root_files=root_hashes,data_files=data_hashes,original_source=source['metadata'],distance_reference=reference,
        checkpoint_files=ranking._hashes(root/'branches/multi',('complete.json','adapter.pt','training.json')),
        fitting_signature=dataset['execution'])
    return adapter.eval().requires_grad_(False),metadata,design


def in_region(points):
    points=np.asarray(points,dtype=float)
    return ((points[...,0]>35)&(points>=4.001).all(axis=-1)&(points<=59.999).all(axis=-1)
        &(np.linalg.norm(points-[50,42],axis=-1)<=34+1e-5))


def make_manifest(design,starts_per_goal=8,goals=None):
    goals=np.asarray(GOALS if goals is None else goals,dtype=float)
    if (goals.ndim!=2 or goals.shape[1]!=2 or not len(goals) or not np.isfinite(goals).all()
            or len(np.unique(goals,axis=0))!=len(goals) or not np.allclose(goals[0],[50,42])
            or not in_region(goals).all() or (goals[:,0]<38).any() or (goals[:,0]>56).any()
            or (goals[:,1]<8).any() or (goals[:,1]>55).any()
            or not isinstance(starts_per_goal,int) or starts_per_goal<1):
        raise ValueError('valid unique in-region goals and positive start count required')
    points=np.asarray(design['points']);references=points[design['goal_ids']]
    goal_indices=[]
    for goal in goals:
        distances=np.linalg.norm(references-goal,axis=1)
        if distances.min()>1e-5:raise ValueError('goal must be original control or a source-withheld reference')
        goal_indices.append(int(distances.argmin()))
    offered=points[design['train_ids']+design['goal_ids']];rng=np.random.default_rng(SEED)
    used=[];cases=[];schedule=('long','near','medium','long','near','medium','long','medium')
    bands=dict(near=[6,10],medium=[10,18],long=[18,30])
    for gi,goal in enumerate(goals):
        for index in range(starts_per_goal):
            band=schedule[index%len(schedule)];lo,hi=bands[band]
            for _ in range(50000):
                radius=rng.uniform(lo,hi);angle=rng.uniform(0,2*np.pi)
                p=goal+radius*np.array([np.cos(angle),np.sin(angle)])
                if not (38<=p[0]<=56 and 8<=p[1]<=55 and in_region(p)):continue
                if np.linalg.norm(offered-p,axis=1).min()<.5:continue
                if used and np.linalg.norm(np.asarray(used)-p,axis=1).min()<.5:continue
                break
            else:raise ValueError('cannot sample distinct in-region starts')
            used.append(p.tolist())
            cases.append(dict(case_id=f'g{gi:02d}_s{index:02d}',goal_id=gi,source_goal_index=goal_indices[gi],
                split='eval',band=band,goal_status='seen' if gi==0 else 'withheld',
                start=p.tolist(),goal=goal.tolist(),layout=[32,30],seed=int(rng.integers(0,2**31-1)),run_seed=None))
    return dict(kind='ranking_planning_v1',seed=SEED,starts_per_goal=starts_per_goal,
        goals=goals.tolist(),evaluation=cases,training={},fitting={},planning=dict(PLANNING),
        start_distance_bands=bands,success_threshold=4.5,
        region=dict(center=[50,42],radius=34,right_of_wall=35,start_x=[38,56],start_y=[8,55]))


def summarize(raw,adapted):
    original={r['case_id']:r for r in raw};learned={r['case_id']:r for r in adapted}
    if (not original or len(original)!=len(raw) or len(learned)!=len(adapted)
            or original.keys()!=learned.keys()
            or any(original[k]['case']!=learned[k]['case'] for k in original)):
        raise ValueError('paired summaries require identical unique cases and seeds')
    def group(keys):
        n=len(keys);result=dict(episodes=n,conditions={})
        for label,rows in (('raw',original),('adapted',learned)):
            items=[rows[k] for k in keys]
            def mean(field):return float(np.mean([r[field] for r in items])) if items else None
            successes=sum(bool(r['success']) for r in items)
            result['conditions'][label]=dict(successes=successes,success_rate=successes/n if n else None,
                mean_final_distance=mean('final_goal_distance'),mean_steps=mean('micro_steps'),mean_path_length=mean('path_length'),
                excursion_episodes=sum(bool(r['ever_outside_fitting']) for r in items),
                in_region_successes=sum(bool(r['success']) and not r['ever_outside_fitting'] for r in items),
                failed_case_ids=[r['case_id'] for r in items if not r['success']])
        won=[k for k in keys if learned[k]['success'] and not original[k]['success']]
        lost=[k for k in keys if original[k]['success'] and not learned[k]['success']]
        both=[k for k in keys if original[k]['success'] and learned[k]['success']]
        result.update(wins=len(won),losses=len(lost),won_case_ids=won,lost_case_ids=lost,
            both_success=len(both),both_fail=sum(not original[k]['success'] and not learned[k]['success'] for k in keys),
            both_success_efficiency=dict(episodes=len(both)))
        efficiency=result['both_success_efficiency']
        for field in ('micro_steps','path_length'):
            name='steps' if field=='micro_steps' else field
            for label,rows in (('raw',original),('adapted',learned)):
                efficiency[f'{label}_mean_{name}']=float(np.mean([rows[k][field] for k in both])) if both else None
            efficiency[f'mean_{name}_delta']=float(np.mean([learned[k][field]-original[k][field] for k in both])) if both else None
        return result
    keys=list(original);by_goal={};by_band={}
    for gi in sorted({r['case']['goal_id'] for r in raw}):
        selected=[k for k in keys if original[k]['case']['goal_id']==gi]
        by_goal[str(gi)]=dict(goal=original[selected[0]]['case']['goal'],status=original[selected[0]]['case']['goal_status'],**group(selected))
    for band in ('near','medium','long'):by_band[band]=group([k for k in keys if original[k]['case']['band']==band])
    return dict(overall=group(keys),withheld=group([k for k in keys if original[k]['case']['goal_status']=='withheld']),
        by_goal=by_goal,by_distance_band=by_band,
        note='Paired fixed cases; inspect per-goal results, both-success efficiency and excursions. Not independent replications or a significance test.')
