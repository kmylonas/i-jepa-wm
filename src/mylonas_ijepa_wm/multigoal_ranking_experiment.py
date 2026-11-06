"""Separate manual ranking runs over the established multi-goal dataset."""
import argparse
from pathlib import Path

import numpy as np
import torch

from src.mylonas_ijepa_wm import radial_multigoal_experiment as data_engine
from src.mylonas_ijepa_wm.curriculum_multiband_protocol import read
from src.mylonas_ijepa_wm.curriculum_protocol import digest,write_json
from src.mylonas_ijepa_wm.curriculum_replay import load_adapter,save_adapter
from src.mylonas_ijepa_wm.curriculum_runtime import file_sha256,save_tensors
from src.mylonas_ijepa_wm.radial_experiment import locations
from src.mylonas_ijepa_wm.radial_goal_transfer import load_source,_new_goals
from src.mylonas_ijepa_wm.radial_multigoal import FITTING,pair_costs
from src.mylonas_ijepa_wm.multigoal_ranking import triples,fit,ordinal_rows,MARGIN,GAP


def _hashes(root,names):return {n:file_sha256(root/n) for n in names}


def load_distance_reference(root):
    """Read-only historical loading; old code fingerprints are provenance."""
    root=Path(root).resolve();config=read(root/'config.json')
    if config['kind']!='radial_multigoal_v1' or config['distance_scale']!=34:
        raise ValueError('reference must be a multi-goal distance run')
    locations(root,config['source']['root'])
    source=load_source(config['source']['root'],4)
    hashes=_hashes(root,data_engine._root_files(config))
    design=read(root/'design.json');manifest=read(root/'cache_manifest.json')
    if (config['source']!=source['metadata'] or read(root/'run.json')['files']!=hashes
            or digest(design)!=config['design_digest'] or manifest['config_digest']!=digest(config)):
        raise ValueError('distance reference input hashes/source/config differ')
    adapter,_=data_engine._branch(root,config,design,'multi')
    branch=root/'branches/multi'
    metadata=dict(root=str(root),config_digest=digest(config),root_files=hashes,
        checkpoint_files=_hashes(branch,('complete.json','adapter.pt','training.json')))
    return adapter,metadata,config,design


def _prepared(output,cache=None):
    output=Path(output).resolve();config=read(output/'config.json')
    locations(output,config['distance_reference']['root'])
    dataset,design,source=data_engine._prepared(output/'data',cache)
    _,reference,_,_=load_distance_reference(config['distance_reference']['root'])
    if (config['kind']!='radial_multigoal_ranking_v1' or config['dataset_config_digest']!=digest(dataset)
            or config['distance_reference']!=reference or config['fitting']!=dataset['fitting']
            or config['margin']!=MARGIN or config['min_physical_gap']!=GAP
            or not np.isfinite(config['scale']) or config['scale']<=0):
        raise ValueError('ranking inputs/config/reference differ')
    if read(output/'run.json')['files']!=_hashes(output,('config.json','triplets.pt')):
        raise ValueError('ranking input artifact hash differs')
    return config,design,source


def _finish_prepare(output,cache,baseline,*,device='cpu'):
    dataset,design,_=data_engine._prepared(output/'data',cache)
    _,reference,baseline_config,baseline_design=load_distance_reference(baseline)
    if (baseline_config['source']!=dataset['source'] or baseline_config['new_goals']!=dataset['new_goals']
            or baseline_design['points']!=design['points']
            or baseline_design['train_ids']!=design['train_ids']
            or baseline_design['frame_sha256']!=design['frame_sha256']):
        raise ValueError('distance reference must share geometry, goals and fitting images')
    if (output/'run.json').is_file():
        config,design,_=_prepared(output,cache)
        if config['distance_reference']!=reference:raise ValueError('distance reference changed')
        return dict(config=config,design=design)
    data=torch.load(output/'data/training_data.pt',map_location='cpu',weights_only=True)
    original=data['point_ids'].index(design['goal_ids'][0])
    populations=dict(single=triples(data['positions'],[original]),
        multi=triples(data['positions'],list(range(len(data['positions'])))))
    with torch.no_grad():raw=pair_costs(None,data['tokens'],data['pairs']['multi'],device=device)
    positive=raw[raw>0]
    if not len(positive):raise ValueError('zero training-only raw scale')
    config=dict(kind='radial_multigoal_ranking_v1',dataset_config_digest=digest(dataset),
        distance_reference=reference,fitting=dataset['fitting'],scale=float(positive.median()),
        margin=MARGIN,min_physical_gap=GAP,triplet_counts={b:len(t) for b,t in populations.items()})
    save_tensors(output/'triplets.pt',populations);write_json(output/'config.json',config)
    write_json(output/'run.json',dict(files=_hashes(output,('config.json','triplets.pt'))))
    return dict(config=config,design=design)


def prepare(source,baseline,output,cache,env,encode,execution,*,fitting=None,
            new_goals=data_engine.DEFAULT_GOALS,batch_size=8):
    output,cache=locations(output,cache,(source,baseline))
    _,_,base_config,_=load_distance_reference(baseline)
    if base_config['new_goals']!=_new_goals(new_goals).tolist():raise ValueError('reference goal list differs')
    if output.exists() and any(output.iterdir()) and not (output/'run.json').is_file():
        raise ValueError('incomplete ranking output; choose a new run')
    data_engine.prepare(source,output/'data',cache,env,encode,execution,
        fitting=fitting,new_goals=new_goals,batch_size=batch_size)
    return _finish_prepare(output,cache,baseline,device=execution.get('device_type','cpu'))


def _inputs(output,config,branch):
    if branch not in ('single','multi'):raise ValueError('branch must be single or multi')
    return dict(config_digest=digest(config),branch=branch,triplets_sha256=file_sha256(output/'triplets.pt'))


def _branch(output,config,branch):
    folder=output/'branches'/branch;inputs=_inputs(output,config,branch);complete=read(folder/'complete.json')
    if complete['inputs']!=inputs or complete['files']!=_hashes(folder,('adapter.pt','training.json')):
        raise ValueError('ranking checkpoint input/artifact hash differs')
    adapter,saved=load_adapter(folder/'adapter.pt','cpu',expected_runtime_id=digest(config))
    if saved['provenance'].get('inputs_digest')!=digest(inputs):raise ValueError('ranking checkpoint provenance differs')
    return adapter,read(folder/'training.json')


def train(output,cache,branch,*,device='cpu'):
    output,cache=locations(output,cache);config,_,_=_prepared(output,cache)
    inputs=_inputs(output,config,branch);folder=output/'branches'/branch
    if (folder/'complete.json').is_file():return _branch(output,config,branch)[1]
    if folder.exists() and any(folder.iterdir()):raise ValueError('incomplete ranking branch; choose a new run')
    data=torch.load(output/'data/training_data.pt',map_location='cpu',weights_only=True)
    population=torch.load(output/'triplets.pt',map_location='cpu',weights_only=True)[branch]
    def progress(row):print(f"{branch}: update {row['update']}: fixed rank loss {row['rank_loss']:.6f}, accuracy {row['rank_accuracy']:.3f}",flush=True)
    adapter,t=fit(data['tokens'],population,config['scale'],device=device,progress=progress,**config['fitting'])
    t.update(branch=branch,point_ids=data['point_ids'])
    folder.mkdir(parents=True,exist_ok=True)
    save_adapter(folder/'adapter.pt',adapter,t,dict(runtime_id=digest(config),inputs_digest=digest(inputs)))
    write_json(folder/'training.json',t);_prepared(output)
    write_json(folder/'complete.json',dict(inputs=inputs,files=_hashes(folder,('adapter.pt','training.json'))))
    return t


def _evaluation_inputs(output,config):
    return dict(config_digest=digest(config),checkpoints={b:file_sha256(output/'branches'/b/'adapter.pt') for b in ('single','multi')})


def report(output,*,render=True):
    output=Path(output).resolve();config,design,_=_prepared(output)
    for b in ('single','multi'):_branch(output,config,b)
    complete=read(output/'complete.json')
    if complete['inputs']!=_evaluation_inputs(output,config) or complete['files']!=_hashes(output,('report.json',)):
        raise ValueError('ranking report input/artifact hash differs')
    r=read(output/'report.json')
    if render:
        from src.mylonas_ijepa_wm.multigoal_ranking_plots import plot_run
        names=['coverage','training_loss','goal_summary']+[f'goal_{i}_heatmaps' for i in range(len(design['goal_ids']))]
        if any(not (output/f'{n}.{ext}').is_file() for n in names for ext in ('png','pdf')):plot_run(output,r)
    return r


def evaluate(output,cache,*,device='cpu',render=True):
    output,cache=locations(output,cache)
    if (output/'complete.json').is_file():return report(output,render=render)
    config,design,source=_prepared(output,cache)
    models={b:_branch(output,config,b)[0] for b in ('single','multi')}
    models['distance']=load_distance_reference(config['distance_reference']['root'])[0]
    tokens=torch.load(cache/'tokens.pt',map_location='cpu',weights_only=True)['tokens']
    raw=data_engine._scores(None,tokens,design['goal_ids'],device)
    nx,ny=len(design['grid']['xs']),len(design['grid']['ys']);edges=[]
    for y in range(ny):
        for x in range(nx):
            a=y*nx+x
            if x+1<nx:edges.append((a,a+1))
            if y+1<ny:edges.append((a,a+nx))
    goals=[design['points'][i] for i in design['goal_ids']];methods={}
    for name,adapter in models.items():
        scores=data_engine._scores(adapter,tokens,design['goal_ids'],device)
        rows=ordinal_rows(design['points'],goals,raw,scores,design['heldout_mask'],edges)
        for j,row in enumerate(rows):
            i=design['goal_ids'][j]
            row.update(reference_status='seen' if j==0 else 'withheld',
                raw_self_cost=float(raw[j,i]),adapted_self_cost=float(scores[j,i]))
        methods[name]=rows
    _prepared(output)
    r=dict(kind=config['kind'],source=source['metadata'],distance_reference=config['distance_reference'],
        references_unchanged=True,points=design['points'],grid=design['grid'],grid_count=design['grid_count'],
        goal_ids=design['goal_ids'],heldout_mask=design['heldout_mask'],
        training_positions=[design['points'][i] for i in design['train_ids']],
        methods=methods,training={b:_branch(output,config,b)[1] for b in ('single','multi')},
        scale=config['scale'],min_physical_gap=GAP,
        original_goal_consistency=dict(raw_token_mse=float((tokens[design['goal_ids'][0]]-source['cached_goal']).square().mean())),
        note='Ranking costs have arbitrary units; no distance regression/calibration. Same ordinal cohort for all models. Regression comparator uses a different training recipe.')
    write_json(output/'report.json',r)
    write_json(output/'complete.json',dict(inputs=_evaluation_inputs(output,config),files=_hashes(output,('report.json',))))
    if render:
        from src.mylonas_ijepa_wm.multigoal_ranking_plots import plot_run
        plot_run(output,r)
    return r


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=('prepare','train','evaluate','report'))
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--cache-dir',type=Path)
    p.add_argument('--source-run',type=Path);p.add_argument('--distance-baseline',type=Path)
    p.add_argument('--encoder-checkpoint',type=Path);p.add_argument('--dino-wm-root',type=Path)
    p.add_argument('--branch',choices=('single','multi'));p.add_argument('--device',default='cuda')
    a=p.parse_args(argv)
    if a.operation=='report':
        r=report(a.output_dir);print(f"Verified ordinal report: {len(r['goal_ids'])} goals");return r
    if a.cache_dir is None:p.error('operation requires --cache-dir')
    if a.operation=='train':
        if a.branch is None:p.error('train requires --branch')
        return train(a.output_dir,a.cache_dir,a.branch,device=a.device)
    if a.operation=='evaluate':return evaluate(a.output_dir,a.cache_dir,device=a.device)
    if any(v is None for v in (a.source_run,a.distance_baseline,a.encoder_checkpoint,a.dino_wm_root)):
        p.error('prepare requires source run, distance baseline, encoder and renderer')
    output,cache=locations(a.output_dir,a.cache_dir,(a.source_run,a.distance_baseline,a.encoder_checkpoint,a.dino_wm_root))
    load_distance_reference(a.distance_baseline)
    if output.exists() and any(output.iterdir()) and not (output/'run.json').is_file():
        raise ValueError('incomplete ranking output; choose a new run')
    data_engine.main(['prepare','--output-dir',str(output/'data'),'--cache-dir',str(cache),
        '--source-run',str(a.source_run),'--encoder-checkpoint',str(a.encoder_checkpoint),
        '--dino-wm-root',str(a.dino_wm_root),'--device',a.device])
    return _finish_prepare(output,cache,a.distance_baseline,device=a.device)


if __name__=='__main__':main()
