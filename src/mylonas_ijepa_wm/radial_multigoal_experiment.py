"""Matched single/multi-goal fitting and reserved-reference transfer evaluation."""
import argparse
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from src.mylonas_ijepa_wm.curriculum_multiband_protocol import read
from src.mylonas_ijepa_wm.curriculum_protocol import digest,write_json
from src.mylonas_ijepa_wm.curriculum_replay import save_adapter,load_adapter
from src.mylonas_ijepa_wm.curriculum_runtime import (
    EXPECTED_ENCODER,DINO_COMMIT,PilotRuntime,file_sha256,git_output,save_tensors,source_digest)
from src.mylonas_ijepa_wm.radial_experiment import locations
from src.mylonas_ijepa_wm.radial_goal_transfer import load_source,measure,_new_goals,_signature
from src.mylonas_ijepa_wm.radial_multigoal import SCALE,FITTING,make_split,fit,pair_costs


DEFAULT_GOALS=[
    [round(50+math.sqrt(32)*math.cos(math.radians(a)),6),
     round(42+math.sqrt(32)*math.sin(math.radians(a)),6)] for a in range(0,360,45)
]+[[40.,18.],[50.,18.],[56.,18.],[40.,10.],[50.,10.],[56.,10.],[50.,8.]]


def _hashes(root,names):return {n:file_sha256(root/n) for n in names}


def _root_files(config):
    return ('config.json','design.json','training_data.pt','goals.pt','cache_manifest.json',
            *(f'goal_{i}.png' for i in range(len(config['new_goals'])+1)))


def _prepared(output,cache=None):
    output=Path(output).resolve();config=read(output/'config.json')
    output,source_root=locations(output,config['source']['root'])
    source=load_source(source_root,4)
    if (config['kind']!='radial_multigoal_v1' or config['distance_scale']!=SCALE
            or config['source']!=source['metadata']
            or config['diagnostic_source']!=source_digest(Path(__file__).parents[1])):
        raise ValueError('multi-goal source/config/signature differs')
    if read(output/'run.json')['files']!=_hashes(output,_root_files(config)):
        raise ValueError('multi-goal input artifact hash differs')
    design=read(output/'design.json');manifest=read(output/'cache_manifest.json')
    if config['design_digest']!=digest(design) or manifest['config_digest']!=digest(config):
        raise ValueError('multi-goal design/config signature differs')
    if cache is not None:
        _,cache=locations(output,cache,(source_root,))
        if (read(cache/'index.json')!=manifest or file_sha256(cache/'tokens.pt')!=manifest['tokens_sha256']):
            raise ValueError('multi-goal token cache hash differs')
    return config,design,source


def prepare(source_root,output,cache,env,encode,execution,*,new_goals=DEFAULT_GOALS,
            fitting=None,batch_size=8):
    output,cache=locations(output,cache,(source_root,));source=load_source(source_root,4)
    _signature(source,execution)
    new_goals=_new_goals(new_goals).tolist();old=np.asarray(source['design']['goal'])
    candidates=np.asarray(new_goals)
    if (np.linalg.norm(candidates-old,axis=1).max()>SCALE+1e-5
            or (np.linalg.norm(candidates-old,axis=1)<1e-5).any()
            or (candidates[:,0]<=35).any() or (candidates<7.4).any() or (candidates>56.6).any()):
        raise ValueError('reserved goals must differ and stay inside R34 and reset-safe bounds')
    fitting=dict(FITTING if fitting is None else fitting)
    if set(fitting)!=set(FITTING) or batch_size<1:raise ValueError('invalid fitting/encoding configuration')
    settings=dict(kind='radial_multigoal_v1',source=source['metadata'],new_goals=new_goals,
        fitting=fitting,execution=dict(execution),distance_scale=SCALE,encoder_batch_size=batch_size,
        diagnostic_source=source_digest(Path(__file__).parents[1]))
    if (output/'run.json').is_file():
        config,design,_=_prepared(output,cache)
        if {k:v for k,v in config.items() if k!='design_digest'}!=settings:
            raise ValueError('multi-goal inputs/config changed; choose a new run')
        return dict(config=config,design=design)
    if any(p.exists() and any(p.iterdir()) for p in (output,cache)):
        raise ValueError('incomplete/unrelated output/cache; choose new empty directories')
    original=source['design'];grid=original['grid']
    points=[original['points'][i] for i in grid['point_ids']]
    lookup={tuple(p):i for i,p in enumerate(points)}
    def intern(point):
        key=tuple(map(float,point))
        if key not in lookup:lookup[key]=len(points);points.append(list(key))
        return lookup[key]
    candidate_ids=[intern(old)]+[intern(original['points'][i]) for i in original['stages'][3]['training_ids']]
    goal_ids=[intern(old)]+[intern(p) for p in new_goals]
    env.reset(0,layout=[32,30],start=[44,34],goal=old.tolist())
    fixed=env._render_at(points[0])
    for goal in new_goals:
        env.reset(0,layout=[32,30],start=[44,34],goal=goal)
        if not np.array_equal(fixed,env._render_at(points[0])):
            raise ValueError('goal change altered fixed-position observation')
    hashes=[];parts=[]
    with torch.inference_mode():
        for start in range(0,len(points),batch_size):
            frames=np.stack([env._render_at(p) for p in points[start:start+batch_size]])
            tokens=encode(frames).detach().cpu().float()
            if (tokens.ndim!=3 or len(tokens)!=len(frames) or tokens.shape[1:]!=source['cached_goal'].shape
                    or not torch.isfinite(tokens).all()):raise ValueError('invalid multi-goal encoded tokens')
            parts.append(tokens)
            hashes.extend(hashlib.sha256(f.tobytes()).hexdigest() for f in frames)
            if start==0 or start+batch_size>=len(points):print(f'Encoded {min(start+batch_size,len(points))}/{len(points)} positions',flush=True)
    tokens=torch.cat(parts)
    split=make_split(points,candidate_ids,goal_ids,hashes)
    design=dict(points=points,grid=dict(xs=grid['xs'],ys=grid['ys'],point_ids=list(range(len(grid['point_ids'])))),
        grid_count=len(grid['point_ids']),goal_ids=goal_ids,candidate_ids=candidate_ids,
        frame_sha256=hashes,**split)
    config=dict(settings,design_digest=digest(design));ids=split['train_ids'];remap={i:j for j,i in enumerate(ids)}
    data=dict(tokens=tokens[ids],point_ids=ids,positions=[points[i] for i in ids],
        pairs={b:[[remap[a],remap[c]] for a,c in split[f'{b}_pairs']] for b in ('single','multi')})
    output.mkdir(parents=True,exist_ok=True);cache.mkdir(parents=True,exist_ok=True)
    save_tensors(cache/'tokens.pt',dict(tokens=tokens))
    manifest=dict(config_digest=digest(config),tokens_sha256=file_sha256(cache/'tokens.pt'),shape=list(tokens.shape))
    save_tensors(output/'training_data.pt',data)
    save_tensors(output/'goals.pt',dict(tokens=tokens[goal_ids],positions=[points[i] for i in goal_ids]))
    for j,i in enumerate(goal_ids):Image.fromarray(env._render_at(points[i])).save(output/f'goal_{j}.png')
    write_json(output/'config.json',config);write_json(output/'design.json',design)
    write_json(output/'cache_manifest.json',manifest);write_json(cache/'index.json',manifest)
    if load_source(source_root,4)['metadata']!=source['metadata']:raise ValueError('source changed during preparation')
    write_json(output/'run.json',dict(files=_hashes(output,_root_files(config))))
    return dict(config=config,design=design)


def _branch_inputs(config,design,branch):
    if branch not in ('single','multi'):raise ValueError('branch must be single or multi')
    return dict(config_digest=digest(config),branch=branch,pairs_digest=digest(design[f'{branch}_pairs']))


def _branch(output,config,design,branch):
    folder=output/'branches'/branch;expected=_branch_inputs(config,design,branch)
    complete=read(folder/'complete.json')
    if complete['inputs']!=expected or complete['files']!=_hashes(folder,('adapter.pt','training.json')):
        raise ValueError('multi-goal checkpoint inputs/artifact hash differs')
    adapter,saved=load_adapter(folder/'adapter.pt','cpu',expected_runtime_id=digest(config))
    if saved['provenance'].get('inputs_digest')!=digest(expected):raise ValueError('checkpoint provenance differs')
    return adapter,read(folder/'training.json')


def train(output,cache,branch,*,device='cpu'):
    output,cache=locations(output,cache);config,design,_=_prepared(output,cache)
    inputs=_branch_inputs(config,design,branch);folder=output/'branches'/branch
    if (folder/'complete.json').is_file():return _branch(output,config,design,branch)[1]
    if folder.exists() and any(folder.iterdir()):raise ValueError('incomplete branch; choose a new run')
    data=torch.load(output/'training_data.pt',map_location='cpu',weights_only=True)
    def progress(row):print(f"{branch}: update {row['update']}: fixed train probe loss {row['regression_loss']:.6f}",flush=True)
    adapter,training=fit(data['tokens'],data['positions'],data['pairs'][branch],
        device=device,progress=progress,**config['fitting'])
    training.update(branch=branch,point_ids=data['point_ids'],pair_count=len(data['pairs'][branch]))
    folder.mkdir(parents=True,exist_ok=True)
    save_adapter(folder/'adapter.pt',adapter,training,dict(runtime_id=digest(config),inputs_digest=digest(inputs)))
    write_json(folder/'training.json',training)
    _prepared(output)
    write_json(folder/'complete.json',dict(inputs=inputs,files=_hashes(folder,('adapter.pt','training.json'))))
    return training


@torch.no_grad()
def _scores(adapter,tokens,goal_ids,device):
    refs=tokens[goal_ids].to(device)
    if adapter is not None:adapter=adapter.to(device).eval().requires_grad_(False);refs=adapter(refs)
    parts=[]
    for chunk in tokens.split(8):
        values=chunk.to(device)
        if adapter is not None:values=adapter(values)
        parts.append(torch.stack([(values-g).square().mean(dim=(1,2)).cpu() for g in refs],dim=0))
    if adapter is not None:adapter.cpu()
    return torch.cat(parts,dim=1).numpy()


def _evaluation_inputs(output,config):
    return dict(config_digest=digest(config),checkpoints={b:file_sha256(output/'branches'/b/'adapter.pt') for b in ('single','multi')})


def report(output,*,render=True):
    output=Path(output).resolve();config,design,_=_prepared(output)
    for branch in ('single','multi'):_branch(output,config,design,branch)
    complete=read(output/'complete.json')
    if (complete['inputs']!=_evaluation_inputs(output,config)
            or complete['files']!=_hashes(output,('report.json',))):
        raise ValueError('multi-goal report inputs/artifact hash differs')
    result=read(output/'report.json')
    if render:
        from src.mylonas_ijepa_wm.radial_multigoal_plots import plot_run
        expected=['coverage','training_loss','goal_summary']+[f'goal_{j}_heatmaps' for j in range(len(design['goal_ids']))]
        if any(not (output/f'{name}.{ext}').is_file() for name in expected for ext in ('png','pdf')):
            plot_run(output,result)
    return result


def evaluate(output,cache,*,device='cpu',render=True):
    output,cache=locations(output,cache)
    if (output/'complete.json').is_file():return report(output,render=render)
    config,design,source=_prepared(output,cache)
    models={b:_branch(output,config,design,b)[0] for b in ('single','multi')}
    training={b:_branch(output,config,design,b)[1] for b in ('single','multi')}
    models['historical']=source['adapter']
    tokens=torch.load(cache/'tokens.pt',map_location='cpu',weights_only=True)['tokens']
    if list(tokens.shape)!=read(cache/'index.json')['shape']:raise ValueError('token cache shape differs')
    raw=_scores(None,tokens,design['goal_ids'],device)
    data=torch.load(output/'training_data.pt',map_location='cpu',weights_only=True)
    with torch.no_grad():train_raw=pair_costs(None,data['tokens'],data['pairs']['multi'],device=device).numpy()
    positions=np.asarray(data['positions']);pairs=np.asarray(data['pairs']['multi'])
    distances=np.linalg.norm(positions[pairs[:,0]]-positions[pairs[:,1]],axis=1)
    denominator=float(train_raw@train_raw)
    calibration=max(0.,float(train_raw@distances)/denominator) if denominator>0 else 0.
    nx,ny=len(design['grid']['xs']),len(design['grid']['ys']);edges=[]
    for y in range(ny):
        for x in range(nx):
            a=y*nx+x
            if x+1<nx:edges.append((a,a+1))
            if y+1<ny:edges.append((a,a+nx))
    goals=[design['points'][i] for i in design['goal_ids']];methods={}
    for name,adapter in models.items():
        costs=_scores(adapter,tokens,design['goal_ids'],device)
        rows=measure(design['points'],goals,raw,costs,SCALE,calibration,design['heldout_mask'],edges)
        for j,row in enumerate(rows):
            i=design['goal_ids'][j]
            row.update(reference_status='seen' if j==0 else 'withheld',raw_self_cost=float(raw[j,i]),adapted_self_cost=float(costs[j,i]))
        methods[name]=rows
    unchanged=load_source(source['root'],4)['metadata']==source['metadata']
    if not unchanged:raise ValueError('source changed during evaluation')
    result=dict(kind=config['kind'],source=config['source'],source_unchanged=unchanged,
        distance_scale=SCALE,raw_calibration=calibration,raw_calibration_population='all multi-goal training pairs only',
        historical_raw_calibration=source['report']['distance']['raw_calibration'],
        points=design['points'],grid=design['grid'],grid_count=design['grid_count'],
        goal_ids=design['goal_ids'],training_positions=[design['points'][i] for i in design['train_ids']],
        removed_candidate_ids=design['removed_candidate_ids'],heldout_mask=design['heldout_mask'],
        methods=methods,training=training,
        original_goal_consistency=dict(raw_token_mse=float((tokens[design['goal_ids'][0]]-source['cached_goal']).square().mean())),
        note='Matched fresh models use all R34 nodes at once; historical curriculum is a descriptive comparator. No test calibration or model selection.')
    write_json(output/'report.json',result)
    write_json(output/'complete.json',dict(inputs=_evaluation_inputs(output,config),files=_hashes(output,('report.json',))))
    if render:
        from src.mylonas_ijepa_wm.radial_multigoal_plots import plot_run
        plot_run(output,result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('prepare','train','evaluate','report'))
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--cache-dir',type=Path)
    parser.add_argument('--source-run',type=Path);parser.add_argument('--branch',choices=('single','multi'))
    parser.add_argument('--encoder-checkpoint',type=Path);parser.add_argument('--dino-wm-root',type=Path)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args(argv)
    if args.operation=='report':
        result=report(args.output_dir)
        print(f"Verified multi-goal report: {len(result['goal_ids'])} goals, same heldout cohort.");return result
    if args.cache_dir is None:parser.error('operation requires --cache-dir')
    if args.operation=='train':
        if args.branch is None:parser.error('train requires --branch')
        return train(args.output_dir,args.cache_dir,args.branch,device=args.device)
    if args.operation=='evaluate':return evaluate(args.output_dir,args.cache_dir,device=args.device)
    if args.source_run is None or args.encoder_checkpoint is None or args.dino_wm_root is None:
        parser.error('prepare requires source run, encoder and renderer')
    locations(args.output_dir,args.cache_dir,(args.source_run,args.encoder_checkpoint,args.dino_wm_root))
    if file_sha256(args.encoder_checkpoint)!=EXPECTED_ENCODER:raise ValueError('unexpected encoder weights')
    if (git_output(args.dino_wm_root,'rev-parse','HEAD')!=DINO_COMMIT
            or git_output(args.dino_wm_root,'status','--porcelain','--untracked-files=no')):
        raise ValueError('renderer must match pinned source')
    from src.mylonas_ijepa_wm import plan_pointmaze as visual
    from src.mylonas_ijepa_wm.environment_adapters import WallAdapter
    from src.mylonas_ijepa_wm.environment_configs import EnvironmentConfig
    device=torch.device(args.device);encoder=visual.load_frozen_ijepa_encoder(args.encoder_checkpoint,device)
    runtime=SimpleNamespace(device=device,precision='fp32',args=SimpleNamespace(encoder_batch_size=8),load_encoder=lambda:encoder)
    env=WallAdapter(EnvironmentConfig(env='wall',wall_layouts=[[32,30]],dino_wm_root=str(args.dino_wm_root)),100)
    execution=dict(encoder_sha256=EXPECTED_ENCODER,renderer_commit=DINO_COMMIT,precision='fp32',
        device_type=device.type,torch_version=str(torch.__version__),matmul_precision=torch.get_float32_matmul_precision())
    try:return prepare(args.source_run,args.output_dir,args.cache_dir,env,lambda f:PilotRuntime.encode_frames(runtime,f),execution)
    finally:env.close()


if __name__=='__main__':main()
