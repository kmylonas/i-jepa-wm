"""Cached radial observations and signed ranking or distance-regression stages."""

import argparse
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from src.mylonas_ijepa_wm.cost_adapter import token_goal_cost
from src.mylonas_ijepa_wm.curriculum_heatmaps import physical_bounds
from src.mylonas_ijepa_wm.curriculum_multiband_protocol import read
from src.mylonas_ijepa_wm.curriculum_protocol import digest,write_json
from src.mylonas_ijepa_wm.curriculum_replay import save_adapter,load_adapter
from src.mylonas_ijepa_wm.curriculum_runtime import (
    DINO_COMMIT,EXPECTED_ENCODER,PilotRuntime,file_sha256,git_output,save_tensors,source_digest)
from src.mylonas_ijepa_wm.radial_geometry import make_design
from src.mylonas_ijepa_wm.radial_metrics import summarize
from src.mylonas_ijepa_wm.radial_training import fit


FITTING=dict(updates=500,batch_size=16,bottleneck=64,lr=1e-4)


def locations(output,cache,sources=()):
    output,cache=(Path(p).resolve() for p in (output,cache))
    if output==cache or output in cache.parents or cache in output.parents:
        raise ValueError('output and local token cache must be separate nonoverlapping directories')
    for source in map(lambda p:Path(p).resolve(),sources):
        if any(p==source or source in p.parents or p in source.parents for p in (output,cache)):
            raise ValueError('output/cache must not overlap encoder or renderer source')
    # Check directories, files and dangling links without following aliases.
    # This runs before loading inputs or fitting, including completed-stage reuse.
    for root in (output,cache):
        for parent,dirs,files in os.walk(root,followlinks=False):
            if any((Path(parent)/name).is_symlink() for name in dirs+files):
                raise ValueError('radial output/cache must not contain descendant symlinks or aliases')
    return output,cache


def _hashes(root,names):
    return {name:file_sha256(root/name) for name in names}


def validate(output,cache):
    output,cache=locations(output,cache)
    run=read(output/'run.json')
    if run['files']!=_hashes(output,run['files']):raise ValueError('radial input artifact hash differs')
    config,design=read(output/'config.json'),read(output/'design.json')
    index=read(output/'cache_manifest.json')
    if (index!=read(cache/'index.json') or config['kind'] not in ('radial_ranking_v1','radial_distance_v1')
            or (config['kind']=='radial_distance_v1' and config.get('distance_scale')!=34.)
            or config['design_digest']!=digest(design) or design!=make_design(tuple(design['bounds']))
            or index['config_digest']!=digest(config)
            or config['signature'].get('source')!=source_digest(Path(__file__).parents[1])):
        raise ValueError('radial cache/config/signature differs')
    n=0
    for entry in index['shards']:
        if entry['start']!=n or entry['stop']<=n or entry['stop']>len(design['points']):
            raise ValueError('radial shard coverage differs')
        if file_sha256(cache/entry['path'])!=entry['sha256']:raise ValueError('radial token cache shard hash differs')
        n=entry['stop']
    if n!=len(design['points']):raise ValueError('incomplete radial token cache')
    return config,design,index


def prepare(output,cache,env,encode,signature,*,fitting=None,batch_size=8,objective='ranking'):
    output,cache=locations(output,cache)
    if objective not in ('ranking','distance'):raise ValueError('unknown radial objective')
    env.reset(0,layout=[32,30],start=[44,34],goal=[50,42])
    design=make_design(physical_bounds(env.env))
    default=FITTING if objective=='ranking' else dict(FITTING,batch_size=32)
    fitting=dict(default if fitting is None else fitting)
    if set(fitting)!=set(FITTING) or batch_size<1:raise ValueError('invalid radial fitting configuration')
    config=dict(kind='radial_ranking_v1' if objective=='ranking' else 'radial_distance_v1',design_digest=digest(design),
        signature=dict(signature,source=source_digest(Path(__file__).parents[1])),
        fitting=fitting,encoder_batch_size=batch_size,stage_seeds=[0,1,2,3])
    if objective=='distance':config['distance_scale']=34.
    if (output/'run.json').exists():
        old,_,_=validate(output,cache)
        if old!=config:raise ValueError('radial config/signature changed; use a new output')
        return dict(config=config,design=design,raw_costs=read(output/'raw_costs.json'))
    if (output.exists() and any(output.iterdir())) or (cache.exists() and any(cache.iterdir())):
        raise ValueError('incomplete/unrelated output or cache; use new empty directories')
    output.mkdir(parents=True,exist_ok=True);(cache/'shards').mkdir(parents=True,exist_ok=True)
    train_ids=[r['point_id'] for r in design['training']]
    train_set=set(train_ids);train_tokens={};raw=[];frame_hashes=[];entries=[]
    goal=None
    points=design['points']
    for start in range(0,len(points),batch_size):
        stop=min(start+batch_size,len(points))
        frames=np.stack([env._render_at(p) for p in points[start:stop]])
        tokens=encode(frames).detach().cpu().float()
        if (tokens.ndim!=3 or len(tokens)!=len(frames) or not torch.isfinite(tokens).all()
                or (goal is not None and tokens.shape[1:]!=goal.shape)):
            raise ValueError('invalid fresh radial tokens')
        if start==0:
            assert design['goal_id']==0
            goal=tokens[0].clone();save_tensors(output/'goal.pt',dict(tokens=goal))
            Image.fromarray(frames[0]).save(output/'goal.png')
        raw.extend((tokens-goal).square().mean(dim=(1,2)).tolist())
        frame_hashes.extend(hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames)
        for i in range(start,stop):
            if i in train_set:train_tokens[i]=tokens[i-start].clone()
        path=f'shards/{start:06d}.pt'
        save_tensors(cache/path,dict(tokens=tokens,point_ids=list(range(start,stop))))
        entries.append(dict(start=start,stop=stop,path=path,sha256=file_sha256(cache/path)))
        if start==0 or stop==len(points) or start//batch_size%50==0:
            print(f'Encoded {stop}/{len(points)} unique radial/probe images',flush=True)
    data=dict(tokens=torch.stack([train_tokens[i] for i in train_ids]),point_ids=train_ids,
        radii=torch.tensor([r['radius'] for r in design['training']],dtype=torch.float32))
    save_tensors(output/'training_data.pt',data)
    index=dict(config_digest=digest(config),shards=entries,frame_sha256=frame_hashes)
    write_json(cache/'index.json',index);write_json(output/'cache_manifest.json',index)
    write_json(output/'config.json',config);write_json(output/'design.json',design)
    write_json(output/'raw_costs.json',raw)
    names=('goal.pt','goal.png','training_data.pt','cache_manifest.json','config.json','design.json','raw_costs.json')
    write_json(output/'run.json',dict(files=_hashes(output,names)))
    return dict(config=config,design=design,raw_costs=raw)


def completed_stage(output,index,expected_inputs):
    folder=output/f'stages/s{index}'
    commit=read(folder/'complete.json')
    if commit['inputs']!=expected_inputs or commit['files']!=_hashes(folder,commit['files']):
        raise ValueError('radial checkpoint/stage artifact hash or inputs differ')
    adapter,saved=load_adapter(folder/'adapter.pt','cpu',expected_runtime_id=expected_inputs['config_digest'])
    if saved['provenance'].get('inputs_digest')!=digest(expected_inputs):
        raise ValueError('radial checkpoint provenance differs')
    return adapter,read(folder/'report.json')


def stage(output,cache,index,*,device='cpu',render=True):
    output,cache=locations(output,cache)
    if index not in (1,2,3,4):raise ValueError('stage index must be1..4')
    config,design,cache_index=validate(output,cache)
    objective='distance' if config['kind']=='radial_distance_v1' else 'ranking'
    incoming=None;incoming_sha=None
    # Validate the whole completed chain before fitting or admitting results.
    for earlier in range(1,index):
        prior_inputs=dict(config_digest=digest(config),stage=earlier,seed=config['stage_seeds'][earlier-1],
            incoming_sha256=incoming_sha,training_ids=design['stages'][earlier-1]['training_ids'])
        incoming,_=completed_stage(output,earlier,prior_inputs)
        incoming_sha=file_sha256(output/f'stages/s{earlier}/adapter.pt')
    ids=design['stages'][index-1]['training_ids']
    inputs=dict(config_digest=digest(config),stage=index,seed=config['stage_seeds'][index-1],
        incoming_sha256=incoming_sha,training_ids=ids)
    folder=output/f'stages/s{index}'
    if (folder/'complete.json').exists():
        return completed_stage(output,index,inputs)[1]
    goal=torch.load(output/'goal.pt',map_location='cpu',weights_only=True)['tokens']
    data=torch.load(output/'training_data.pt',map_location='cpu',weights_only=True)
    lookup={point_id:i for i,point_id in enumerate(data['point_ids'])}
    take=[lookup[i] for i in ids]
    fitter=fit;loss_key='rank_loss'
    if objective=='distance':
        from src.mylonas_ijepa_wm.radial_distance import fit as distance_fit
        fitter=distance_fit;loss_key='regression_loss'
    adapter,training=fitter(data['tokens'][take],data['radii'][take],goal,initial_adapter=incoming,
        seed=inputs['seed'],device=device,**config['fitting'],progress=lambda row:print(
            f"Stage {index}, R={design['stages'][index-1]['outer_radius']}: update {row['update']} {loss_key}={row[loss_key]:.6f}",flush=True))
    actual={ids[i] for i,n in enumerate(training['sample_counts']) if n}
    # Include actual input exposure from the earlier cumulative fits as well.
    for earlier in range(1,index):
        actual.update(read(output/f'stages/s{earlier}/report.json')['actual_fitting_ids'])
    actual.add(design['goal_id'])
    scores=[];target=goal.to(device)
    with torch.no_grad():
        for entry in cache_index['shards']:
            shard=torch.load(cache/entry['path'],map_location='cpu',weights_only=True)
            if shard['point_ids']!=list(range(entry['start'],entry['stop'])):
                raise ValueError('cached radial point IDs differ')
            scores.extend(token_goal_cost(adapter,shard['tokens'].to(device),target).cpu().tolist())
    if len(scores)!=len(design['points']) or not np.isfinite(scores).all():
        raise ValueError('invalid radial adapted scores')
    folder.mkdir(parents=True,exist_ok=True)
    save_adapter(folder/'adapter.pt',adapter,training,dict(runtime_id=digest(config),inputs_digest=digest(inputs)))
    raw=read(output/'raw_costs.json')
    measured=summarize(design,raw,scores,cache_index['frame_sha256'],ids,actual,
        design['stages'][index-1]['outer_radius'])
    report=dict(stage=index,objective=objective,outer_radius=design['stages'][index-1]['outer_radius'],
        training_positions=len(ids),offered_training_ids=ids,actual_fitting_ids=sorted(actual),
        incoming_sha256=incoming_sha,checkpoint_sha256=file_sha256(folder/'adapter.pt'),
        initialization=training['initialization'],training=training,adapted_costs=scores,
        raw_goal_self_cost=raw[design['goal_id']],adapted_goal_self_cost=scores[design['goal_id']],
        primitive_interactions=0,**measured)
    if objective=='distance':
        from src.mylonas_ijepa_wm.radial_distance import distance_summary
        report['distance']=distance_summary(design,raw,scores,report)
    write_json(folder/'training.json',training);write_json(folder/'report.json',report)
    if render:
        from src.mylonas_ijepa_wm.radial_plots import plot_stage
        plot_stage(output,design,raw,report)
        if objective=='distance':
            from src.mylonas_ijepa_wm.radial_distance_plots import plot_distance
            plot_distance(output,design,raw,report)
    write_json(folder/'complete.json',dict(inputs=inputs,
        files=_hashes(folder,('adapter.pt','training.json','report.json'))))
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('prepare','stage'))
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--cache-dir',type=Path,required=True)
    parser.add_argument('--encoder-checkpoint',type=Path)
    parser.add_argument('--dino-wm-root',type=Path)
    parser.add_argument('--stage-index',type=int)
    parser.add_argument('--objective',choices=('ranking','distance'),help='prepare only; stages use their signed configuration')
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args(argv)
    if args.operation=='stage':
        if args.objective is not None:parser.error('--objective applies only to prepare')
        result=stage(args.output_dir,args.cache_dir,args.stage_index,device=args.device)
        print(f"Completed stage {result['stage']}: {result['training_positions']} training positions",flush=True)
        return result
    if args.encoder_checkpoint is None or args.dino_wm_root is None:
        parser.error('prepare needs --encoder-checkpoint and --dino-wm-root')
    locations(args.output_dir,args.cache_dir,(args.encoder_checkpoint,args.dino_wm_root))
    if file_sha256(args.encoder_checkpoint)!=EXPECTED_ENCODER:raise ValueError('unexpected encoder checkpoint')
    if (git_output(args.dino_wm_root,'rev-parse','HEAD')!=DINO_COMMIT
            or git_output(args.dino_wm_root,'status','--porcelain','--untracked-files=no')):
        raise ValueError('renderer must match unmodified pinned DINO source')
    from src.mylonas_ijepa_wm import plan_pointmaze as visual
    from src.mylonas_ijepa_wm.environment_adapters import WallAdapter
    from src.mylonas_ijepa_wm.environment_configs import EnvironmentConfig
    device=torch.device(args.device)
    encoder=visual.load_frozen_ijepa_encoder(args.encoder_checkpoint,device)
    env=WallAdapter(EnvironmentConfig(env='wall',wall_layouts=[[32,30]],dino_wm_root=str(args.dino_wm_root)),100)
    runtime=SimpleNamespace(device=device,precision='fp32',args=SimpleNamespace(encoder_batch_size=8),load_encoder=lambda:encoder)
    try:
        result=prepare(args.output_dir,args.cache_dir,env,lambda f:PilotRuntime.encode_frames(runtime,f),
            dict(encoder_sha256=EXPECTED_ENCODER,renderer_commit=DINO_COMMIT,torch_version=str(torch.__version__),
                device_type=device.type,precision='fp32',encoder_batch_size=8,
                matmul_precision=torch.get_float32_matmul_precision()),objective=args.objective or 'ranking')
    finally:env.close()
    print('Prepared radial data; all four stages reuse the local token cache.',flush=True)
    return result


if __name__=='__main__':main()
