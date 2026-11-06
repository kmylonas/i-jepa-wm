"""Read-only goal transfer of a saved Euclidean-distance cost adapter."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from src.mylonas_ijepa_wm.cost_adapter import token_goal_cost
from src.mylonas_ijepa_wm.curriculum_protocol import digest,write_json
from src.mylonas_ijepa_wm.curriculum_multiband_protocol import read
from src.mylonas_ijepa_wm.curriculum_runtime import (
    file_sha256,save_tensors,source_digest,EXPECTED_ENCODER,DINO_COMMIT,PilotRuntime,git_output)
from src.mylonas_ijepa_wm.radial_experiment import locations,completed_stage
from src.mylonas_ijepa_wm.radial_geometry import make_design


SCALE=34.
def _files(count):return ('report.json','goals.pt',*(f'goal_{j}.png' for j in range(count)))


def _new_goals(values):
    try:goals=np.asarray(values,dtype=float)
    except (ValueError,TypeError) as exc:raise ValueError('finite goal coordinate pairs required') from exc
    if goals.shape==(2,):goals=goals.reshape(1,2)
    if goals.ndim!=2 or goals.shape[1]!=2 or not len(goals) or not np.isfinite(goals).all():
        raise ValueError('nonempty finite goal coordinate pairs required')
    if len(np.unique(goals,axis=0))!=len(goals):raise ValueError('duplicate goals are not allowed')
    return goals


def load_source(root,stage):
    root=Path(root).resolve()
    if stage not in (1,2,3,4):raise ValueError('source stage must be1..4')
    config=read(root/'config.json');design=read(root/'design.json');run=read(root/'run.json')
    if config['kind']!='radial_distance_v1' or config.get('distance_scale')!=SCALE:
        raise ValueError('source must be the Euclidean-distance adapter, scale34')
    hashes={name:file_sha256(root/name) for name in run['files']}
    if hashes!=run['files'] or digest(design)!=config['design_digest'] or design!=make_design(tuple(design['bounds'])):
        raise ValueError('source input hashes/design differ')
    cache=read(root/'cache_manifest.json')
    if cache['config_digest']!=digest(config):raise ValueError('source config signature differs')
    incoming=None;chain=[]
    for i in range(1,stage+1):
        inputs=dict(config_digest=digest(config),stage=i,seed=config['stage_seeds'][i-1],
            incoming_sha256=incoming,training_ids=design['stages'][i-1]['training_ids'])
        adapter,report=completed_stage(root,i,inputs)
        folder=root/f'stages/s{i}';incoming=file_sha256(folder/'adapter.pt')
        chain.append({name:file_sha256(folder/name) for name in ('complete.json','adapter.pt','training.json','report.json')})
        if report['objective']!='distance' or report['checkpoint_sha256']!=incoming:
            raise ValueError('source checkpoint kind/lineage differs')
    cached_goal=torch.load(root/'goal.pt',map_location='cpu',weights_only=True)['tokens']
    return dict(root=root,adapter=adapter,config=config,design=design,cache=cache,report=report,
        cached_goal=cached_goal,raw=read(root/'raw_costs.json'),
        metadata=dict(root=str(root),stage=stage,radius=design['stages'][stage-1]['outer_radius'],
            config_digest=digest(config),root_files=hashes,chain=chain,checkpoint_sha256=incoming,
            fitting_signature=config['signature']))


def measure(points,goals,raw,adapted,radius,calibration,heldout,edges):
    points=np.asarray(points);goals=np.asarray(goals);raw=np.asarray(raw);adapted=np.asarray(adapted)
    truth=np.linalg.norm(points[None]-goals[:,None],axis=2)
    common=(np.linalg.norm(points-goals[0],axis=1)<=radius+1e-5)&(truth.max(axis=0)<=radius+1e-5)
    heldout=np.asarray(heldout,dtype=bool);rows=[]
    for j,goal in enumerate(goals):
        methods=dict(raw_calibrated=raw[j]*calibration,adapted=adapted[j]*SCALE)
        def metrics(mask):
            row=dict(count=int(mask.sum()))
            for name,prediction in methods.items():
                error=prediction[mask]-truth[j,mask]
                row[name]=dict(mae=float(np.abs(error).mean()) if mask.any() else None,
                    rmse=float(np.sqrt(np.mean(error**2))) if mask.any() else None)
            return row
        directed=[]
        for a,b in edges:
            if common[a] and common[b] and abs(truth[j,a]-truth[j,b])>1e-5:
                directed.append((a,b) if truth[j,a]>truth[j,b] else (b,a))
        def directions(pairs):
            result=dict(comparisons=len(pairs))
            for name,values in (('raw',raw[j]),('adapted',adapted[j])):
                tolerance=max(1e-7,.001*float(np.ptp(values[common]))) if common.any() else 1e-7
                differences=np.array([values[b]-values[a] for a,b in pairs])
                result[name+'_up_steps']=int((differences>tolerance).sum())
                result[name+'_ties']=int((np.abs(differences)<=tolerance).sum())
                result[name+'_up_fraction']=float((differences>tolerance).mean()) if len(pairs) else None
            return result
        rows.append(dict(goal=goal.tolist(),true_distances=truth[j].tolist(),raw_costs=raw[j].tolist(),
            adapted_costs=adapted[j].tolist(),common_mask=common.tolist(),
            full_domain=metrics(np.ones(len(points),dtype=bool)),common=metrics(common),
            heldout_common=metrics(common&heldout),local=metrics(common&(truth[j]<=6+1e-5)),
            directional=directions(directed),heldout_directional=directions([(a,b) for a,b in directed if heldout[a] and heldout[b]])))
    return rows


def _signature(source,execution):
    for key in ('encoder_sha256','renderer_commit','precision'):
        if source['config']['signature'].get(key)!=execution.get(key) or execution.get(key) is None:
            raise ValueError(f'encoder/renderer/precision signature mismatch: {key}')


def _inputs(source,new_goal,execution):
    return dict(source=source['metadata'],new_goals=_new_goals(new_goal).tolist(),execution=execution,
        diagnostic_source=source_digest(Path(__file__).parents[1]))


def _load(output,inputs,render):
    complete=read(output/'complete.json')
    count=len(inputs['new_goals'])+1
    if complete['inputs']!=inputs or complete['files']!={n:file_sha256(output/n) for n in _files(count)}:
        raise ValueError('goal-transfer inputs/config/artifact hashes differ')
    report=read(output/'report.json')
    if len(report['goals'])!=count:raise ValueError('goal-transfer report goal count differs')
    if render:
        from src.mylonas_ijepa_wm.radial_goal_transfer_plots import plot_transfer
        names=[f'goal_{j}_{name}' for j in range(count) for name in ('heatmaps','calibration')]+['goal_summary']
        if any(not (output/f'{name}.{ext}').is_file() for name in names for ext in ('png','pdf')):
            plot_transfer(output,report)
    return report


def load_report(source_root,output,stage,new_goal,*,render=True):
    output,source_root=locations(output,source_root)
    source=load_source(source_root,stage)
    execution=read(output/'complete.json')['inputs']['execution'];_signature(source,execution)
    return _load(output,_inputs(source,new_goal,execution),render)


def evaluate(source_root,output,stage,new_goal,env,encode,execution,*,device='cpu',batch_size=8,render=True):
    output,source_root=locations(output,source_root);source=load_source(source_root,stage)
    if batch_size<2:raise ValueError('goal-transfer batch size must be at least2')
    _signature(source,execution)
    old=np.asarray(source['design']['goal']);new_goals=_new_goals(new_goal)
    radius=source['metadata']['radius'];lower,upper=source['design']['bounds']
    distances=np.linalg.norm(new_goals-old,axis=1)
    if ((new_goals[:,0]<=35).any() or (new_goals<lower).any() or (new_goals>upper).any()
            or (distances>radius+1e-5).any() or (distances<1e-5).any()):
        raise ValueError('new goal must differ and stay inside original fitting region/radius')
    inputs=_inputs(source,new_goals,dict(execution,encoder_batch_size=batch_size))
    if (output/'complete.json').exists():return _load(output,inputs,render)
    if output.exists() and any(output.iterdir()):raise ValueError('incomplete/unrelated diagnostic output; choose a new directory')
    goals=np.vstack([old,new_goals]);count=len(goals)
    grid=source['design']['grid'];grid_ids=grid['point_ids']
    points=[source['design']['points'][i] for i in grid_ids];lookup={tuple(p):i for i,p in enumerate(points)}
    goal_ids=[]
    for goal in goals:
        key=tuple(goal)
        if key not in lookup:lookup[key]=len(points);points.append(list(key))
        goal_ids.append(lookup[key])
    env.reset(0,layout=[32,30],start=[44,34],goal=old.tolist())
    reference=env._render_at(points[0]);goal_frames=[env._render_at(old)]
    for new in new_goals:
        env.reset(0,layout=[32,30],start=[44,34],goal=new.tolist())
        if not np.array_equal(reference,env._render_at(points[0])):
            raise ValueError('goal change altered observation scene; this marker-free protocol does not apply')
        goal_frames.append(env._render_at(new))
    adapter=source['adapter'].to(device).eval().requires_grad_(False)
    before={k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()}
    # Encode goal coordinates first, then each other coordinate exactly once.
    # Query self-costs use the very same tokens as the reference, even when
    # encoder numerics depend on batch shape.
    order=goal_ids+[i for i in range(len(points)) if i not in goal_ids]
    raw=np.empty((count,len(points)));adapted=np.empty_like(raw);hashes=[None]*len(points)
    pending=[];target_parts=[];reference_count=0;goal_tokens=None
    import hashlib
    with torch.inference_mode():
        for start in range(0,len(order),batch_size):
            indices=order[start:start+batch_size]
            frames=np.stack([env._render_at(points[i]) for i in indices])
            tokens=encode(frames).detach().to(device).float()
            if tokens.ndim!=3 or len(tokens)!=len(indices) or tokens.shape[1:]!=source['cached_goal'].shape or not torch.isfinite(tokens).all():
                raise ValueError('invalid fresh goal-transfer tokens')
            for i,frame in zip(indices,frames):hashes[i]=hashlib.sha256(frame.tobytes()).hexdigest()
            if goal_tokens is None:
                pending.append((indices,tokens))
                take=min(count-reference_count,len(tokens));target_parts.append(tokens[:take]);reference_count+=take
                if reference_count<count:continue
                target_tokens=torch.cat(target_parts).cpu().clone();goal_tokens=target_tokens.to(device)
                batches=pending
            else:batches=[(indices,tokens)]
            for batch_ids,batch in batches:
                for j in range(count):
                    raw[j,batch_ids]=(batch-goal_tokens[j]).square().mean(dim=(1,2)).cpu().numpy()
                    adapted[j,batch_ids]=token_goal_cost(adapter,batch,goal_tokens[j]).cpu().numpy()
            pending=[];target_parts=[]
    if not np.isfinite(raw).all() or not np.isfinite(adapted).all():raise ValueError('nonfinite goal-transfer scores')
    offered=source['design']['stages'][stage-1]['training_ids']+[source['design']['goal_id']]
    fitted=np.asarray([source['design']['points'][i] for i in offered])
    offered_hashes={source['cache']['frame_sha256'][i] for i in offered}
    # Query-reference images are observed too, even if their coordinates differ
    # slightly from a grid point that rasterizes to exactly the same pixels.
    offered_hashes.update(hashes[i] for i in goal_ids)
    heldout=np.array([np.linalg.norm(fitted-np.asarray(p),axis=1).min()>1e-5 and h not in offered_hashes
        for p,h in zip(points,hashes)])
    heldout[goal_ids]=False
    nx,ny=len(grid['xs']),len(grid['ys']);edges=[]
    for y in range(ny):
        for x in range(nx):
            a=y*nx+x
            if x+1<nx:edges.append((a,a+1))
            if y+1<ny:edges.append((a,a+nx))
    calibration=source['report']['distance']['raw_calibration']
    rows=measure(points,goals,raw,adapted,radius,calibration,heldout,edges)
    for j,row in enumerate(rows):
        row.update(raw_self_cost=float(raw[j][goal_ids[j]]),adapted_self_cost=float(adapted[j][goal_ids[j]]))
    unchanged=all(torch.equal(before[k],v.detach().cpu()) for k,v in adapter.state_dict().items())
    if not unchanged or load_source(source_root,stage)['metadata']!=source['metadata']:
        raise ValueError('source checkpoint or frozen adapter changed during evaluation')
    old_raw=np.asarray(source['raw'])[grid_ids];old_adapt=np.asarray(source['report']['adapted_costs'])[grid_ids]
    report=dict(source=source['metadata'],execution=inputs['execution'],points=points,grid=grid,
        grid_count=len(grid_ids),goals=rows,distance_scale=SCALE,raw_calibration=calibration,
        heldout_mask=heldout.tolist(),adapter_unchanged=unchanged,training_positions=fitted.tolist(),
        original_goal_consistency=dict(raw_token_mse=float((target_tokens[0]-source['cached_goal']).square().mean()),
            mean_raw_grid_change=float(np.abs(np.asarray(raw[0])[:len(grid_ids)]-old_raw).mean()),
            mean_adapted_grid_change=float(np.abs(np.asarray(adapted[0])[:len(grid_ids)]-old_adapt).mean())),
        note='No fitting or recalibration. Main cohort stays inside original fitting disk and known distance range for every goal.')
    output.mkdir(parents=True,exist_ok=True)
    save_tensors(output/'goals.pt',dict(tokens=target_tokens,positions=goals.tolist()))
    for j,frame in enumerate(goal_frames):Image.fromarray(frame).save(output/f'goal_{j}.png')
    write_json(output/'report.json',report)
    if render:
        from src.mylonas_ijepa_wm.radial_goal_transfer_plots import plot_transfer
        plot_transfer(output,report)
    write_json(output/'complete.json',dict(inputs=inputs,files={n:file_sha256(output/n) for n in _files(count)}))
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=('evaluate','report'));p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--stage',type=int,default=4)
    p.add_argument('--new-goal',type=float,nargs=2,action='append',help='repeat for multiple goals');p.add_argument('--device',default='cuda')
    p.add_argument('--encoder-checkpoint',type=Path);p.add_argument('--dino-wm-root',type=Path)
    args=p.parse_args(argv)
    if args.new_goal is None:args.new_goal=[[46,38]]
    if args.operation=='report':
        r=load_report(args.source_run,args.output_dir,args.stage,args.new_goal)
        print(f"Verified goal-transfer report: {len(r['goals'])} goals, frozen stage{args.stage}");return r
    if args.encoder_checkpoint is None or args.dino_wm_root is None:p.error('evaluate needs encoder and renderer')
    locations(args.output_dir,args.source_run,(args.encoder_checkpoint,args.dino_wm_root))
    source=load_source(args.source_run,args.stage)
    if file_sha256(args.encoder_checkpoint)!=EXPECTED_ENCODER:raise ValueError('unexpected encoder checkpoint')
    if git_output(args.dino_wm_root,'rev-parse','HEAD')!=DINO_COMMIT or git_output(args.dino_wm_root,'status','--porcelain','--untracked-files=no'):
        raise ValueError('renderer must match pinned source')
    execution=dict(encoder_sha256=EXPECTED_ENCODER,renderer_commit=DINO_COMMIT,precision='fp32',
        torch_version=str(torch.__version__),device_type=torch.device(args.device).type,
        matmul_precision=torch.get_float32_matmul_precision())
    _signature(source,execution)
    from src.mylonas_ijepa_wm import plan_pointmaze as visual
    from src.mylonas_ijepa_wm.environment_adapters import WallAdapter
    from src.mylonas_ijepa_wm.environment_configs import EnvironmentConfig
    encoder=visual.load_frozen_ijepa_encoder(args.encoder_checkpoint,torch.device(args.device))
    runtime=SimpleNamespace(device=torch.device(args.device),precision='fp32',args=SimpleNamespace(encoder_batch_size=8),load_encoder=lambda:encoder)
    env=WallAdapter(EnvironmentConfig(env='wall',wall_layouts=[[32,30]],dino_wm_root=str(args.dino_wm_root)),100)
    try:r=evaluate(args.source_run,args.output_dir,args.stage,args.new_goal,env,lambda f:PilotRuntime.encode_frames(runtime,f),execution,device=args.device)
    finally:env.close()
    print('Completed frozen goal transfer; no fitting or source writes.');return r


if __name__=='__main__':main()
