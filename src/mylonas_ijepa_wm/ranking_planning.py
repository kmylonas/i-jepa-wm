"""Paired frozen raw/adapted multi-goal world-model planning evaluation."""
import argparse
import hashlib
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from src.mylonas_ijepa_wm import plan_proprio,plan_pointmaze as visual
from src.mylonas_ijepa_wm.cost_adapter import adapter_goal_cost
from src.mylonas_ijepa_wm.curriculum_multiband_protocol import read
from src.mylonas_ijepa_wm.curriculum_protocol import digest,write_json
from src.mylonas_ijepa_wm.curriculum_runtime import (
    EXPECTED_ENCODER,EXPECTED_WM,PilotRuntime,InteractionJournal,interaction_budget,
    file_sha256,save_tensors,source_digest)
from src.mylonas_ijepa_wm.radial_experiment import locations
from src.mylonas_ijepa_wm.ranking_planning_protocol import load_ranking_source,make_manifest,summarize,in_region


CONDITIONS=('raw','adapted')
def _frame_hash(frame):return hashlib.sha256(np.asarray(frame).tobytes()).hexdigest()
def _hashes(root,names):return {n:file_sha256(root/n) for n in names}
def _root(output,role):
    if role not in (*CONDITIONS,'reference'):raise ValueError('unknown planning condition')
    return Path(output)/('reference' if role=='reference' else f'conditions/{role}')


def prepare(source,output,*,goals=None,starts_per_goal=8,models=None):
    output,source=locations(output,source)
    _,snapshot,design=load_ranking_source(source)
    manifest=make_manifest(design,starts_per_goal,goals)
    models=dict(world_model=EXPECTED_WM,encoder=EXPECTED_ENCODER) if models is None else dict(models)
    if models['encoder']!=snapshot['fitting_signature']['encoder_sha256']:
        raise ValueError('encoder differs from frozen ranking source')
    config=dict(kind='ranking_planning_v1',source=snapshot,models=models,manifest_digest=digest(manifest),
        diagnostic_source=source_digest(Path(__file__).parents[1]))
    for name,value in (('experiment.json',config),('manifest.json',manifest)):
        if (output/name).is_file() and read(output/name)!=value:raise ValueError('planning inputs/config changed; use a new output')
    if output.exists() and any(output.iterdir()) and not (output/'experiment.json').is_file():
        raise ValueError('incomplete/unrelated planning output')
    write_json(output/'experiment.json',config);write_json(output/'manifest.json',manifest)
    return config


def _validated(source,output):
    output,source=locations(output,source);config=read(output/'experiment.json');manifest=read(output/'manifest.json')
    adapter,snapshot,design=load_ranking_source(source)
    if (config['kind']!='ranking_planning_v1' or config['source']!=snapshot
            or config['manifest_digest']!=digest(manifest)
            or manifest!=make_manifest(design,manifest['starts_per_goal'],manifest['goals'])
            or config['diagnostic_source']!=source_digest(Path(__file__).parents[1])
            or config['models']['encoder']!=snapshot['fitting_signature']['encoder_sha256']):
        raise ValueError('planning source/config/manifest signature differs')
    return config,manifest,adapter,design


def runtime_variant(config,role,device='cpu',videos=False):
    device=torch.device(device)
    return dict(experiment_digest=digest(config),condition=role,
        device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else device.type,
        matmul_precision=torch.get_float32_matmul_precision(),videos=bool(videos))


def _runtime(output,config,manifest,role,runtime=None):
    root=_root(output,role);saved=read(root/'runtime.json');signature=saved['signature']
    variant=signature.get('runtime_variant',{})
    if (saved['runtime_id']!=digest(signature) or signature.get('manifest')!=digest(manifest)
            or signature.get('models')!=config['models'] or signature.get('source')!=config['diagnostic_source']
            or signature.get('precision')!='fp32' or signature.get('encoder_batch_size')!=8
            or signature.get('dino_commit')!=config['source']['fitting_signature']['renderer_commit']
            or variant.get('experiment_digest')!=digest(config) or variant.get('condition')!=role):
        raise ValueError('planning runtime signature differs')
    if runtime is not None and (Path(runtime.root).resolve()!=root.resolve() or runtime.manifest!=manifest
            or runtime.runtime_id!=saved['runtime_id'] or runtime.precision!='fp32'
            or variant!=runtime_variant(config,role,runtime.device,getattr(runtime.args,'videos',False))):
        raise ValueError('planning runtime root/device/config differs')
    return root,saved


def _reference_inputs(config,manifest,saved):
    return dict(experiment_digest=digest(config),goals=manifest['goals'],runtime_id=saved['runtime_id'])


def _references(output,config,manifest):
    root,saved=_runtime(output,config,manifest,'reference')
    complete=read(root/'complete.json')
    names=('goals.pt','consistency.json',*(f'goal_{i}.png' for i in range(len(manifest['goals']))))
    if complete['inputs']!=_reference_inputs(config,manifest,saved) or complete['files']!=_hashes(root,names):
        raise ValueError('shared goal reference input/artifact hash differs')
    payload=torch.load(root/'goals.pt',map_location='cpu',weights_only=True)
    if (payload['positions']!=manifest['goals'] or payload['tokens'].ndim!=3
            or len(payload['tokens'])!=len(manifest['goals']) or not torch.isfinite(payload['tokens']).all()):
        raise ValueError('shared goal reference shape/position differs')
    return payload,saved


def prepare_goals(source,output,runtime):
    output=Path(output).resolve();config,manifest,_,design=_validated(source,output)
    root,saved=_runtime(output,config,manifest,'reference',runtime)
    if (root/'complete.json').is_file():return _references(output,config,manifest)[0]
    if (root/'goals.pt').exists():raise ValueError('incomplete reference encoding; choose a new output')
    frames=[];fixed=None
    for goal in manifest['goals']:
        runtime.env.reset(0,layout=[32,30],start=[44,34],goal=goal)
        rendered=runtime.env._render_at([44,34])
        if fixed is not None and not np.array_equal(fixed,rendered):raise ValueError('goal marker changed encoder observation')
        fixed=rendered;frames.append(runtime.env._render_at(goal))
    offered_hashes={design['frame_sha256'][i] for i in design['train_ids']}
    reference_hashes={_frame_hash(f) for f in frames}
    for case in manifest['evaluation']:
        h=_frame_hash(runtime.env._render_at(case['start']))
        if h in offered_hashes|reference_hashes:raise ValueError('planning start aliases a fitting/reference image')
    tokens=runtime.encode_frames(np.stack(frames)).detach().cpu().float()
    cached=torch.load(Path(config['source']['original_source']['root'])/'goal.pt',map_location='cpu',weights_only=True)['tokens']
    if tokens.shape!=(len(frames),*cached.shape) or not torch.isfinite(tokens).all():raise ValueError('invalid planning goal tokens')
    payload=dict(tokens=tokens,positions=manifest['goals'],frame_sha256=[_frame_hash(f) for f in frames])
    save_tensors(root/'goals.pt',payload)
    for i,frame in enumerate(frames):Image.fromarray(frame).save(root/f'goal_{i}.png')
    write_json(root/'consistency.json',dict(original_cached_fresh_token_mse=float((tokens[0]-cached).square().mean())))
    _validated(source,output)
    names=('goals.pt','consistency.json',*(f'goal_{i}.png' for i in range(len(frames))))
    write_json(root/'complete.json',dict(inputs=_reference_inputs(config,manifest,saved),files=_hashes(root,names)))
    return payload


class PlanningRuntime(PilotRuntime):
    @torch.no_grad()
    def run_case(self,case,target,metric=None,goal_frame_sha256=None):
        self.load_world_model();self.load_encoder()
        self.model.eval().requires_grad_(False);self.encoder.eval().requires_grad_(False)
        target=target.detach().to(self.device).float()
        if metric is not None:scorer=adapter_goal_cost(metric,target)
        else:
            @torch.no_grad()
            def scorer(tokens):return (tokens.float()-target.unsqueeze(0)).square().mean(dim=(1,2))
        initial=self.env.reset(case['seed'],layout=case['layout'],start=case['start'],goal=case['goal'])
        if goal_frame_sha256 is not None and _frame_hash(self.env.render_goal())!=goal_frame_sha256:
            raise ValueError('case goal image differs from shared target reference')
        positions=[];video=[] if self.args.videos else None
        def observe(step,outcome,action):
            positions.append(np.asarray(outcome.state)[:2].tolist())
            if step:journal.record(step,action)
        args=SimpleNamespace(**self.manifest['planning'],cost='latent')
        with InteractionJournal(self.root,self.runtime_id,'evaluate',case['case_id']) as journal:
            record=plan_proprio.run_episode(self.env,self.config,self.encoder,self.model,self.metadata,args,
                self.device,self.precision,case['seed'],video,initial=initial,latent_cost_fn=scorer,observation_observer=observe)
        xy=np.asarray(positions);inside=in_region(xy)
        record.update(attempt_id=journal.identity['attempt_id'],primitive_positions=positions,
            path_length=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()),
            ever_outside_fitting=bool((~inside).any()),outside_fitting_steps=int((~inside[1:]).sum()),
            ever_left_right_side=bool((xy[:,0]<=35).any()),video_path=None)
        if video:
            index=int(case['case_id'].split('_s')[-1])
            record['video_path']=str(visual.save_episode_video(video,self.root/'videos'/f"g{case['goal_id']:02d}",index,case['seed'],record['success'],10))
        return record


def _condition_inputs(config,references,saved):
    return dict(experiment_digest=digest(config),runtime_id=saved['runtime_id'],
        reference_tokens_sha256=file_sha256(references/'goals.pt'))


def _episode(root,case,inputs):
    path=root/'episodes'/f"{case['case_id']}.json";marker=path.with_suffix('.complete.json')
    if not marker.exists():return None
    complete=read(marker)
    if complete['inputs']!=dict(inputs,case=case) or complete['record_sha256']!=file_sha256(path):
        raise ValueError('episode input/artifact hash differs')
    record=read(path)
    if record.get('case')!=case or record.get('runtime_id')!=inputs['runtime_id'] or record.get('case_id')!=case['case_id']:
        raise ValueError('episode provenance differs')
    video=complete.get('video')
    if video and file_sha256(root/video['path'])!=video['sha256']:raise ValueError('episode video artifact hash differs')
    return record


def evaluate_condition(source,output,condition,runtime,*,limit=None):
    if condition not in CONDITIONS:raise ValueError('unknown planning condition')
    if limit is not None and (not isinstance(limit,int) or limit<1):raise ValueError('positive case limit required')
    output=Path(output).resolve();config,manifest,metric,_=_validated(source,output)
    root,saved=_runtime(output,config,manifest,condition,runtime)
    refs,ref_saved=_references(output,config,manifest)
    a=dict(saved['signature']);b=dict(ref_saved['signature'])
    a['runtime_variant']=dict(a['runtime_variant']);b['runtime_variant']=dict(b['runtime_variant'])
    a['runtime_variant'].pop('condition');b['runtime_variant'].pop('condition')
    if a!=b:raise ValueError('goal encoding and planning runtime signatures differ')
    inputs=_condition_inputs(config,_root(output,'reference'),saved)
    status=root/'stage.json'
    if status.exists() and read(status)['inputs']!=inputs:raise ValueError('planning stage inputs differ')
    write_json(status,dict(inputs=inputs,status='running'))
    metric=metric.to(runtime.device).eval().requires_grad_(False) if condition=='adapted' else None
    before={k:v.detach().cpu().clone() for k,v in metric.state_dict().items()} if metric is not None else None
    records=[];cases=manifest['evaluation'] if limit is None else manifest['evaluation'][:limit]
    for index,case in enumerate(cases):
        record=_episode(root,case,inputs)
        if record is None:
            started=time.monotonic();gi=case['goal_id']
            record=runtime.run_case(case,refs['tokens'][gi],metric=metric,goal_frame_sha256=refs['frame_sha256'][gi])
            record.update(case=case,case_id=case['case_id'],runtime_id=runtime.runtime_id,condition=condition,
                elapsed_seconds=time.monotonic()-started)
            path=root/'episodes'/f"{case['case_id']}.json";write_json(path,record)
            video=None
            if record.get('video_path'):
                p=Path(record['video_path']).resolve();relative=p.relative_to(root.resolve())
                video=dict(path=str(relative),sha256=file_sha256(p))
            write_json(path.with_suffix('.complete.json'),dict(inputs=dict(inputs,case=case),record_sha256=file_sha256(path),video=video))
        records.append(record)
        print(f"{condition} {index+1}/{len(cases)} {case['case_id']} goal={case['goal']} success={record['success']} steps={record['micro_steps']} final_d={record['final_goal_distance']:.2f}",flush=True)
    if metric is not None and not all(torch.equal(before[k],v.detach().cpu()) for k,v in metric.state_dict().items()):
        raise ValueError('frozen ranking adapter changed in memory')
    _validated(source,output)
    completed=[r for case in manifest['evaluation'] if (r:=_episode(root,case,inputs)) is not None]
    write_json(status,dict(inputs=inputs,status='complete' if len(completed)==len(manifest['evaluation']) else 'partial',
        completed=len(completed),total=len(manifest['evaluation']),primitive_interactions_committed=sum(r['micro_steps'] for r in completed)))
    return records


def report(source,output,*,render=True):
    output=Path(output).resolve();config,manifest,_,_=_validated(source,output)
    _references(output,config,manifest);records={};journals={}
    for condition in CONDITIONS:
        root,saved=_runtime(output,config,manifest,condition)
        inputs=_condition_inputs(config,_root(output,'reference'),saved);stage=read(root/'stage.json')
        if stage['inputs']!=inputs or stage['status']!='complete':raise ValueError('planning condition is incomplete')
        records[condition]=[_episode(root,c,inputs) for c in manifest['evaluation']]
        if any(r is None for r in records[condition]):raise ValueError('missing committed planning episode')
        journals[condition]=interaction_budget(root,saved['runtime_id'])
    result=dict(manifest=manifest,source=config['source'],references_unchanged=True,
        summary=summarize(records['raw'],records['adapted']),records=records,journals=journals,
        reference_consistency=read(_root(output,'reference')/'consistency.json'))
    write_json(output/'report.json',result)
    if render:
        from src.mylonas_ijepa_wm.ranking_planning_plots import plot_report
        plot_report(output,result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=('prepare','goals','evaluate','report'))
    p.add_argument('--source-run',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--condition',choices=CONDITIONS);p.add_argument('--limit',type=int)
    p.add_argument('--world-model-checkpoint',type=Path);p.add_argument('--encoder-checkpoint',type=Path)
    p.add_argument('--dino-wm-root',type=Path);p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    p.add_argument('--videos',action='store_true');a=p.parse_args(argv)
    if a.operation=='prepare':return prepare(a.source_run,a.output_dir)
    if a.operation=='report':return report(a.source_run,a.output_dir)
    if any(v is None for v in (a.world_model_checkpoint,a.encoder_checkpoint,a.dino_wm_root)):
        p.error('goals/evaluate require encoder, world model and renderer')
    if a.operation=='evaluate' and a.condition is None:p.error('evaluate requires --condition')
    locations(a.output_dir,a.source_run,(a.world_model_checkpoint,a.encoder_checkpoint,a.dino_wm_root))
    config,manifest,_,_=_validated(a.source_run,a.output_dir)
    role='reference' if a.operation=='goals' else a.condition
    args=SimpleNamespace(output_dir=_root(a.output_dir,role),world_model_checkpoint=a.world_model_checkpoint,
        encoder_checkpoint=a.encoder_checkpoint,dino_wm_root=a.dino_wm_root,device=a.device,precision='fp32',
        encoder_batch_size=8,expected_world_model_sha256=config['models']['world_model'],
        expected_encoder_sha256=config['models']['encoder'],run_seed=0,videos=a.videos,
        runtime_signature_extra=runtime_variant(config,role,a.device,a.videos))
    runtime=PlanningRuntime(args,manifest)
    try:
        if role=='reference':return prepare_goals(a.source_run,a.output_dir,runtime)
        return evaluate_condition(a.source_run,a.output_dir,role,runtime,limit=a.limit)
    finally:runtime.close()


if __name__=='__main__':main()
