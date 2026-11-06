"""Read-only fresh-image probes around actual clean no-anchor v0 fitting pairs."""

import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.mylonas_ijepa_wm import curriculum_heatmaps as heatmaps
from src.mylonas_ijepa_wm import curriculum_local_geometry as geometry
from src.mylonas_ijepa_wm import curriculum_multiband_protocol as protocol
from src.mylonas_ijepa_wm.cost_adapter import token_goal_cost
from src.mylonas_ijepa_wm.curriculum_multiband import Curriculum
from src.mylonas_ijepa_wm.curriculum_protocol import write_json
from src.mylonas_ijepa_wm.curriculum_runtime import DINO_COMMIT, PilotRuntime, file_sha256, git_output


def _output(curriculum, output):
    output = Path(output).resolve()
    sources = (curriculum.root,curriculum.parent,curriculum.clean,curriculum.adjacent)
    if any(output==p or p in output.parents or output in p.parents for p in sources):
        raise ValueError('diagnostic output must be separate from all source trees')
    if output.exists() and any(output.iterdir()):
        raise ValueError('use a new empty diagnostic output directory')
    return output


def _aggregate(rows, method):
    items = [row[method] for row in rows]
    unseen = sum(m['unseen_steps'] for m in items)
    unseen_up = sum(m['unseen_up_steps'] for m in items)
    return dict(profiles=len(items),endpoint_accuracy=float(np.mean([m['endpoint_correct'] for m in items])),
        informative_monotone_fraction=float(np.mean([m['informative_monotone'] for m in items])),
        up_step_fraction=float(np.mean([m['up_fraction'] for m in items])),
        flat_profiles=sum(m['flat'] for m in items),
        unseen_steps=unseen,unseen_up_steps=unseen_up,
        unseen_up_fraction=unseen_up/unseen if unseen else None,
        median_normalized_uphill=float(np.median([m['uphill_normalized'] for m in items
            if m['uphill_normalized'] is not None])) if any(m['uphill_normalized'] is not None for m in items) else None)


def audit(curriculum, output, env, encode, *, pair_count=50, batch_size=8, device='cpu', execution=None):
    output = _output(curriculum,output)
    if batch_size<1:
        raise ValueError('positive encoder batch size required')
    if curriculum.config.get('ablation') != protocol.NO_ANCHOR:
        raise ValueError('this diagnostic requires the fresh no-anchor experiment')
    # Never allow check_runtime to create source metadata on a damaged run.
    for name in ('execution.json','branches/clean/runtime.json'):
        if not (curriculum.root/name).is_file():
            raise ValueError('completed source execution metadata missing')
    curriculum.guard()
    adapter, saved, checkpoint_sha = curriculum.checkpoint('clean',0,device)
    training = saved['training']
    entries = curriculum.replay_entries('clean',0)
    episodes = curriculum.load_replay('clean',0,entries)
    paths = [Path(item['path']) for item in entries]
    paths += [curriculum.root/name for name in ('experiment.json','manifest.json','execution.json',
        'cache/goal.pt','cache/goal.json','branches/clean/runtime.json','branches/clean/checkpoints/v0.pt',
        'branches/clean/updates/v0/status.json','branches/clean/updates/v0/training.json')]
    before = {str(path):file_sha256(path) for path in paths}
    sampled = geometry.sampled_pairs(episodes,training)
    env.reset(0,layout=[32,30],start=[44,34],goal=[50,42])
    bounds = heatmaps.physical_bounds(env.env)
    selection = geometry.select_pairs(sampled['pairs'],episodes,[50,42],bounds,count=pair_count)
    if not selection['selected']:
        raise ValueError('no sampled fitting pairs have eligible interpolation geometry')
    fitting = [episodes[e]['states'][i].tolist() for e,i in sampled['inputs']]
    replay = torch.cat([episode['states'] for episode in episodes]).tolist()
    # Goal is also passed to the adapter at every fitting update.
    fitting.append([50.,42.])
    target = curriculum.goal.to(device)
    adapter.eval().requires_grad_(False)
    cache = {}

    @torch.no_grad()
    def score(points):
        keys = [tuple(map(float,point)) for point in points]
        missing = list(dict.fromkeys(key for key in keys if key not in cache))
        for start in range(0,len(missing),batch_size):
            chunk = missing[start:start+batch_size]
            frames = np.stack([env._render_at(point) for point in chunk])
            tokens = encode(frames).to(device=device,dtype=torch.float32)
            if (tokens.ndim!=3 or tokens.shape[0]!=len(chunk) or tokens.shape[1:]!=target.shape
                    or not torch.isfinite(tokens).all()):
                raise ValueError('fresh encoder returned invalid/misaligned tokens')
            raw = (tokens-target).square().mean(dim=(1,2)).cpu().tolist()
            adapted = token_goal_cost(adapter,tokens,target).cpu().tolist()
            for key,frame,a,b in zip(chunk,frames,raw,adapted):
                if not np.isfinite([a,b]).all():
                    raise ValueError('nonfinite fresh costs')
                cache[key] = dict(raw=a,adapted=b,frame_sha256=hashlib.sha256(frame.tobytes()).hexdigest())
        return [cache[key] for key in keys]

    # Hash freshly rendered positions conservatively; this does not assert that
    # unknown historical goal pixels were identical to a fresh goal render.
    def frame_hashes(positions):
        return {hashlib.sha256(env._render_at(p).tobytes()).hexdigest() for p in positions}
    fitting_hashes, replay_hashes = frame_hashes(fitting), frame_hashes(replay)
    curves, rejected, endpoint_checks = [],0,[]
    for pair_id,pair in enumerate(selection['selected']):
        generated = geometry.profiles(pair['a'],pair['b'],[50,42],bounds)
        rejected += generated['rejected']
        for profile in generated['valid']:
            positions = profile['points']
            values = score(positions)
            exposure = geometry.coverage(positions,fitting,replay)
            hashes = [v['frame_sha256'] for v in values]
            exposure['same_render_as_fitting_position'] = [h in fitting_hashes for h in hashes]
            exposure['same_render_as_replay_position'] = [h in replay_hashes for h in hashes]
            unseen = [not (p or h) for p,h in zip(exposure['fitted_position'],
                                                exposure['same_render_as_fitting_position'])]
            exposure['unseen_relative_to_fitting_inputs'] = unseen
            row = dict(pair_id=pair_id,case_id=episodes[pair['episode']]['case_id'],pair=pair,
                offset=profile['offset'],points=positions,coverage=exposure,frame_sha256=hashes)
            for method in ('raw','adapted'):
                costs = [v[method] for v in values]
                metrics = geometry.curve_metrics(costs)
                tol = metrics['tolerance']
                new_steps = np.asarray(unseen[:-1]) & np.asarray(unseen[1:])
                metrics['unseen_steps'] = int(new_steps.sum())
                metrics['unseen_up_steps'] = int(((np.diff(costs)>tol)&new_steps).sum())
                row[method],row[method+'_costs'] = metrics,costs
            curves.append(row)
            if profile['offset']==0:
                episode = episodes[pair['episode']]
                tokens = episode['latents'][[pair['i'],pair['j']]].to(device).float()
                with torch.no_grad():
                    old = token_goal_cost(adapter,tokens,target).cpu().tolist()
                fresh = [row['adapted_costs'][0],row['adapted_costs'][-1]]
                old_correct = geometry.curve_metrics(old)['endpoint_correct']
                endpoint_checks.append(dict(pair_id=pair_id,replay_costs=old,fresh_costs=fresh,
                    replay_correct=old_correct,fresh_correct=row['adapted']['endpoint_correct'],
                    ranking_changed=old_correct!=row['adapted']['endpoint_correct']))
        print(f"Pair {pair_id+1}/{len(selection['selected'])}: scored {len(generated['valid'])} profiles",flush=True)
    grid_points = heatmaps.points(bounds[1]).tolist()
    values = score(grid_points)
    grid = dict(points=grid_points,raw=[v['raw'] for v in values],adapted=[v['adapted'] for v in values],
        x_max=bounds[1],reset_safe_x_max=float(env.env.img_size-1-env.env.border_padding))
    by_offset = defaultdict(list)
    for row in curves:
        by_offset[abs(row['offset'])].append(row)
    summaries = [dict(offset=offset,raw=_aggregate(rows,'raw'),adapted=_aggregate(rows,'adapted'))
                 for offset,rows in sorted(by_offset.items())]
    distance_bins = []
    for lo,hi in ((0,.01),(.01,.5),(.5,1),(1,2),(2,float('inf'))):
        block = dict(lower=lo,upper=hi if np.isfinite(hi) else None)
        for method in ('raw','adapted'):
            count,up = 0,0
            for row in curves:
                distances = np.asarray(row['coverage']['nearest_fitting_distance'])
                nearest = np.minimum(distances[:-1],distances[1:])
                mask = (nearest>=lo)&(nearest<hi)
                count += int(mask.sum())
                up += int(((np.diff(row[method+'_costs'])>row[method]['tolerance'])&mask).sum())
            block[method] = dict(steps=count,up_steps=up,up_fraction=up/count if count else None)
        distance_bins.append(block)
    if any(file_sha256(path)!=sha for path,sha in before.items()):
        raise ValueError('source artifacts changed during diagnostic')
    result = dict(schema_version=1,checkpoint_sha256=checkpoint_sha,goal_sha256=curriculum.goal_sha,
        source_hashes=before,fresh_execution=execution or {},selected_pairs=len(selection['selected']),
        selection=selection,sampling=dict(total_pair_draws=sampled['total_pair_draws'],
            unique_sampled_pairs=len(sampled['pairs']),unique_fitting_replay_inputs=len(sampled['inputs'])),
        rejected_offset_profiles=rejected,offset_summary=summaries,distance_bins=distance_bins,
        endpoint_reencoding=dict(pairs=len(endpoint_checks),
            ranking_changes=sum(row['ranking_changed'] for row in endpoint_checks),details=endpoint_checks),
        grid=grid,fitting_positions=fitting,primitive_interactions=0,
        note='Descriptive probes, not independent trials. Fresh rendering at fitting coordinates is a conservative exposure check; cached target retained. No fitting, planning or world-model predictions.')
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'probes.json',curves)
    write_json(output/'report.json',result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent-dir','clean-dir','adjacent-dir','curriculum-dir','output-dir','encoder-checkpoint','dino-wm-root'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--pair-count',type=int,default=50)
    parser.add_argument('--device',default='cuda')
    args = parser.parse_args(argv)
    curriculum = Curriculum(args.parent_dir,args.clean_dir,args.adjacent_dir,args.curriculum_dir)
    _output(curriculum,args.output_dir)
    if file_sha256(args.encoder_checkpoint)!=curriculum.config['sources']['models']['encoder']:
        raise ValueError('encoder differs from the saved experiment')
    if (git_output(args.dino_wm_root,'rev-parse','HEAD')!=DINO_COMMIT
            or git_output(args.dino_wm_root,'status','--porcelain','--untracked-files=no')):
        raise ValueError('renderer source must match the unmodified pinned DINO revision')
    from src.mylonas_ijepa_wm import plan_pointmaze as visual
    from src.mylonas_ijepa_wm.environment_adapters import WallAdapter
    from src.mylonas_ijepa_wm.environment_configs import EnvironmentConfig
    device = torch.device(args.device)
    encoder = visual.load_frozen_ijepa_encoder(args.encoder_checkpoint,device)
    env = WallAdapter(EnvironmentConfig(env='wall',wall_layouts=[[32,30]],dino_wm_root=str(args.dino_wm_root)),100)
    runtime = SimpleNamespace(device=device,precision='fp32',
        args=SimpleNamespace(encoder_batch_size=8),load_encoder=lambda:encoder)
    try:
        report = audit(curriculum,args.output_dir,env,
            lambda frames:PilotRuntime.encode_frames(runtime,frames),device=device,
            pair_count=args.pair_count,execution=dict(torch_version=torch.__version__,device_type=device.type,
                precision='fp32',encoder_batch_size=8,renderer_commit=DINO_COMMIT,
                encoder_sha256=curriculum.config['sources']['models']['encoder']))
    finally:
        env.close()
    from src.mylonas_ijepa_wm.curriculum_local_plots import render
    render(args.output_dir)
    print('Completed diagnostic:',args.output_dir,flush=True)
    return report


if __name__=='__main__':
    main()
