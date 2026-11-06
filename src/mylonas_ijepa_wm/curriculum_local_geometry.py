"""Sample exposure, real-coordinate interpolation and descriptive local metrics."""

from collections import Counter, defaultdict

import numpy as np
import torch

from src.mylonas_ijepa_wm.curriculum_replay import eligible_pairs


def sampled_pairs(episodes, training):
    """Replay the fitter's RNG, including the unused zero-anchor draw."""
    pools = []
    margin = training.get('pair_filter', {}).get('min_goal_progress')
    for e, episode in enumerate(episodes):
        if episode.get('split') != 'train':
            raise ValueError('only training replay may enter local probes')
        if not episode.get('success'):
            continue
        _, pairs = eligible_pairs(episode['states'], episode['primitive_steps'],
            goal=episode['case']['goal'], min_goal_progress=margin)
        if pairs:
            pools.append((e, pairs))
    if not pools or [episodes[e]['case_id'] for e, _ in pools] != training['training_case_ids']:
        raise ValueError('reconstructed usable replay differs from fitted case IDs/order')
    groups = defaultdict(list)
    for entry in pools:
        groups[episodes[entry[0]]['case']['band']].append(entry)
    names = list(groups)
    balanced = training.get('sampling') == 'uniform_band_then_episode_then_pair'
    rng = np.random.default_rng(training['seed'])
    counts, inputs = Counter(), set()
    for _ in range(training['updates'] * training['batch_size']):
        pool = groups[names[int(rng.integers(len(names)))]] if balanced else pools
        e, pairs = pool[int(rng.integers(len(pool)))]
        i, j = pairs[int(rng.integers(len(pairs)))]
        anchor = int(rng.integers(len(episodes[e]['latents'])))
        counts[e, i, j] += 1
        inputs.update(((e, i), (e, j)))
        if training['anchor_weight']:
            inputs.add((e, anchor))
    return dict(pairs=[dict(episode=e, i=i, j=j, draws=n) for (e, i, j), n in counts.items()],
                inputs=sorted(inputs), total_pair_draws=sum(counts.values()))


def profiles(a, b, goal, bounds, *, wall_x=32, samples=11):
    a, b, goal = (np.asarray(v, dtype=float) for v in (a, b, goal))
    if (any(v.shape != (2,) or not np.isfinite(v).all() for v in (a,b,goal))
            or samples < 2):
        raise ValueError('finite 2D coordinates and at least two samples required')
    vector = b-a
    length = np.linalg.norm(vector)
    offsets = (0., -.5, .5, -1., 1., -2., 2.)
    valid = []
    if length > 1e-8:
        normal = np.array([-vector[1], vector[0]]) / length
        for offset in offsets:
            start, end = a+offset*normal, b+offset*normal
            points = start + np.linspace(0,1,samples)[:,None]*vector
            # Squared distance has increasing derivative along a straight line;
            # checking its endpoint derivative excludes hidden goal overshoots.
            if (np.any(points < bounds[0]) or np.any(points > bounds[1])
                    or np.any(points[:,0] <= wall_x+3)
                    or np.dot(end-goal,vector) > 1e-8
                    or np.linalg.norm(start-goal)-np.linalg.norm(end-goal) < .5):
                continue
            valid.append(dict(offset=offset,points=points.tolist()))
    return dict(valid=valid, rejected=len(offsets)-len(valid))


def select_pairs(pairs, episodes, goal, bounds, *, count=50, seed=20261106):
    if count < 1:
        raise ValueError('positive pair count required')
    groups = defaultdict(list)
    for pair in pairs:
        states = episodes[pair['episode']]['states']
        a, b = states[pair['i']].tolist(), states[pair['j']].tolist()
        if not any(p['offset']==0 for p in profiles(a,b,goal,bounds)['valid']):
            continue
        length = float(np.linalg.norm(np.array(b)-a))
        bucket = int(np.searchsorted([4,8,16],length,side='right'))
        groups[pair['episode'],bucket].append(dict(pair,a=a,b=b,length=length))
    eligible = sum(map(len,groups.values()))
    rng = np.random.default_rng(seed)
    keys = list(groups)
    rng.shuffle(keys)
    for values in groups.values():
        rng.shuffle(values)
    selected = []
    while len(selected) < count and any(groups.values()):
        for key in keys:
            if groups[key] and len(selected)<count:
                selected.append(groups[key].pop())
    return dict(selected=selected,geometrically_eligible=eligible,
                sampled_unique_pairs=len(pairs),selection_seed=seed)


def curve_metrics(costs):
    values = np.asarray(costs,dtype=float)
    if values.ndim != 1 or len(values)<2 or not np.isfinite(values).all():
        raise ValueError('a finite cost curve with at least two points is required')
    tolerance = max(1e-7,1e-3*float(np.ptp(values)))
    delta = np.diff(values)
    up, down = delta>tolerance, delta < -tolerance
    contrast = float(values[0]-values[-1])
    return dict(tolerance=tolerance,endpoint_correct=bool(contrast>tolerance),
        endpoint_contrast=contrast,up_steps=int(up.sum()),down_steps=int(down.sum()),
        tie_steps=int((~(up|down)).sum()),up_fraction=float(up.mean()),
        max_uphill=float(np.maximum(delta,0).max()),
        uphill_normalized=float(delta[up].sum()/abs(contrast)) if abs(contrast)>tolerance else None,
        informative_monotone=bool(not up.any() and down.any()),flat=bool(not (up|down).any()))


def coverage(points, fitting_positions, replay_positions):
    points = np.asarray(points,dtype=float)
    def distances(reference):
        reference = np.asarray(reference,dtype=float).reshape(-1,2)
        if not len(reference):
            raise ValueError('coverage requires nonempty reference positions')
        return np.concatenate([np.linalg.norm(chunk[:,None]-reference[None],axis=2).min(axis=1)
                               for chunk in np.array_split(points,max(1,(len(points)+255)//256))])
    fitted, replay = distances(fitting_positions), distances(replay_positions)
    return dict(nearest_fitting_distance=fitted.tolist(),nearest_replay_distance=replay.tolist(),
                fitted_position=(fitted<=1e-5).tolist(),replay_position=(replay<=1e-5).tolist())
