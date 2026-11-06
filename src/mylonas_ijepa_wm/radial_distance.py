"""Fit adapted token MSE to physical Euclidean distance with a fixed scale."""
from copy import deepcopy

import numpy as np
import torch

from src.mylonas_ijepa_wm.cost_adapter import ResidualTokenAdapter,token_goal_cost


DISTANCE_SCALE=34.


def fit(tokens,radii,goal,*,seed,updates=500,batch_size=32,lr=1e-4,bottleneck=64,
        initial_adapter=None,device='cpu',progress=None):
    tokens,goal=tokens.detach().cpu().float(),goal.detach().cpu().float()
    radii=torch.as_tensor(radii).detach().cpu().float()
    if (tokens.ndim!=3 or tokens.shape[1:]!=goal.shape or not len(tokens)
            or radii.shape!=(len(tokens),) or not torch.isfinite(radii).all() or (radii<0).any()
            or not torch.isfinite(tokens).all() or not torch.isfinite(goal).all()
            or updates<1 or batch_size<1 or not np.isfinite(lr) or lr<=0):
        raise ValueError('invalid distance fitting inputs or schedule')
    if initial_adapter is None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            adapter=ResidualTokenAdapter(goal.shape[-1],bottleneck).to(device)
    else:
        if initial_adapter.dim!=goal.shape[-1] or initial_adapter.bottleneck!=bottleneck:
            raise ValueError('incoming adapter dimensions differ')
        adapter=deepcopy(initial_adapter).to(device).requires_grad_(True).train()
    target=goal.to(device)
    labels=radii/DISTANCE_SCALE
    @torch.no_grad()
    def full_metrics():
        costs=torch.cat([token_goal_cost(adapter,chunk.to(device),target).cpu() for chunk in tokens.split(16)])
        residual=costs-labels
        return dict(full_regression_loss=float(residual.square().mean()),
            distance_mae=float(residual.abs().mean()*DISTANCE_SCALE),
            distance_rmse=float(residual.square().mean().sqrt()*DISTANCE_SCALE))
    before=full_metrics()
    optimizer=torch.optim.Adam(adapter.parameters(),lr=lr)
    rng=np.random.default_rng(seed);counts=torch.zeros(len(tokens),dtype=torch.int64)
    history=[dict(update=0,**before)]
    for update in range(updates):
        index=torch.as_tensor(rng.integers(len(tokens),size=batch_size))
        counts+=torch.bincount(index,minlength=len(tokens))
        costs=token_goal_cost(adapter,tokens[index].to(device),target)
        loss=(costs-labels[index].to(device)).square().mean()
        if not torch.isfinite(loss):raise ValueError('nonfinite distance regression loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        row=dict(update=update+1,regression_loss=float(loss.detach()))
        if update==0 or (update+1)%50==0 or update+1==updates:row.update(full_metrics())
        history.append(row)
        if progress and (update==0 or (update+1)%50==0):progress(row)
    after=full_metrics();adapter.eval()
    return adapter,dict(seed=seed,updates=updates,batch_size=batch_size,lr=lr,bottleneck=bottleneck,
        anchor_weight=0,objective='distance',distance_scale=DISTANCE_SCALE,
        loss='mean((adapted_mse-radius/34)^2)',sampling='uniform_positions_with_replacement',
        initialization='identity' if initial_adapter is None else 'previous_checkpoint',optimizer_reset=True,
        sample_counts=counts.tolist(),history=history,
        full_regression_loss_before=before['full_regression_loss'],
        full_regression_loss_after=after['full_regression_loss'],
        distance_mae_before=before['distance_mae'],distance_mae_after=after['distance_mae'],
        distance_rmse_before=before['distance_rmse'],distance_rmse_after=after['distance_rmse'])


def distance_summary(design,raw,adapted,report):
    """Calibrate on offered training only; evaluate fixed, deduplicated probes."""
    radii=np.linalg.norm(np.asarray(design['points'])-np.asarray(design['goal']),axis=1)
    raw,adapted=np.asarray(raw),np.asarray(adapted)
    train=np.asarray(report['offered_training_ids'],dtype=int)
    heldout=np.asarray(report['exposure']['heldout_from_offered_training'],dtype=bool)
    denominator=float(raw[train]@raw[train])
    calibration=max(0.,float(raw[train]@radii[train])/denominator) if denominator>0 else 0.
    predictions=dict(raw_calibrated=raw*calibration,adapted=adapted*DISTANCE_SCALE)
    def measured(ids):
        ids=np.asarray(sorted(set(ids)),dtype=int)
        row=dict(count=len(ids))
        for method,values in predictions.items():
            error=values[ids]-radii[ids]
            row[method]=dict(mae=float(np.abs(error).mean()) if len(ids) else None,
                rmse=float(np.sqrt(np.mean(error**2))) if len(ids) else None)
        return row
    def probe_ids(kind=None):
        return sorted({i for curve in design['curves'] if kind is None or curve['kind']==kind
                       for i in curve['point_ids'] if heldout[i]})
    def regions(ids):
        return {name:measured([i for i in ids if (radii[i]<=report['outer_radius']+1e-5)==inside])
                for name,inside in (('inside',True),('outside',False))}
    withheld=probe_ids('withheld_direction')
    return dict(distance_scale=DISTANCE_SCALE,raw_calibration=calibration,
        true_distances=radii.tolist(),training=measured(train),
        directional=regions(probe_ids()),withheld_direction=regions(withheld),
        fixed_near_withheld_direction=measured([i for i in withheld if radii[i]<=6+1e-5]))
