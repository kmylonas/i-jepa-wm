"""Spatial coordinate readout over frozen image tokens, with physical metrics."""
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.mylonas_ijepa_wm.curriculum_local_geometry import coverage


CENTER=(50.,42.)
SCALE=34.
FITTING=dict(updates=500,batch_size=32,projection=32,hidden=64,lr=1e-4)


class PositionReadout(nn.Module):
    def __init__(self,patches,dim,projection=32,hidden=64):
        super().__init__()
        if min(patches,dim,projection,hidden)<1:raise ValueError('positive position-model dimensions required')
        self.shape=dict(patches=patches,dim=dim,projection=projection,hidden=hidden)
        self.project=nn.Linear(dim,projection)
        self.hidden=nn.Linear(patches*projection,hidden)
        self.output=nn.Linear(hidden,2)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,tokens):
        if tokens.ndim!=3 or tuple(tokens.shape[1:])!=(self.shape['patches'],self.shape['dim']):
            raise ValueError('position readout needs aligned spatial tokens')
        spatial=F.gelu(self.project(tokens.float())).flatten(1)
        return self.output(F.gelu(self.hidden(spatial)))

    def predict(self,tokens):
        return self(tokens)*SCALE+tokens.new_tensor(CENTER)


def fit(tokens,positions,*,seed,updates=500,batch_size=32,projection=32,hidden=64,lr=1e-4,
        initial_model=None,device='cpu',progress=None):
    tokens=tokens.detach().cpu().float();positions=torch.as_tensor(positions).detach().cpu().float()
    if (tokens.ndim!=3 or not len(tokens) or positions.shape!=(len(tokens),2)
            or not torch.isfinite(tokens).all() or not torch.isfinite(positions).all()
            or updates<1 or batch_size<1 or not np.isfinite(lr) or lr<=0):
        raise ValueError('invalid position training inputs or schedule')
    shape=dict(patches=tokens.shape[1],dim=tokens.shape[2],projection=projection,hidden=hidden)
    if initial_model is None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);model=PositionReadout(**shape).to(device)
    else:
        if initial_model.shape!=shape:raise ValueError('incoming position model shape differs')
        model=deepcopy(initial_model).to(device).requires_grad_(True).train()
    labels=(positions-torch.tensor(CENTER))/SCALE
    @torch.no_grad()
    def full_metrics():
        predictions=torch.cat([model(chunk.to(device)).cpu() for chunk in tokens.split(16)])
        residual=predictions-labels;errors=residual.norm(dim=1)*SCALE
        return dict(full_position_loss=float(residual.square().mean()),
            mean_position_error=float(errors.mean()),position_rmse=float(errors.square().mean().sqrt()))
    before=full_metrics();history=[dict(update=0,**before)]
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    rng=np.random.default_rng(seed);counts=torch.zeros(len(tokens),dtype=torch.int64)
    for update in range(updates):
        index=torch.as_tensor(rng.integers(len(tokens),size=batch_size))
        counts+=torch.bincount(index,minlength=len(tokens))
        loss=(model(tokens[index].to(device))-labels[index].to(device)).square().mean()
        if not torch.isfinite(loss):raise ValueError('nonfinite position loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        row=dict(update=update+1,position_loss=float(loss.detach()))
        if update==0 or (update+1)%50==0 or update+1==updates:row.update(full_metrics())
        history.append(row)
        if progress and (update==0 or (update+1)%50==0):progress(row)
    after=full_metrics();model.eval()
    return model,dict(seed=seed,updates=updates,batch_size=batch_size,projection=projection,hidden=hidden,lr=lr,
        position_center=list(CENTER),position_scale=SCALE,model_shape=shape,
        loss='mean((normalized_xy-(true_xy-center)/34)^2)',sampling='uniform_positions_with_replacement',
        initialization='fresh' if initial_model is None else 'previous_checkpoint',optimizer_reset=True,
        sample_counts=counts.tolist(),history=history,full_position_loss_before=before['full_position_loss'],
        full_position_loss_after=after['full_position_loss'],position_error_before=before['mean_position_error'],
        position_error_after=after['mean_position_error'],position_rmse_before=before['position_rmse'],
        position_rmse_after=after['position_rmse'])


def summarize_positions(design,predictions,frame_hashes,offered_ids,actual_ids,radius):
    truth=np.asarray(design['points']);predictions=np.asarray(predictions)
    if predictions.shape!=truth.shape or not np.isfinite(predictions).all():
        raise ValueError('finite xy prediction for every position required')
    goal=np.asarray(design['goal']);radii=np.linalg.norm(truth-goal,axis=1)
    train=sorted(set(offered_ids));actual=sorted(set(actual_ids))
    exposure=coverage(truth,truth[actual],truth[train])
    offered_hashes={frame_hashes[i] for i in train};actual_hashes={frame_hashes[i] for i in actual}
    heldout=np.array([not (exposure['replay_position'][i] or frame_hashes[i] in offered_hashes)
        for i in range(len(truth))])
    exposure['heldout_from_offered_training']=heldout.tolist()
    exposure['unseen_to_actual_fit']=[not (exposure['fitted_position'][i] or frame_hashes[i] in actual_hashes)
        for i in range(len(truth))]
    baseline=truth[train].mean(axis=0)
    def measured(ids):
        ids=np.asarray(sorted(set(ids)),dtype=int);row=dict(count=len(ids))
        for method,xy in (('model',predictions),('constant',np.broadcast_to(baseline,truth.shape))):
            residual=xy[ids]-truth[ids];error=np.linalg.norm(residual,axis=1)
            derror=np.abs(np.linalg.norm(xy[ids]-goal,axis=1)-radii[ids])
            row[method]=dict(mean_position_error=float(error.mean()) if len(ids) else None,
                position_rmse=float(np.sqrt(np.mean(error**2))) if len(ids) else None,
                xy_mae=np.abs(residual).mean(axis=0).tolist() if len(ids) else None,
                xy_bias=residual.mean(axis=0).tolist() if len(ids) else None,
                distance_mae=float(derror.mean()) if len(ids) else None)
        return row
    def probe_ids(kind=None):
        return sorted({i for c in design['curves'] if kind is None or c['kind']==kind
            for i in c['point_ids'] if heldout[i]})
    def regions(ids):
        return {name:measured([i for i in ids if (radii[i]<=radius+1e-5)==inside])
            for name,inside in (('inside',True),('outside',False))}
    all_ids=probe_ids();withheld=probe_ids('withheld_direction')
    shells=[]
    for lower,upper in ((0,2),(2,6),(6,12),(12,None)):
        ids=[i for i in all_ids if radii[i]>radius+lower+1e-5
            and (upper is None or radii[i]<=radius+upper+1e-5)]
        shells.append(dict(lower=lower,upper=upper,metrics=measured(ids)))
    return dict(exposure=exposure,constant_position=baseline.tolist(),training=measured(train),
        directional=regions(all_ids),withheld_direction=regions(withheld),outside_shells=shells,
        fixed_near_withheld_direction=measured([i for i in withheld if radii[i]<=6+1e-5]),
        position_errors=np.linalg.norm(predictions-truth,axis=1).tolist(),
        predicted_goal_distances=np.linalg.norm(predictions-goal,axis=1).tolist(),true_distances=radii.tolist())
