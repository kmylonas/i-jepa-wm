"""Ranking-only metric fitting on explicitly labelled radial positions."""

from copy import deepcopy

import numpy as np
import torch
from torch.nn import functional as F

from src.mylonas_ijepa_wm.cost_adapter import ResidualTokenAdapter,token_goal_cost


def training_pairs(radii):
    radii=torch.as_tensor(radii).cpu().float()
    if radii.ndim!=1 or not torch.isfinite(radii).all() or (radii<0).any():
        raise ValueError('finite nonnegative radius vector required')
    pairs=torch.nonzero(radii[:,None]-radii[None,:]>=.5)
    if not len(pairs):raise ValueError('no distinct-radius training pairs')
    return pairs


def fit(tokens,radii,goal,*,seed,updates=500,batch_size=16,lr=1e-4,bottleneck=64,
        initial_adapter=None,device='cpu',progress=None):
    tokens,goal=tokens.detach().cpu().float(),goal.detach().cpu().float()
    if (tokens.ndim!=3 or tokens.shape[1:]!=goal.shape or len(tokens)!=len(radii)
            or not torch.isfinite(tokens).all() or not torch.isfinite(goal).all()
            or updates<1 or batch_size<1 or not np.isfinite(lr) or lr<=0):
        raise ValueError('invalid radial fitting inputs or schedule')
    pairs=training_pairs(radii)
    if initial_adapter is None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            adapter=ResidualTokenAdapter(goal.shape[-1],bottleneck).to(device)
    else:
        if initial_adapter.dim!=goal.shape[-1] or initial_adapter.bottleneck!=bottleneck:
            raise ValueError('incoming adapter dimensions differ')
        adapter=deepcopy(initial_adapter).to(device).requires_grad_(True).train()
    raw=(tokens-goal).square().mean(dim=(1,2))
    positive=raw[raw>0]
    if not len(positive):raise ValueError('zero radial raw-cost scale')
    scale=float(positive.median())
    target=goal.to(device)
    @torch.no_grad()
    def full_metrics():
        costs=torch.cat([token_goal_cost(adapter,chunk.to(device),target).cpu() for chunk in tokens.split(16)])
        differences=costs[pairs[:,0]]-costs[pairs[:,1]]
        return float((differences>1e-7).float().mean()),float(F.softplus(.2-differences/scale).mean())
    before,loss_before=full_metrics()
    optimizer=torch.optim.Adam(adapter.parameters(),lr=lr)
    rng=np.random.default_rng(seed)
    history=[];counts=torch.zeros(len(tokens),dtype=torch.int64)
    for update in range(updates):
        chosen=pairs[rng.integers(len(pairs),size=batch_size)]
        index=torch.cat([chosen[:,0],chosen[:,1]])
        counts+=torch.bincount(index,minlength=len(tokens))
        costs=token_goal_cost(adapter,tokens[index].to(device),target)
        loss=F.softplus(.2-(costs[:batch_size]-costs[batch_size:])/scale).mean()
        if not torch.isfinite(loss):raise ValueError('nonfinite radial ranking loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        row=dict(update=update+1,rank_loss=float(loss.detach()))
        if update==0 or (update+1)%50==0 or update+1==updates:
            row['full_rank_accuracy'],row['full_rank_loss']=full_metrics()
        history.append(row)
        if progress and (update==0 or (update+1)%50==0):progress(row)
    adapter.eval()
    return adapter,dict(seed=seed,updates=updates,batch_size=batch_size,lr=lr,bottleneck=bottleneck,
        anchor_weight=0,loss='softplus(0.2-(far-near)/scale)',scale=scale,eligible_pairs=len(pairs),
        initialization='identity' if initial_adapter is None else 'previous_checkpoint',optimizer_reset=True,
        sample_counts=counts.tolist(),rank_accuracy_before=before,rank_accuracy_after=full_metrics()[0],
        full_rank_loss_before=loss_before,history=history)
