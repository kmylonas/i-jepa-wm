"""Manual reference/far/near comparisons, ranking-only fitting and metrics."""
from copy import deepcopy

import numpy as np
import torch
from torch.nn import functional as F

from src.mylonas_ijepa_wm.cost_adapter import ResidualTokenAdapter
from src.mylonas_ijepa_wm.radial_goal_transfer import measure
from src.mylonas_ijepa_wm.radial_multigoal import FITTING


GAP=.5
MARGIN=.2


def triples(positions,reference_ids):
    xy=np.asarray(positions,dtype=float);refs=np.asarray(reference_ids)
    if (xy.ndim!=2 or xy.shape[1]!=2 or not np.isfinite(xy).all()
            or refs.ndim!=1 or not len(refs) or not np.issubdtype(refs.dtype,np.integer)
            or (refs<0).any() or (refs>=len(xy)).any() or len(np.unique(refs))!=len(refs)):
        raise ValueError('invalid positions or triple references')
    parts=[]
    for g in refs:
        distances=np.linalg.norm(xy-xy[g],axis=1)
        eligible=(distances>0)&(distances<=34+1e-5);eligible[g]=False
        far,near=np.where(eligible[:,None]&eligible[None,:]&(distances[:,None]-distances[None,:]>=GAP))
        if len(far):parts.append(np.column_stack([np.full(len(far),g),far,near]))
    if not parts:raise ValueError('no eligible ranking triples')
    return torch.as_tensor(np.concatenate(parts),dtype=torch.int32)


def triplet_costs(adapter,tokens,population,*,device='cpu',batch_size=16):
    parts=[[],[]]
    for selection in torch.as_tensor(population,dtype=torch.long).split(batch_size):
        g,f,n=[tokens[selection[:,j]].to(device,dtype=torch.float32) for j in range(3)]
        if adapter is not None:g,f,n=adapter(g),adapter(f),adapter(n)
        parts[0].append((f-g).square().mean(dim=(1,2)).cpu())
        parts[1].append((n-g).square().mean(dim=(1,2)).cpu())
    return tuple(torch.cat(p) for p in parts)


def fit(tokens,population,scale,*,seed=0,updates=2000,batch_size=32,bottleneck=64,lr=1e-4,
        device='cpu',progress=None,initial_adapter=None):
    tokens=tokens.detach().cpu()
    if tokens.dtype!=torch.float16:tokens=tokens.float()
    values=np.asarray(population)
    if (values.ndim!=2 or values.shape[1:]!=(3,) or not len(values)
            or not np.issubdtype(values.dtype,np.integer) or (values<0).any()
            or (values>=len(tokens)).any() or any((values[:,a]==values[:,b]).any() for a,b in ((0,1),(0,2),(1,2)))
            or not np.isfinite(scale) or scale<=0):raise ValueError('invalid triples or raw scale')
    if (tokens.ndim!=3 or not all(torch.isfinite(chunk).all() for chunk in tokens.split(32)) or updates<1 or batch_size<1
            or bottleneck<1 or not np.isfinite(lr) or lr<=0):raise ValueError('invalid ranking inputs or schedule')
    population=torch.as_tensor(values,dtype=torch.long)
    if initial_adapter is None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);adapter=ResidualTokenAdapter(tokens.shape[-1],bottleneck).to(device)
    else:
        if (initial_adapter.dim,initial_adapter.bottleneck)!=(tokens.shape[-1],bottleneck):
            raise ValueError('initial adapter shape differs from ranking inputs')
        adapter=deepcopy(initial_adapter).to(device).train().requires_grad_(True)
    reference_ids=torch.unique(population[:,0]).tolist()
    buckets=[torch.nonzero(population[:,0]==g).flatten().numpy() for g in reference_ids]
    probe_ids=np.sort(np.random.default_rng(1729).choice(len(population),min(256,len(population)),replace=False))
    @torch.no_grad()
    def probe():
        far,near=triplet_costs(adapter,tokens,population[probe_ids],device=device)
        delta=far-near
        return dict(rank_loss=float(F.softplus(MARGIN-delta/scale).mean()),
            rank_accuracy=float((delta>1e-7).float().mean()),mean_far_cost=float(far.mean()),mean_near_cost=float(near.mean()))
    before=probe();history=[dict(update=0,**before)]
    optimizer=torch.optim.Adam(adapter.parameters(),lr=lr);rng=np.random.default_rng(seed)
    endpoints=torch.zeros(len(tokens),dtype=torch.long);refs=torch.zeros_like(endpoints)
    farther=torch.zeros_like(endpoints);nearer=torch.zeros_like(endpoints);sampled=[]
    for update in range(1,updates+1):
        groups=rng.integers(len(buckets),size=batch_size)
        ids=torch.as_tensor([int(rng.choice(buckets[g])) for g in groups])
        selection=population[ids];sampled.extend(ids.tolist())
        endpoints+=torch.bincount(selection.flatten(),minlength=len(tokens))
        refs+=torch.bincount(selection[:,0],minlength=len(tokens))
        farther+=torch.bincount(selection[:,1],minlength=len(tokens))
        nearer+=torch.bincount(selection[:,2],minlength=len(tokens))
        far,near=triplet_costs(adapter,tokens,selection,device=device,batch_size=batch_size)
        loss=F.softplus(MARGIN-(far-near)/scale).mean()
        if not torch.isfinite(loss):raise ValueError('nonfinite ranking loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        row=dict(update=update,minibatch_loss=float(loss.detach()))
        if update==1 or update%100==0 or update==updates:
            row.update(probe())
            if progress:progress(row)
        history.append(row)
    after=probe();adapter.eval()
    return adapter,dict(seed=seed,updates=updates,batch_size=batch_size,bottleneck=bottleneck,lr=lr,
        initialization='identity' if initial_adapter is None else 'previous_adapter',anchor_weight=0,scale=float(scale),margin=MARGIN,min_physical_gap=GAP,
        objective='ranking',loss='mean(softplus(0.2-(far_cost-near_cost)/scale))',
        sampling='uniform_reference_then_uniform_valid_comparison',eligible_triples=len(population),
        reference_ids=reference_ids,reference_counts=refs.tolist(),endpoint_counts=endpoints.tolist(),
        far_counts=farther.tolist(),near_counts=nearer.tolist(),
        sampled_triplet_ids=sampled,probe_triplet_ids=probe_ids.tolist(),
        probe_before=before,probe_after=after,history=history)


def ordinal_rows(points,goals,raw,costs,heldout,edges):
    # Reuse the exact previous cohort and adjacent-edge convention, retaining
    # ordinal fields only. Ranking costs are never interpreted as distances.
    measured=measure(points,goals,raw,costs,34,1,heldout,edges);rows=[]
    for row in measured:
        common=np.asarray(row['common_mask']);mask=common&np.asarray(heldout,dtype=bool)
        ids=np.flatnonzero(mask);truth=np.asarray(row['true_distances'])
        far,near=np.where(truth[ids,None]-truth[None,ids]>=GAP)
        for key in ('directional','heldout_directional'):
            direction=row[key];count=direction['comparisons']
            for prefix in ('raw','adapted'):
                decreasing=count-direction[prefix+'_up_steps']-direction[prefix+'_ties']
                direction[prefix+'_decreasing_steps']=decreasing
                direction[prefix+'_decreasing_fraction']=decreasing/count if count else None
                direction[prefix+'_tie_fraction']=direction[prefix+'_ties']/count if count else None
        def ranking(values):
            values=np.asarray(values)
            tolerance=max(1e-7,.001*float(np.ptp(values[common]))) if common.any() else 1e-7
            delta=values[ids[far]]-values[ids[near]];n=len(delta)
            counts=dict(correct=int((delta>tolerance).sum()),reversed=int((delta<-tolerance).sum()),ties=int((np.abs(delta)<=tolerance).sum()))
            return dict(comparisons=n,tolerance=tolerance,**counts,
                correct_fraction=counts['correct']/n if n else None,
                reversed_fraction=counts['reversed']/n if n else None,tie_fraction=counts['ties']/n if n else None)
        keep=('goal','true_distances','raw_costs','adapted_costs','common_mask','directional','heldout_directional')
        rows.append(dict(**{k:row[k] for k in keep},common_count=int(common.sum()),heldout_count=int(mask.sum()),
            raw_rank=ranking(row['raw_costs']),adapted_rank=ranking(row['adapted_costs'])))
    return rows
