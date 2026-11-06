"""Pairwise distance fitting with explicitly reserved reference images."""
import itertools

import numpy as np
import torch

from src.mylonas_ijepa_wm.cost_adapter import ResidualTokenAdapter


SCALE=34.
FITTING=dict(updates=2000,batch_size=32,bottleneck=64,lr=1e-4,seed=0)


def make_split(points,candidate_ids,goal_ids,frame_hashes):
    points=np.asarray(points,dtype=float)
    if (points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all()
            or len(frame_hashes)!=len(points) or len(goal_ids)<2
            or len(set(goal_ids))!=len(goal_ids) or len(set(candidate_ids))!=len(candidate_ids)
            or any(i<0 or i>=len(points) for i in [*candidate_ids,*goal_ids])):
        raise ValueError('invalid point/reference population')
    reserved=points[goal_ids[1:]];reserved_hashes={frame_hashes[i] for i in goal_ids[1:]}
    train=[];seen=set();removed=[]
    for i in candidate_ids:
        h=frame_hashes[i]
        if h in reserved_hashes or h in seen or np.linalg.norm(reserved-points[i],axis=1).min()<1e-5:
            removed.append(i)
        else:train.append(i);seen.add(h)
    if goal_ids[0] not in train:raise ValueError('reserved image aliases original training reference')
    pairs=[(a,b) for a,b in itertools.combinations(sorted(train),2)
           if 0<np.linalg.norm(points[a]-points[b])<=SCALE+1e-5]
    single=[(i,goal_ids[0]) for i in train if i!=goal_ids[0]
            and 0<np.linalg.norm(points[i]-points[goal_ids[0]])<=SCALE+1e-5]
    if not single or not pairs:raise ValueError('empty training pair population')
    exposed=points[train+list(goal_ids)];hashes={frame_hashes[i] for i in train+list(goal_ids)}
    heldout=[bool(h not in hashes and np.linalg.norm(exposed-p,axis=1).min()>1e-5)
             for p,h in zip(points,frame_hashes)]
    return dict(train_ids=train,removed_candidate_ids=removed,
        single_pairs=[list(p) for p in single],multi_pairs=[list(p) for p in pairs],heldout_mask=heldout)


def pair_costs(adapter,tokens,pairs,*,device='cpu',batch_size=16):
    """Differentiable costs, chunked so a full pair population fits in memory."""
    pairs=torch.as_tensor(pairs,dtype=torch.long);result=[]
    for chosen in pairs.split(batch_size):
        a,b=tokens[chosen[:,0]].to(device),tokens[chosen[:,1]].to(device)
        if adapter is not None:a,b=adapter(a),adapter(b)
        result.append((a-b).square().mean(dim=(1,2)).cpu())
    return torch.cat(result)


def fit(tokens,positions,pairs,*,seed=0,updates=2000,batch_size=32,bottleneck=64,lr=1e-4,
        device='cpu',progress=None):
    tokens=tokens.detach().cpu().float();positions=torch.as_tensor(positions).float().cpu()
    values=np.asarray(pairs)
    if (values.ndim!=2 or values.shape[1:]!=(2,) or not len(values)
            or not np.issubdtype(values.dtype,np.integer) or (values<0).any()
            or (values>=len(tokens)).any() or (values[:,0]==values[:,1]).any()
            or len(np.unique(np.sort(values,axis=1),axis=0))!=len(values)):
        raise ValueError('invalid or duplicate pair population')
    pairs=torch.as_tensor(values,dtype=torch.long)
    if (tokens.ndim!=3 or positions.shape!=(len(tokens),2) or not torch.isfinite(tokens).all()
            or not torch.isfinite(positions).all() or updates<1 or batch_size<1
            or bottleneck<1 or not np.isfinite(lr) or lr<=0):
        raise ValueError('invalid fitting inputs or schedule')
    labels=(positions[pairs[:,0]]-positions[pairs[:,1]]).norm(dim=1)/SCALE
    if (labels<=0).any() or (labels>1+1e-5).any():raise ValueError('pair distance outside (0,34]')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        adapter=ResidualTokenAdapter(tokens.shape[-1],bottleneck).to(device)
    rng=np.random.default_rng(seed)
    probe_ids=np.sort(np.random.default_rng(1729).choice(len(pairs),min(256,len(pairs)),replace=False))
    @torch.no_grad()
    def probe():
        errors=pair_costs(adapter,tokens,pairs[probe_ids],device=device)-labels[probe_ids]
        return dict(regression_loss=float(errors.square().mean()),
            distance_mae=float(errors.abs().mean()*SCALE),distance_rmse=float(errors.square().mean().sqrt()*SCALE))
    before=probe();history=[dict(update=0,**before)]
    optimizer=torch.optim.Adam(adapter.parameters(),lr=lr)
    counts=torch.zeros(len(pairs),dtype=torch.long);endpoints=torch.zeros(len(tokens),dtype=torch.long)
    for update in range(1,updates+1):
        chosen=torch.as_tensor(rng.integers(len(pairs),size=batch_size));selection=pairs[chosen]
        counts+=torch.bincount(chosen,minlength=len(pairs))
        endpoints+=torch.bincount(selection.flatten(),minlength=len(tokens))
        loss=(pair_costs(adapter,tokens,selection,device=device,batch_size=batch_size)-labels[chosen]).square().mean()
        if not torch.isfinite(loss):raise ValueError('nonfinite pair regression loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        row=dict(update=update,minibatch_loss=float(loss.detach()))
        if update==1 or update%100==0 or update==updates:
            row.update(probe())
            if progress:progress(row)
        history.append(row)
    after=probe();adapter.eval()
    return adapter,dict(seed=seed,updates=updates,batch_size=batch_size,lr=lr,bottleneck=bottleneck,
        initialization='identity',anchor_weight=0,distance_scale=SCALE,
        loss='mean((adapted_pair_mse-euclidean_distance/34)^2)',
        sampling='uniform_pairs_with_replacement',pair_sample_counts=counts.tolist(),
        endpoint_counts=endpoints.tolist(),probe_pair_ids=probe_ids.tolist(),
        probe_before=before,probe_after=after,history=history)
