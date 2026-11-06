"""Fixed-cohort radial and offset measurements, split by training radius."""

import numpy as np

from src.mylonas_ijepa_wm.curriculum_local_geometry import coverage,curve_metrics


def summarize(design,raw,adapted,frame_hashes,offered_ids,actual_ids,radius,*,include_goal_reference=True):
    points=np.asarray(design['points'])
    all_radii=np.linalg.norm(points-np.asarray(design['goal']),axis=1)
    reference={design['goal_id']} if include_goal_reference else set()
    offered_ids=sorted(set(offered_ids)|reference)
    actual_ids=sorted(set(actual_ids)|reference)
    exposure=coverage(points,points[actual_ids],points[offered_ids])
    offered_hashes={frame_hashes[i] for i in offered_ids}
    actual_hashes={frame_hashes[i] for i in actual_ids}
    unseen=np.array([not (exposure['fitted_position'][i] or frame_hashes[i] in actual_hashes)
                     for i in range(len(points))])
    heldout=np.array([not (exposure['replay_position'][i] or frame_hashes[i] in offered_hashes)
                      for i in range(len(points))])
    methods=dict(raw=np.asarray(raw),adapted=np.asarray(adapted))
    strata=('inside','outside','crossing')
    generalization={kind:{region:{method:dict(comparisons=0,up_steps=0,tie_steps=0,
        heldout_comparisons=0,heldout_up_steps=0) for method in methods} for region in strata}
        for kind in ('trained_direction','withheld_direction')}
    curves=[]
    for curve in design['curves']:
        ids=np.array(curve['point_ids'])
        r=all_radii[ids]
        regions=np.where((r[:-1]<=radius+1e-5)&(r[1:]<=radius+1e-5),'inside',
            np.where((r[:-1]>radius+1e-5)&(r[1:]>radius+1e-5),'outside','crossing'))
        row=dict(curve,heldout=heldout[ids].tolist())
        for method,values in methods.items():
            costs=values[ids];row[method+'_costs']=costs.tolist()
            for region in strata:
                mask=regions==region
                if not mask.any():continue
                endpoints=np.unique(np.r_[np.where(mask)[0],np.where(mask)[0]+1])
                tolerance=curve_metrics(costs[endpoints])['tolerance']
                up=np.diff(costs)>tolerance;ties=np.abs(np.diff(costs))<=tolerance
                fresh=heldout[ids[:-1]]&heldout[ids[1:]]&mask
                m=generalization[curve['kind']][region][method]
                m['comparisons']+=int(mask.sum());m['up_steps']+=int((up&mask).sum())
                m['tie_steps']+=int((ties&mask).sum());m['heldout_comparisons']+=int(fresh.sum())
                m['heldout_up_steps']+=int((up&fresh).sum())
        curves.append(row)
    for regions in generalization.values():
        for methods_block in regions.values():
            for m in methods_block.values():
                m['up_fraction']=m['up_steps']/m['comparisons'] if m['comparisons'] else None
                m['heldout_up_fraction']=m['heldout_up_steps']/m['heldout_comparisons'] if m['heldout_comparisons'] else None
    profiles=[]
    matched=set(design['matched_family_ids'])
    for family in design['families']:
        for p in family['profiles']:
            ids=np.array(p['point_ids'])
            r=all_radii[ids]
            region='inside' if r.max()<=radius+1e-5 else 'outside' if r.min()>radius+1e-5 else 'crossing'
            row=dict(family_id=family['family_id'],angle=family['angle'],offset=p['offset'],
                point_ids=p['point_ids'],region=region,matched=family['family_id'] in matched,
                heldout=heldout[ids].tolist(),unseen_to_actual_fit=unseen[ids].tolist())
            for method,values in methods.items():
                costs=values[ids];metrics=curve_metrics(costs)
                mask=heldout[ids[:-1]]&heldout[ids[1:]]
                metrics['heldout_comparisons']=int(mask.sum())
                metrics['heldout_up_steps']=int(((np.diff(costs)>metrics['tolerance'])&mask).sum())
                row[method]=metrics;row[method+'_costs']=costs.tolist()
            profiles.append(row)
    def offset_rows(cohort):
        offsets=[]
        for offset in (0.,.5,1.,2.):
            selected=[p for p in profiles if p['family_id'] in cohort and abs(p['offset'])==offset]
            row=dict(offset=offset,families=len(cohort),profiles=len(selected),
                all_profiles=sum(abs(p['offset'])==offset for p in profiles))
            for method in methods:
                # Two signs are averaged within family before averaging families.
                averages=[]
                for family in sorted(cohort):
                    items=[p[method] for p in selected if p['family_id']==family]
                    averages.append({key:float(np.mean([m[key] for m in items]))
                        for key in ('up_fraction','endpoint_correct','informative_monotone','flat')})
                row[method]={key:float(np.mean([m[key] for m in averages])) if averages else None
                             for key in ('up_fraction','endpoint_correct','informative_monotone','flat')}
            offsets.append(row)
        return offsets
    offered=set(offered_ids)
    fitted_centers={p['family_id'] for p in profiles if p['matched'] and p['offset']==0
                    and p['point_ids'][0] in offered and p['point_ids'][-1] in offered}
    return dict(generalization=generalization,offset_summary=offset_rows(matched),
        offset_fitted_centers=offset_rows(fitted_centers),curves=curves,profiles=profiles,
        matched_family_ids=design['matched_family_ids'],exposure=dict(**exposure,
            heldout_from_offered_training=heldout.tolist(),unseen_to_actual_fit=unseen.tolist()))
