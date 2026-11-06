"""Distance-calibration diagnostics in physical simulation units."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_distance(output,design,raw,report):
    distance=report['distance'];truth=np.asarray(distance['true_distances'])
    prediction=np.asarray(report['adapted_costs'])*distance['distance_scale']
    baseline=np.asarray(raw)*distance['raw_calibration']
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,kind in zip(axes[0],('trained_direction','withheld_direction')):
        curves=[c for c in design['curves'] if c['kind']==kind]
        for i,curve in enumerate(curves):
            ids=curve['point_ids']
            ax.plot(truth[ids],baseline[ids],c='gray',alpha=.3,lw=.8,label='Calibrated raw' if i==0 else None)
            ax.plot(truth[ids],prediction[ids],c='#0072b2',alpha=.35,lw=.9,label='Adapted' if i==0 else None)
        ax.plot([0,34],[0,34],'k--',label='True distance')
        ax.axvline(report['outer_radius'],c='orange',ls='--',label='Training limit')
        ax.set(title=kind.replace('_',' '),xlabel='True distance',ylabel='Predicted distance')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    ids=sorted({i for c in design['curves'] for i in c['point_ids']
        if report['exposure']['heldout_from_offered_training'][i]})
    ids=np.asarray(ids,dtype=int);inside=truth[ids]<=report['outer_radius']+1e-5
    ax=axes[1,0]
    for mask,name,color in ((inside,'Inside','#0072b2'),(~inside,'Outside','#d55e00')):
        ax.scatter(truth[ids[mask]],prediction[ids[mask]],s=8,alpha=.4,label=name,c=color)
    ax.plot([0,34],[0,34],'k--');ax.set(title='Genuinely heldout directional positions',
        xlabel='True distance',ylabel='Adapted predicted distance');ax.legend();ax.grid(alpha=.2)
    ax=axes[1,1]
    groups=[distance['training'],distance['withheld_direction']['inside'],
        distance['withheld_direction']['outside'],distance['fixed_near_withheld_direction']]
    labels=['Training','Unseen rays\ninside','Unseen rays\noutside','Fixed unseen\nR≤6']
    for offset,method,color,label in ((-.18,'raw_calibrated','gray','Calibrated raw'),
                                    (.18,'adapted','#0072b2','Adapted')):
        y=[g[method]['mae'] if g['count'] else np.nan for g in groups]
        ax.bar(np.arange(4)+offset,y,width=.36,color=color,label=label)
    ax.set(xticks=np.arange(4),xticklabels=[f'{label}\nn={g["count"]}' for label,g in zip(labels,groups)],
        ylabel='Mean absolute distance error',title='Empty groups omitted (N/A)')
    ax.grid(axis='y',alpha=.2);ax.legend()
    fig.suptitle(f"Stage {report['stage']}: adapted MSE ×34 estimates Euclidean distance\n"
        'Raw scalar fitted only on offered training positions; identical probe cohorts for both methods.')
    folder=Path(output)/f"stages/s{report['stage']}/figures";folder.mkdir(parents=True,exist_ok=True)
    paths=[]
    for suffix in ('png','pdf'):
        path=folder/f'distance_calibration.{suffix}';fig.savefig(path,dpi=140,bbox_inches='tight');paths.append(str(path))
    plt.close(fig)
    return paths
