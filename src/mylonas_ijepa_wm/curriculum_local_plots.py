"""Plots of saved local probes: measured samples, costs and exposure strata."""

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.mylonas_ijepa_wm.curriculum_multiband_protocol import read


def render(output):
    output = Path(output)
    report, probes = read(output/'report.json'),read(output/'probes.json')
    folder = output/'figures'
    folder.mkdir(exist_ok=True)
    exported = []
    def save(fig,name):
        for suffix in ('png','pdf'):
            path = folder/f'{name}.{suffix}'
            fig.savefig(path,dpi=150,bbox_inches='tight')
            exported.append(str(path))
        plt.close(fig)
    grid = report['grid']
    points = np.asarray(grid['points'])
    xs,ys = np.unique(points[:,0]),np.unique(points[:,1])
    fitting = np.asarray(report['fitting_positions'])
    fig,axes = plt.subplots(1,2,figsize=(10,8),constrained_layout=True)
    for ax,method in zip(axes,('raw','adapted')):
        costs = np.asarray(grid[method])
        graphic = ax.pcolormesh(xs,ys,costs.reshape(len(ys),len(xs)),shading='nearest',
            vmin=0,vmax=max(float(costs.max()),1e-7)*1.03,cmap='viridis')
        ax.scatter(fitting[:,0],fitting[:,1],s=2,c='white',alpha=.25)
        ax.axvline(grid['reset_safe_x_max'],c='white',ls=':',lw=1,label='Reset-safe edge')
        ax.axvline(grid['x_max'],c='orange',lw=1,label='Physical right border')
        for y in (8,14,22,30,36):ax.axhline(y,c='white',ls='--',lw=.5)
        ax.scatter(50,42,marker='*',s=110,c='orange',edgecolors='black')
        ax.set(title=method.capitalize()+' MSE',xlabel='x',ylabel='y',aspect='equal',xlim=(39,grid['x_max']+.05),ylim=(7,51))
        fig.colorbar(graphic,ax=ax,shrink=.75,label='MSE (independent full-range scale)')
    axes[1].legend(fontsize=8,loc='upper left')
    fig.suptitle('Clean no-anchor v0: full right domain\nWhite points: positions passed to the adapter during fitting')
    save(fig,'full_right_heatmaps')

    summaries = report['offset_summary']
    fig,axes = plt.subplots(1,3,figsize=(13,3.7),constrained_layout=True)
    for method,color in (('raw','black'),('adapted','#0072b2')):
        offsets = [r['offset'] for r in summaries]
        for ax,key,title in zip(axes,('endpoint_accuracy','up_step_fraction','unseen_up_fraction'),
                ('Endpoint ranking accuracy','Steps that increase cost toward B','Same error on unseen comparisons only')):
            ax.plot(offsets,[r[method][key] if r[method][key] is not None else np.nan for r in summaries],
                marker='o',c=color,label=method)
            ax.set(title=title,xlabel='Absolute sideways offset (units)',ylim=(-.03,1.03))
            ax.grid(alpha=.2)
    axes[0].legend()
    fig.suptitle('Offsets outside valid/decreasing-distance geometry are omitted; counts printed below')
    save(fig,'offset_generalization')

    centers = [row for row in probes if row['offset']==0]
    worst = sorted(centers,key=lambda row:row['adapted']['up_fraction'],reverse=True)
    chosen = list(dict.fromkeys([row['pair_id'] for row in centers[:3]+worst[:3]]))
    groups = defaultdict(list)
    for row in probes:groups[row['pair_id']].append(row)
    fig,axes = plt.subplots(len(chosen),2,figsize=(12,3*len(chosen)),squeeze=False,constrained_layout=True)
    for row_axes,pair_id in zip(axes,chosen):
        family = sorted(groups[pair_id],key=lambda r:r['offset'])
        center = next(r for r in family if r['offset']==0)
        for ax,method in zip(row_axes,('raw','adapted')):
            for row in family:
                costs = row[method+'_costs']
                ax.plot(np.linspace(0,1,len(costs)),costs,lw=2 if row['offset']==0 else 1,
                    alpha=1 if row['offset']==0 else .55,label=f"offset {row['offset']:g}")
                if row['offset']==0:
                    unseen = np.asarray(row['coverage']['unseen_relative_to_fitting_inputs'])
                    ax.scatter(np.linspace(0,1,len(costs))[unseen],np.asarray(costs)[unseen],
                        s=25,c='black',marker='x',zorder=5)
            ax.set(title=f"Pair {pair_id}: {method}, center uphill {center[method]['up_fraction']:.0%}",
                xlabel='A → B (coordinate interpolation)',ylabel='MSE')
            ax.grid(alpha=.2)
    axes[0,1].legend(ncol=2,fontsize=7)
    fig.suptitle('First three geometry-selected pairs + three worst center curves (deduplicated)\n'
        'Thick = original segment; thin = parallel segments; black crosses = unseen center samples')
    save(fig,'pair_curves')

    fig,axes = plt.subplots(len(chosen),1,figsize=(9,2.6*len(chosen)),squeeze=False,constrained_layout=True)
    for (ax,),pair_id in zip(axes,chosen):
        for method,color in (('raw','black'),('adapted','#0072b2')):
            row = next(r for r in groups[pair_id] if r['offset']==0)
            costs = np.asarray(row[method+'_costs'])
            scale = max(abs(costs[0]-costs[-1]),np.ptp(costs),1e-7)
            ax.plot(np.linspace(0,1,len(costs)),(costs-costs[0])/scale,marker='.',c=color,label=method)
        ax.set(title=f'Pair {pair_id}',xlabel='A → B',ylabel='Cost change / contrast or range')
        ax.grid(alpha=.2)
    axes[0,0].legend()
    fig.suptitle('Pair-relative center costs: falling is the desired direction\n'
        'Scaling preserves reversals; flat curves stay flat')
    save(fig,'relative_center_curves')

    zoom_ids = list(dict.fromkeys([centers[0]['pair_id'],worst[0]['pair_id']]))
    fig,axes = plt.subplots(len(zoom_ids),2,figsize=(11,5*len(zoom_ids)),squeeze=False,constrained_layout=True)
    for row_axes,pair_id in zip(axes,zoom_ids):
        family = groups[pair_id]
        xy = np.concatenate([row['points'] for row in family])
        low,high = xy.min(axis=0)-.5,xy.max(axis=0)+.5
        nearby = fitting[((fitting>=low)&(fitting<=high)).all(axis=1)]
        for ax,method in zip(row_axes,('raw','adapted')):
            if len(nearby):ax.scatter(nearby[:,0],nearby[:,1],c='gray',marker='x',s=10,label='Fitted positions')
            costs = np.concatenate([row[method+'_costs'] for row in family])
            graphic = ax.scatter(xy[:,0],xy[:,1],c=costs,cmap='viridis',s=24)
            a,b = family[0]['pair']['a'],family[0]['pair']['b']
            ax.annotate('A',a,weight='bold');ax.annotate('B',b,weight='bold')
            ax.set(title=f'Pair {pair_id}: {method}',xlabel='x',ylabel='y',aspect='equal',xlim=(low[0],high[0]),ylim=(low[1],high[1]))
            fig.colorbar(graphic,ax=ax,shrink=.65,label='Measured MSE')
    fig.suptitle('Local zooms: colored points are fresh renders, not interpolated costs')
    save(fig,'local_sample_zooms')

    fig,ax = plt.subplots(figsize=(9,3.8),constrained_layout=True)
    bins = report['distance_bins']
    labels = [f"{r['lower']:g}–{r['upper'] if r['upper'] is not None else '∞'}" for r in bins]
    for method,color in (('raw','black'),('adapted','#0072b2')):
        ax.plot(range(len(bins)),[r[method]['up_fraction'] if r[method]['up_fraction'] is not None else np.nan for r in bins],
            marker='o',c=color,label=method)
    ax.set(xticks=range(len(bins)),xticklabels=labels,xlabel='Nearest fitting-position distance (minimum of step endpoints)',
        ylabel='Uphill step fraction',ylim=(-.03,1.03),title='Spatial coverage and wrong cost direction (counts in report)')
    ax.legend();ax.grid(alpha=.2)
    save(fig,'coverage_error')
    return exported
