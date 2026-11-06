"""Ordinal-transfer summaries and unclipped full-right ranking landscapes."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


LABELS=dict(single='Single-goal ranking',multi='Multi-goal ranking',distance='Frozen multi-goal distance')
COLORS=dict(single='#0072b2',multi='#d55e00',distance='#009e73')


def plot_run(output,r):
    output=Path(output);paths=[];rows=r['methods']['multi']
    goals=np.asarray([g['goal'] for g in rows]);nodes=np.asarray(r['training_positions']);points=np.asarray(r['points'])
    mask=np.asarray(rows[0]['common_mask'])&np.asarray(r['heldout_mask'])
    def save(fig,name):
        for ext in ('png','pdf'):
            p=output/f'{name}.{ext}';fig.savefig(p,dpi=140,bbox_inches='tight');paths.append(str(p))
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,8),constrained_layout=True)
    ax.scatter(points[mask,0],points[mask,1],s=10,c='gray',alpha=.4,label=f'Shared heldout queries (N={mask.sum()})')
    ax.scatter(nodes[:,0],nodes[:,1],s=18,c=COLORS['single'],label=f'Fitting nodes ({len(nodes)})')
    ax.scatter(goals[1:,0],goals[1:,1],marker='*',s=100,c='orange',edgecolors='black',label='Withheld goals')
    ax.scatter(*goals[0],marker='x',s=80,c='black',label='Original goal (seen)')
    for j,g in enumerate(goals):ax.annotate(str(j),g,xytext=(5,5),textcoords='offset points',fontsize=8)
    ax.add_patch(Circle(goals[0],34,fill=False,ls='--',ec='orange'))
    ax.set(xlim=(35,60),ylim=(4,60),aspect='equal',xlabel='Ball x',ylabel='Ball y',title='Manual fitting nodes and reserved goal references')
    ax.legend(fontsize=8,loc='upper left');save(fig,'coverage')
    fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    for b in ('single','multi'):
        history=r['training'][b]['history'];fixed=[h for h in history if 'rank_loss' in h];x=[h['update'] for h in fixed]
        axes[0].plot([h['update'] for h in history if 'minibatch_loss' in h],
            [h['minibatch_loss'] for h in history if 'minibatch_loss' in h],c=COLORS[b],alpha=.25)
        axes[0].plot(x,[max(h['rank_loss'],1e-14) for h in fixed],c=COLORS[b],label=LABELS[b])
        axes[1].plot(x,[h['rank_accuracy']*100 for h in fixed],c=COLORS[b],label=LABELS[b])
        axes[2].plot(x,[h['mean_far_cost'] for h in fixed],c=COLORS[b],label=LABELS[b]+' far')
        axes[2].plot(x,[h['mean_near_cost'] for h in fixed],c=COLORS[b],ls='--',label=LABELS[b]+' near')
    axes[0].set(yscale='log',title='Solid: fixed train probe; faint: minibatches',ylabel='Ranking softplus loss')
    axes[1].set(title='Fixed training probe ordering',ylabel='Correct ranking (%)',ylim=(-2,102))
    axes[2].set(title='Fixed training probe cost magnitudes',ylabel='Adapted MSE (arbitrary units)')
    for ax in axes:ax.set_xlabel('Optimizer update');ax.grid(alpha=.2);ax.legend(fontsize=7)
    save(fig,'training_loss')
    fig,axes=plt.subplots(2,1,figsize=(13,8),constrained_layout=True);ids=np.arange(len(rows))
    for name in ('raw','single','multi','distance'):
        selected=rows if name=='raw' else r['methods'][name];prefix='raw' if name=='raw' else 'adapted'
        up=[g['heldout_directional'][prefix+'_up_fraction'] for g in selected]
        correct=[g[prefix+'_rank']['correct_fraction'] for g in selected]
        kw=dict(color='gray' if name=='raw' else COLORS[name],label='Raw MSE' if name=='raw' else LABELS[name],
            marker='o',ls='--' if name in ('raw','distance') else '-')
        for ax,values in zip(axes,(up,correct)):ax.plot(ids,[np.nan if v is None else v*100 for v in values],**kw)
    labels=[f'{j}\n({g[0]:.1f},{g[1]:.1f})' for j,g in enumerate(goals)]
    for ax in axes:
        ax.set_xticks(ids,labels,fontsize=7);ax.axvspan(-.4,.4,color='gray',alpha=.12)
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].set_ylabel('Uphill approaching steps (%) — lower better')
    axes[1].set_ylabel('Correct heldout pair ranking (%) — higher better')
    axes[1].set_xlabel('Goal ID; goal 0 seen, all other goal references withheld')
    fig.suptitle(f'Same {mask.sum()} heldout queries for every goal/model; no distance calibration\n'
        'Pairs require a physical distance gap >=0.5. N/A is not zero error. Ties are reported separately.')
    save(fig,'goal_summary')
    n=r['grid_count'];shape=(len(r['grid']['ys']),len(r['grid']['xs']))
    for j,row in enumerate(rows):
        values=[np.asarray(row['raw_costs'])[:n]]+[np.asarray(r['methods'][b][j]['adapted_costs'])[:n] for b in ('single','multi','distance')]+[np.asarray(row['true_distances'])[:n]]
        titles=['Raw latent MSE','Single-goal ranking MSE','Multi-goal ranking MSE','Multi-goal distance MSE','True Euclidean distance']
        fig,axes=plt.subplots(1,5,figsize=(19,8),constrained_layout=True)
        for ax,v,title in zip(axes,values,titles):
            graphic=ax.pcolormesh(r['grid']['xs'],r['grid']['ys'],v.reshape(shape),shading='nearest',cmap='viridis',vmin=0,vmax=max(float(v.max()),1e-7)*1.03)
            ax.scatter(nodes[:,0],nodes[:,1],c='white',s=8,edgecolors='black',linewidths=.2)
            ax.add_patch(Circle(goals[0],34,fill=False,ls='--',ec='orange'))
            ax.scatter(*goals[0],marker='x',c='black',s=60,label='Original goal')
            ax.scatter(*row['goal'],marker='*',c='orange',edgecolors='black',s=110,label='Evaluated goal')
            ax.set(title=title,xlim=(35,60),ylim=(4,60),aspect='equal',xlabel='Ball x',ylabel='Ball y')
            fig.colorbar(graphic,ax=ax,shrink=.6,label='Simulation units' if title.startswith('True') else 'MSE cost')
        axes[0].legend(fontsize=7)
        fig.suptitle(f"Goal {j}: {tuple(row['goal'])} ({row['reference_status']})\n"
            'Separate unclipped color scales: compare shapes/ordering, not colors across panels. White: fitting nodes. Full maps include outside-common-region probes.')
        save(fig,f'goal_{j}_heatmaps')
    return paths
