"""Matched goal-transfer comparisons and full-right cost landscapes."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


COLORS=dict(single='#0072b2',multi='#d55e00',historical='#009e73')
LABELS=dict(single='Fresh single-goal',multi='Multi-goal',historical='Historical curriculum')


def plot_run(output,report):
    output=Path(output);paths=[]
    def save(fig,name):
        for ext in ('png','pdf'):
            path=output/f'{name}.{ext}';fig.savefig(path,dpi=140,bbox_inches='tight');paths.append(str(path))
        plt.close(fig)
    rows=report['methods']['multi'];goals=np.array([g['goal'] for g in rows])
    nodes=np.asarray(report['training_positions']);points=np.asarray(report['points'])
    mask=np.asarray(rows[0]['common_mask'])&np.asarray(report['heldout_mask'])
    fig,ax=plt.subplots(figsize=(7,8),constrained_layout=True)
    ax.scatter(points[mask,0],points[mask,1],s=9,c='gray',alpha=.45,label=f'Shared heldout queries (N={mask.sum()})')
    ax.scatter(nodes[:,0],nodes[:,1],s=18,c=COLORS['single'],label=f'Fitting nodes ({len(nodes)})')
    ax.scatter(goals[1:,0],goals[1:,1],s=90,marker='*',c='orange',edgecolors='black',label='Withheld reference goals')
    ax.scatter(*goals[0],s=80,marker='x',c='black',label='Original goal (seen)')
    for j,g in enumerate(goals):ax.annotate(str(j),g,xytext=(5,5),textcoords='offset points',fontsize=8)
    ax.add_patch(Circle(goals[0],34,fill=False,ls='--',ec='orange'))
    ax.set(xlim=(35,60),ylim=(4,60),aspect='equal',xlabel='Ball x',ylabel='Ball y',
        title='Fitting nodes, reserved goals and common heldout queries')
    ax.legend(fontsize=8,loc='upper left');save(fig,'coverage')
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for branch in ('single','multi'):
        history=report['training'][branch]['history']
        fixed=[r for r in history if 'regression_loss' in r]
        axes[0].plot([r['update'] for r in history if 'minibatch_loss' in r],
            [r['minibatch_loss'] for r in history if 'minibatch_loss' in r],alpha=.3,c=COLORS[branch])
        axes[0].plot([r['update'] for r in fixed],[max(r['regression_loss'],1e-14) for r in fixed],
            c=COLORS[branch],label=LABELS[branch]+' fixed train probe')
        axes[1].plot([r['update'] for r in fixed],[r['distance_mae'] for r in fixed],c=COLORS[branch],label=LABELS[branch])
    axes[0].set(yscale='log',ylabel='Pair regression loss',title='Solid: fixed training probe; faint: minibatches')
    axes[1].set(ylabel='Distance MAE (simulation units)',title='Fixed training probes (different pair populations)')
    for ax in axes:ax.set_xlabel('Optimizer update');ax.legend(fontsize=8);ax.grid(alpha=.2)
    save(fig,'training_loss')
    fig,axes=plt.subplots(2,1,figsize=(13,8),constrained_layout=True)
    ids=np.arange(len(rows))
    for method in ('raw','single','multi','historical'):
        selected=rows if method=='raw' else report['methods'][method]
        maes=[r['heldout_common']['raw_calibrated' if method=='raw' else 'adapted']['mae'] for r in selected]
        uphill=[r['heldout_directional'][('raw' if method=='raw' else 'adapted')+'_up_fraction'] for r in selected]
        kw=dict(color='gray' if method=='raw' else COLORS[method],
            label='Raw (training-pair scalar)' if method=='raw' else LABELS[method],
            marker='o',ls='--' if method in ('raw','historical') else '-')
        axes[0].plot(ids,[np.nan if v is None else v for v in maes],**kw)
        axes[1].plot(ids,[np.nan if v is None else v*100 for v in uphill],**kw)
    labels=[f'{j}\n({g[0]:.1f},{g[1]:.1f})' for j,g in enumerate(goals)]
    for ax in axes:
        ax.set_xticks(ids,labels,fontsize=7);ax.axvspan(-.4,.4,color='gray',alpha=.12)
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].set_ylabel('Heldout distance MAE');axes[1].set_ylabel('Uphill steps (%)')
    axes[1].set_xlabel('Goal ID and coordinates; goal 0 is seen, all others withheld')
    fig.suptitle(f'Same {int(mask.sum())} heldout queries across methods/goals; historical run is descriptive\n'
        'Lower is better; N/A is not zero error; these are spatial diagnostics, not planning trials')
    save(fig,'goal_summary')
    n=report['grid_count'];shape=(len(report['grid']['ys']),len(report['grid']['xs']))
    for j,row in enumerate(rows):
        raw=np.asarray(row['raw_costs'])[:n];multi=np.asarray(row['adapted_costs'])[:n]
        single=np.asarray(report['methods']['single'][j]['adapted_costs'])[:n]
        truth=np.asarray(row['true_distances'])[:n]/34;error=np.abs(multi-truth)*34
        maximum=max(float(np.max(v)) for v in (single,multi,truth))*1.03
        fig,axes=plt.subplots(1,5,figsize=(19,8),constrained_layout=True)
        for ax,values,title,vmax in zip(axes,(raw,single,multi,truth,error),
            ('Raw latent MSE','Fresh single-goal MSE','Multi-goal MSE','True distance /34','Multi-goal distance error'),
            (max(float(raw.max()),1e-7)*1.03,maximum,maximum,maximum,max(float(error.max()),1e-7)*1.03)):
            graphic=ax.pcolormesh(report['grid']['xs'],report['grid']['ys'],values.reshape(shape),
                shading='nearest',cmap='viridis',vmin=0,vmax=vmax)
            ax.scatter(nodes[:,0],nodes[:,1],c='white',s=8,edgecolors='black',linewidths=.2)
            ax.add_patch(Circle(goals[0],34,fill=False,ls='--',ec='orange'))
            ax.scatter(*goals[0],marker='x',c='black',s=60,label='Original goal')
            ax.scatter(*row['goal'],marker='*',c='orange',edgecolors='black',s=110,label='Evaluated goal')
            ax.set(title=title,xlim=(35,60),ylim=(4,60),aspect='equal',xlabel='Ball x',ylabel='Ball y')
            fig.colorbar(graphic,ax=ax,shrink=.6,label='Simulation units' if title.endswith('error') else 'Cost')
        axes[0].legend(fontsize=7)
        fig.suptitle(f"Goal {j}: {tuple(row['goal'])} ({row['reference_status']} reference)\n"
            'White: offered fitting nodes. Dashed: original R34 region. Full maps include outside-common-region queries.')
        save(fig,f'goal_{j}_heatmaps')
    return paths
