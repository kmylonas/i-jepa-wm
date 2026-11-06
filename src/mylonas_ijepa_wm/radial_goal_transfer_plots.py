"""Full-right raw/adapted/true maps for original and moved goals."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


def plot_transfer(output,report):
    output=Path(output);grid=report['grid'];n=report['grid_count']
    shape=(len(grid['ys']),len(grid['xs']));training=np.asarray(report['training_positions'])
    old=report['goals'][0]['goal'];radius=report['source']['radius'];paths=[]
    for j,row in enumerate(report['goals']):
        raw=np.array(row['raw_costs']);adapted=np.array(row['adapted_costs']);truth=np.array(row['true_distances'])/34
        errors=np.abs(adapted-truth)*34;common=max(float(adapted[:n].max()),float(truth[:n].max()),1e-7)
        fig,axes=plt.subplots(1,4,figsize=(16,8),constrained_layout=True)
        for ax,values,title,maximum in zip(axes,(raw,adapted,truth,errors),
                ('Raw latent MSE','Frozen adapted MSE','True distance /34','Adapted distance error'),
                (max(float(raw[:n].max()),1e-7),common,common,max(float(errors[:n].max()),1e-7))):
            graphic=ax.pcolormesh(grid['xs'],grid['ys'],values[:n].reshape(shape),shading='nearest',
                cmap='viridis',vmin=0,vmax=maximum*1.03)
            ax.scatter(training[:,0],training[:,1],s=8,c='white',edgecolors='black',linewidths=.2)
            ax.add_patch(Circle(old,radius,fill=False,ec='orange',ls='--'))
            ax.scatter(*old,marker='x',s=60,c='black',label='Original goal')
            ax.scatter(*row['goal'],marker='*',s=110,c='orange',edgecolors='black',label='Evaluated goal')
            ax.set(title=title,xlabel='Ball x',ylabel='Ball y',aspect='equal',xlim=(35,60),ylim=(4,60))
            fig.colorbar(graphic,ax=ax,shrink=.6,label='Simulation units' if title.endswith('error') else 'Cost')
        axes[0].legend(fontsize=7,loc='upper left')
        fig.suptitle(f"{'Original-goal control' if j==0 else 'Changed goal'} {tuple(row['goal'])}; frozen distance adapter stage{report['source']['stage']}\n"
            'White: original fitting positions; dashed circle stays at original fitting region. Maps include outside-region probes.')
        for ext in ('png','pdf'):
            path=output/f'goal_{j}_heatmaps.{ext}';fig.savefig(path,dpi=140,bbox_inches='tight');paths.append(str(path))
        plt.close(fig)
        mask=np.asarray(row['common_mask'])&np.asarray(report['heldout_mask']);actual=np.array(row['true_distances'])
        fig,ax=plt.subplots(figsize=(6,4.5),constrained_layout=True)
        ax.scatter(actual[mask],raw[mask]*report['raw_calibration'],s=12,alpha=.5,c='gray',label='Original-calibration raw')
        ax.scatter(actual[mask],adapted[mask]*34,s=12,alpha=.5,c='#0072b2',label='Frozen adapter')
        maximum=max(float(actual[mask].max()) if mask.any() else 1.,1.)
        ax.plot([0,maximum],[0,maximum],'k--',label='True distance')
        ax.set(xlabel='True distance to evaluated goal',ylabel='Estimated distance',
            title=f"Goal {tuple(row['goal'])}: same heldout common-region positions")
        ax.grid(alpha=.2);ax.legend()
        for ext in ('png','pdf'):
            path=output/f'goal_{j}_calibration.{ext}';fig.savefig(path,dpi=140,bbox_inches='tight');paths.append(str(path))
        plt.close(fig)
    paths.extend(plot_goal_summary(output,report))
    return paths


def plot_goal_summary(output,report):
    goals=np.asarray([g['goal'] for g in report['goals']]);offsets=goals-goals[0]
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    arrays=([g['heldout_common']['adapted']['mae'] for g in report['goals']],
            [g['heldout_directional']['adapted_up_fraction'] for g in report['goals']])
    for ax,items,title,factor in zip(axes,arrays,('Heldout distance MAE','Heldout uphill fraction (%)'),(1,100)):
        values=np.array([np.nan if v is None else v*factor for v in items]);valid=np.isfinite(values)
        if valid.any():
            graphic=ax.scatter(offsets[valid,0],offsets[valid,1],c=values[valid],s=160,cmap='viridis',
                vmin=0,vmax=max(float(values[valid].max()),1e-7),edgecolors='black')
            fig.colorbar(graphic,ax=ax,shrink=.8,label='Simulation units' if factor==1 else '%')
        if (~valid).any():ax.scatter(offsets[~valid,0],offsets[~valid,1],marker='x',c='gray',s=90,label='N/A')
        ax.scatter(0,0,marker='*',s=260,facecolors='none',edgecolors='black',linewidths=1.5,label='Original goal')
        for j,(dx,dy) in enumerate(offsets):ax.annotate(str(j),(dx,dy),xytext=(7,7),textcoords='offset points')
        extent=max(1.,float(np.abs(offsets).max()))+1
        ax.set(title=title,xlabel='Goal shift dx',ylabel='Goal shift dy',aspect='equal',
            xlim=(-extent,extent),ylim=(-extent,extent))
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    count=report['goals'][0]['heldout_common']['count']
    fig.suptitle(f'Frozen adapter: same {count} heldout ball positions for every goal\nNumbers match the per-goal table; N/A is not zero error.')
    paths=[]
    for ext in ('png','pdf'):
        path=Path(output)/f'goal_summary.{ext}';fig.savefig(path,dpi=140,bbox_inches='tight');paths.append(str(path))
    plt.close(fig)
    return paths
