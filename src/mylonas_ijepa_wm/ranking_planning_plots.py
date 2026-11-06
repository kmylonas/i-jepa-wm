"""Paired planning outcomes, full primitive paths and both-success efficiency."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Rectangle
import numpy as np


COLORS=dict(raw='#64748b',adapted='#d55e00')
LABELS=dict(raw='Raw latent MSE',adapted='Ranking-adapted MSE')


def _background(ax,points=None):
    for y,height in ((4,22),(34,26)):ax.add_patch(Rectangle((29,y),6,height,color='black',alpha=.18))
    ax.add_patch(Circle((50,42),34,fill=False,ls='--',ec='#0072b2',alpha=.6))
    limits=[30.,60.,4.,60.]
    if points is not None and len(points):
        p=np.asarray(points)
        for j,axis in ((0,0),(2,1)):
            if p[:,axis].min()<limits[j]:limits[j]=float(p[:,axis].min())-1
            if p[:,axis].max()>limits[j+1]:limits[j+1]=float(p[:,axis].max())+1
    ax.set(xlim=limits[:2],ylim=limits[2:],aspect='equal',xlabel='x',ylabel='y')
    ax.grid(alpha=.15)


def plot_report(output,r):
    output=Path(output);paths=[];manifest=r['manifest'];summary=r['summary']
    raw={x['case_id']:x for x in r['records']['raw']};adapted={x['case_id']:x for x in r['records']['adapted']}
    def save(fig,name):
        for ext in ('png','pdf'):
            p=output/f'{name}.{ext}';fig.savefig(p,dpi=140,bbox_inches='tight');paths.append(str(p))
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,8),constrained_layout=True);_background(ax)
    palette=plt.get_cmap('tab10')
    for gi,goal in enumerate(manifest['goals']):
        starts=np.asarray([c['start'] for c in manifest['evaluation'] if c['goal_id']==gi])
        ax.scatter(starts[:,0],starts[:,1],s=22,color=palette(gi),label=f'Goal {gi}: {tuple(goal)}')
        ax.scatter(*goal,marker='*',s=140,color=palette(gi),edgecolors='black')
        ax.annotate(str(gi),goal,xytext=(6,6),textcoords='offset points')
    ax.legend(fontsize=8,loc='upper left');ax.set_title('Fixed paired cases: dots are starts, stars are goals\nDashed: original R34 fitting region; shaded: wall')
    save(fig,'case_coverage')
    goal_rows=list(summary['by_goal'].values());ids=np.arange(len(goal_rows));width=.36
    fig,axes=plt.subplots(1,3,figsize=(15,4.7),constrained_layout=True)
    for condition in ('raw','adapted'):
        shift=-width/2 if condition=='raw' else width/2
        values=([g['conditions'][condition]['successes'] for g in goal_rows],
                [g['conditions'][condition]['mean_final_distance'] for g in goal_rows],
                [g['both_success_efficiency'][f'{condition}_mean_steps'] for g in goal_rows])
        for ax,array in zip(axes,values):
            ax.bar(ids+shift,[np.nan if v is None else v for v in array],width=width,color=COLORS[condition],label=LABELS[condition])
    for ax,title in zip(axes,('Successes per goal','Mean final distance: all cases','Mean primitive steps: BOTH succeed')):
        ax.set_xticks(ids,[f'G{j}\n({g[0]:g},{g[1]:g})' for j,g in enumerate(manifest['goals'])],fontsize=8)
        ax.set_title(title);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylim(0,manifest['starts_per_goal']);axes[1].set_ylabel('Simulation units')
    for j,g in enumerate(goal_rows):axes[2].annotate(f"n={g['both_success']}",(j,0),xytext=(0,3),textcoords='offset points',ha='center',fontsize=8)
    total=summary['overall'];fig.suptitle(f"Paired planning: raw {total['conditions']['raw']['successes']}/{total['episodes']}, adapted {total['conditions']['adapted']['successes']}/{total['episodes']}\n"
        f"Adapted wins {total['wins']}, losses {total['losses']}; goal 0 seen, others withheld")
    save(fig,'success_comparison')
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
    keys=list(raw);both=[k for k in keys if raw[k]['success'] and adapted[k]['success']]
    for ax,selected,field,title in zip(axes,(keys,both,both),('final_goal_distance','micro_steps','path_length'),
            ('Final distance: all cases','Primitive steps: both succeed','Path length: both succeed')):
        if selected:
            x=[raw[k][field] for k in selected];y=[adapted[k][field] for k in selected]
            ax.scatter(x,y,s=25,alpha=.7,c=[palette(raw[k]['case']['goal_id']) for k in selected])
            maximum=max(max(x),max(y),1.)*1.05;ax.plot([0,maximum],[0,maximum],'k--',alpha=.5)
            ax.set(xlim=(0,maximum),ylim=(0,maximum))
        else:ax.text(.5,.5,'No shared successes',transform=ax.transAxes,ha='center')
        ax.set(title=f'{title} (n={len(selected)})',xlabel='Raw',ylabel='Adapted');ax.grid(alpha=.2)
    fig.suptitle('Below diagonal favors adapted cost; efficiency uses the same successful tasks')
    save(fig,'paired_outcomes')
    for gi,goal in enumerate(manifest['goals']):
        cases=[c for c in manifest['evaluation'] if c['goal_id']==gi]
        cols=min(4,len(cases));rows=(len(cases)+cols-1)//cols
        fig,axes=plt.subplots(rows,cols,figsize=(4.5*cols,5*rows),constrained_layout=True,squeeze=False)
        for ax,case in zip(axes.flat,cases):
            entries=dict(raw=raw[case['case_id']],adapted=adapted[case['case_id']])
            points=[p for item in entries.values() for p in item['primitive_positions']];_background(ax,points)
            for condition,item in entries.items():
                xy=np.asarray(item['primitive_positions']);color=COLORS[condition]
                ax.plot(xy[:,0],xy[:,1],c=color,lw=1.5,label=f"{condition}: {'success' if item['success'] else 'failure'} ({item['micro_steps']} steps)")
                ax.scatter(*xy[-1],marker='x',s=45,c=color)
            ax.scatter(*case['start'],s=30,c='black',marker='o',label='Start')
            ax.scatter(*goal,s=130,c='orange',edgecolors='black',marker='*',label='Goal')
            ax.add_patch(Circle(goal,manifest['success_threshold'],fill=False,ec='orange',ls=':'))
            ax.set_title(f"{case['case_id']} / {case['band']} / d0={np.linalg.norm(np.asarray(case['start'])-goal):.1f}\n"
                f"Exit fitting region: raw={entries['raw']['ever_outside_fitting']}, adapted={entries['adapted']['ever_outside_fitting']}",fontsize=9)
            ax.legend(fontsize=7,loc='upper left')
        for ax in list(axes.flat)[len(cases):]:ax.set_visible(False)
        fig.suptitle(f"Goal {gi}: {tuple(goal)} — {'seen original reference' if gi==0 else 'withheld reference'}\n"
            'Complete primitive paths; limits expand when needed to show excursions. Dotted orange: success radius.')
        save(fig,f'goal_{gi}_trajectories')
    return paths
