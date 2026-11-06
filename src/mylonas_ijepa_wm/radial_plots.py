"""Per-stage radial geometry, costs, heldout results and fitting curves."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


def plot_stage(output,design,raw,report):
    folder=Path(output)/f"stages/s{report['stage']}/figures"
    folder.mkdir(parents=True,exist_ok=True)
    paths=[]
    def save(fig,name):
        for suffix in ('png','pdf'):
            path=folder/f'{name}.{suffix}';fig.savefig(path,dpi=140,bbox_inches='tight');paths.append(str(path))
        plt.close(fig)
    grid=design['grid'];ids=grid['point_ids'];points=np.asarray(design['points'])
    training=points[report['offered_training_ids']]
    radius=report['outer_radius'];values=dict(raw=np.asarray(raw),adapted=np.asarray(report['adapted_costs']))
    fig,axes=plt.subplots(1,2,figsize=(10,9),constrained_layout=True)
    for ax,method in zip(axes,values):
        z=values[method][ids].reshape(len(grid['ys']),len(grid['xs']))
        graphic=ax.pcolormesh(grid['xs'],grid['ys'],z,shading='nearest',cmap='viridis',vmin=0,vmax=max(float(z.max()),1e-7)*1.03)
        ax.scatter(training[:,0],training[:,1],s=12,c='white',edgecolors='black',linewidths=.2,label='Training positions')
        ax.scatter(50,42,c='orange',marker='*',s=120,edgecolors='black')
        ax.add_patch(Circle((50,42),radius,fill=False,ec='orange',ls='--',lw=1.5))
        ax.set(title=f'{method.capitalize()} MSE; max={z.max():.4g}',xlabel='x',ylabel='y',aspect='equal',
            xlim=(35,60),ylim=(4,60))
        fig.colorbar(graphic,ax=ax,shrink=.65,label='Independent full-range cost scale')
    axes[1].legend(fontsize=8,loc='upper left')
    fig.suptitle(f"Stage {report['stage']}: trained through radius {radius}, {len(training)} hand-positioned images\n"
        'Dashed circle: current training radius. Outside it: not yet trained.')
    save(fig,'heatmaps')

    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for row,kind in enumerate(('trained_direction','withheld_direction')):
        for col,method in enumerate(values):
            ax=axes[row,col]
            for curve in report['curves']:
                if curve['kind']!=kind:continue
                highlighted=curve['angle']==(270 if kind=='trained_direction' else 262.5)
                ax.plot(curve['radii'],curve[method+'_costs'],alpha=1 if highlighted else .35,
                    lw=2.5 if highlighted else .8,c='black' if highlighted else None)
            ax.axvline(radius,c='orange',ls='--',label='Training limit')
            ax.set(title=f"{kind.replace('_',' ')}: {method}",xlabel='Radius outward from goal →',ylabel='MSE')
            ax.grid(alpha=.2)
    axes[0,0].legend()
    fig.suptitle('Cost versus radius: rising is desired\nTraining directions include withheld midpoint radii; withheld directions never supply fitting inputs.')
    save(fig,'radial_profiles')

    fig,axes=plt.subplots(1,2,figsize=(12,3.8),constrained_layout=True)
    for ax,key,title in zip(axes,('offset_fitted_centers','offset_summary'),
            ('Matched families with offered training endpoints','Fixed matched families across all radii')):
        rows=report[key]
        for method,color in (('raw','black'),('adapted','#0072b2')):
            ax.plot([r['offset'] for r in rows],[r[method]['up_fraction'] if r[method]['up_fraction'] is not None else np.nan for r in rows],
                c=color,marker='o',label=method)
        ax.set(title=f"{title}\n{rows[0]['families']} families",xlabel='Absolute perpendicular offset',ylabel='Uphill step fraction',ylim=(-.03,1.03))
        ax.grid(alpha=.2)
    axes[0].legend()
    fig.suptitle('Same family cohort at every offset; ± signs averaged within each family')
    save(fig,'offset_summary')

    centers=[p for p in report['profiles'] if p['offset']==0]
    trained=[p for p in centers if p['point_ids'][0] in report['offered_training_ids'] and p['point_ids'][-1] in report['offered_training_ids']]
    worst=sorted(centers,key=lambda p:p['adapted']['up_fraction'],reverse=True)
    examples=[]
    for row in trained[:2]+worst[:2]:
        if row['family_id'] not in [r['family_id'] for r in examples]:examples.append(row)
    fig,axes=plt.subplots(len(examples),2,figsize=(12,2.8*len(examples)),squeeze=False,constrained_layout=True)
    for row_axes,center in zip(axes,examples):
        family=[p for p in report['profiles'] if p['family_id']==center['family_id']]
        for ax,method in zip(row_axes,values):
            for row in family:
                y=row[method+'_costs']
                ax.plot(np.linspace(0,1,len(y)),y,lw=2 if row['offset']==0 else .7,
                    alpha=1 if row['offset']==0 else .5,label=f"offset {row['offset']:g}")
            ax.set(title=f"Family {center['family_id']}, {center['region']}: {method}",xlabel='Far → near (11 rendered coordinates)',ylabel='MSE')
            ax.grid(alpha=.2)
    axes[0,1].legend(ncol=2,fontsize=7)
    fig.suptitle('First two offered-pair examples + two worst center profiles (deduplicated)\nThick: original segment; thin: shifted. Falling is desired.')
    save(fig,'interpolation_examples')

    fig,ax=plt.subplots(figsize=(9,3),constrained_layout=True)
    regression=report.get('objective')=='distance'
    loss_key='regression_loss' if regression else 'rank_loss';full_key='full_'+loss_key
    history=report['training']['history'];sampled=[r for r in history if loss_key in r]
    ax.plot([r['update'] for r in sampled],[r[loss_key] for r in sampled],alpha=.4,label='Sampled batch')
    fixed=[r for r in history if full_key in r]
    ax.plot([r['update'] for r in fixed],[r[full_key] for r in fixed],marker='o',
        label='All training positions' if regression else 'All eligible training pairs')
    ax.set(title=f"Stage {report['stage']}: {'distance regression' if regression else 'ranking only'}, no anchoring",
        xlabel='Optimizer update',ylabel='Regression MSE' if regression else 'Sampled batch rank loss')
    ax.legend();ax.grid(alpha=.2);save(fig,'training_loss')
    return paths
