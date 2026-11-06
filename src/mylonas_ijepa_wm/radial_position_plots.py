"""Coordinate accuracy, distance shape and local tests for each position stage."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


def plot_position(output,design,report):
    folder=Path(output)/f"stages/s{report['stage']}/figures";folder.mkdir(parents=True,exist_ok=True)
    paths=[]
    def save(fig,name):
        for suffix in ('png','pdf'):
            path=folder/f'{name}.{suffix}';fig.savefig(path,dpi=140,bbox_inches='tight');paths.append(str(path))
        plt.close(fig)
    points=np.asarray(design['points']);pred=np.asarray(report['predicted_positions'])
    p=report['position'];truth=np.asarray(p['true_distances']);dist=np.asarray(p['predicted_goal_distances'])
    error=np.asarray(p['position_errors']);radius=report['outer_radius'];goal=design['goal']
    train=points[report['offered_training_ids']];grid=design['grid'];grid_ids=np.array(grid['point_ids'])
    shape=(len(grid['ys']),len(grid['xs']))
    common=max(float(truth[grid_ids].max()),float(dist[grid_ids].max()),1e-7)
    fig,axes=plt.subplots(1,3,figsize=(14,8),constrained_layout=True)
    for ax,values,title,vmax in zip(axes,(truth,dist,error),
            ('True distance to goal','Distance from predicted position','Euclidean position error'),
            (common,common,max(float(error[grid_ids].max()),1e-7))):
        graphic=ax.pcolormesh(grid['xs'],grid['ys'],values[grid_ids].reshape(shape),
            shading='nearest',cmap='viridis',vmin=0,vmax=vmax*1.03)
        ax.scatter(train[:,0],train[:,1],s=9,c='white',edgecolors='black',linewidths=.2)
        ax.scatter(*goal,marker='*',c='orange',s=100,edgecolors='black')
        ax.add_patch(Circle(goal,radius,fill=False,ec='orange',ls='--'))
        ax.set(title=f'{title}\nmax={values[grid_ids].max():.3g}',xlabel='True x',ylabel='True y',
            aspect='equal',xlim=(35,60),ylim=(4,60))
        fig.colorbar(graphic,ax=ax,shrink=.6,label='Simulation units')
    fig.suptitle(f"Position stage {report['stage']}: trained through radius {radius}\n"
        'White points: coordinate supervision; dashed circle: training limit. Full-right grid includes corners beyond34.')
    save(fig,'position_maps')

    ids=np.array(sorted({i for c in design['curves'] for i in c['point_ids']
        if report['exposure']['heldout_from_offered_training'][i]}),dtype=int)
    inside=truth[ids]<=radius+1e-5
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for axis,ax in enumerate(axes[0]):
        for mask,label,color in ((inside,'Heldout inside','#0072b2'),(~inside,'Heldout outside','#d55e00')):
            ax.scatter(points[ids[mask],axis],pred[ids[mask],axis],s=9,c=color,alpha=.4,label=label)
        ax.scatter(train[:,axis],pred[report['offered_training_ids'],axis],marker='x',s=20,c='black',label='Training')
        lower=min(points[ids,axis].min(),pred[ids,axis].min());upper=max(points[ids,axis].max(),pred[ids,axis].max())
        ax.plot([lower,upper],[lower,upper],'k--')
        name='xy'[axis];ax.set(xlabel=f'True {name}',ylabel=f'Predicted {name}',title=f'{name} coordinate')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    ax=axes[1,0];selected=grid_ids.reshape(shape)[::3,::3].ravel();a=points[selected];b=pred[selected]
    ax.quiver(a[:,0],a[:,1],b[:,0]-a[:,0],b[:,1]-a[:,1],angles='xy',scale_units='xy',scale=1,
        width=.004,color='#0072b2',alpha=.6)
    ax.scatter(a[:,0],a[:,1],s=10,c='black');ax.scatter(b[:,0],b[:,1],s=10,c='#0072b2')
    ax.add_patch(Circle(goal,radius,fill=False,ec='orange',ls='--'))
    ax.set(xlabel='x',ylabel='y',aspect='equal',title='Fixed grid subset: true (black) → predicted (blue)')
    ax.set_xlim(min(35,float(b[:,0].min()))-1,max(60,float(b[:,0].max()))+1)
    ax.set_ylim(min(4,float(b[:,1].min()))-1,max(60,float(b[:,1].max()))+1)
    ax=axes[1,1]
    for mask,label,color in ((inside,'Inside','#0072b2'),(~inside,'Outside','#d55e00')):
        ax.scatter(truth[ids[mask]],error[ids[mask]],s=9,c=color,alpha=.4,label=label)
    ax.axvline(radius,c='orange',ls='--');ax.set(xlabel='True radius',ylabel='Position error',title='Heldout directional positions')
    ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Primary test: predict both coordinates, including on unseen positions')
    save(fig,'coordinate_predictions')

    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,kind in zip(axes,('trained_direction','withheld_direction')):
        for c in design['curves']:
            if c['kind']==kind:ax.plot(truth[c['point_ids']],dist[c['point_ids']],alpha=.4,lw=1)
        ax.plot([0,34],[0,34],'k--',label='True distance');ax.axvline(radius,c='orange',ls='--',label='Training limit')
        ax.set(title=kind.replace('_',' '),xlabel='True radius',ylabel='Distance from predicted xy')
        ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Derived distance; good distance alone does not establish good position prediction')
    save(fig,'radial_distances')

    checks=report['spatial_checks'];fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    ax=axes[0,0]
    for key,style,label in (('offset_summary','-','All matched'),('offset_fitted_centers','--','Offered endpoints')):
        rows=checks[key]
        for method,color,name in (('raw','black','raw MSE'),('adapted','#0072b2','position distance')):
            y=[r[method]['up_fraction'] if r[method]['up_fraction'] is not None else np.nan for r in rows]
            ax.plot([r['offset'] for r in rows],y,style,c=color,marker='o',label=f'{label}: {name}')
    ax.set(xlabel='Absolute perpendicular offset',ylabel='Wrong-direction fraction',ylim=(-.03,1.03),title='Matched cohorts at every offset')
    ax.legend(fontsize=7);ax.grid(alpha=.2)
    centers=[p for p in checks['profiles'] if p['offset']==0]
    offered=[p for p in centers if p['point_ids'][0] in report['offered_training_ids'] and p['point_ids'][-1] in report['offered_training_ids']]
    examples=[]
    for row in offered[:1]+sorted(centers,key=lambda p:p['adapted']['up_fraction'],reverse=True):
        if row['family_id'] not in [r['family_id'] for r in examples]:examples.append(row)
        if len(examples)==3:break
    for ax,center in zip((axes[0,1],axes[1,0],axes[1,1]),examples):
        for row in checks['profiles']:
            if row['family_id']!=center['family_id']:continue
            ax.plot(np.linspace(0,1,len(row['point_ids'])),np.asarray(row['adapted_costs'])*34,
                lw=2 if row['offset']==0 else .7,alpha=1 if row['offset']==0 else .5,label=f"offset{row['offset']:g}")
        ax.plot(np.linspace(0,1,len(center['point_ids'])),truth[center['point_ids']],'k--',label='True center distance')
        ax.set(title=f"Family{center['family_id']}, {center['region']}",xlabel='Far → near',ylabel='Position-derived distance')
        ax.grid(alpha=.2)
    axes[0,1].legend(ncol=2,fontsize=7)
    fig.suptitle('Local interpolation and offsets: falling toward goal is desired\nOne offered-endpoint example, then worst central profiles (deduplicated)')
    save(fig,'local_generalization')

    history=report['training']['history'];sampled=[r for r in history if 'position_loss' in r]
    fixed=[r for r in history if 'full_position_loss' in r]
    fig,axes=plt.subplots(1,2,figsize=(12,3.5),constrained_layout=True)
    axes[0].plot([r['update'] for r in sampled],[r['position_loss'] for r in sampled],alpha=.4,label='Sampled batch')
    axes[0].plot([r['update'] for r in fixed],[r['full_position_loss'] for r in fixed],marker='o',label='All training positions')
    axes[0].set(xlabel='Optimizer update',ylabel='Normalized coordinate MSE');axes[0].legend()
    axes[1].plot([r['update'] for r in fixed],[r['mean_position_error'] for r in fixed],marker='o')
    axes[1].set(xlabel='Optimizer update',ylabel='Mean Euclidean position error',title='Training pool, simulation units')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle(f"Position stage {report['stage']}: fixed schedule, no probe-based selection")
    save(fig,'training_loss')
    return paths
