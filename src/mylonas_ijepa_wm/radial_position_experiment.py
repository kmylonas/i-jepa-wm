"""Validated position checkpoint lineage and streaming cached-token predictions."""
import numpy as np
import torch

from src.mylonas_ijepa_wm.curriculum_multiband_protocol import read
from src.mylonas_ijepa_wm.curriculum_protocol import digest,write_json
from src.mylonas_ijepa_wm.curriculum_runtime import file_sha256,save_tensors
from src.mylonas_ijepa_wm.radial_metrics import summarize
from src.mylonas_ijepa_wm import radial_position as position


FIGURES=('position_maps','coordinate_predictions','radial_distances','local_generalization','training_loss')
FILES=('model.pt','training.json','report.json')


def _hashes(folder):return {name:file_sha256(folder/name) for name in FILES}


def _completed(folder,inputs):
    complete=read(folder/'complete.json')
    if complete['inputs']!=inputs or complete['files']!=_hashes(folder):
        raise ValueError('position checkpoint/artifact hash or inputs differ')
    saved=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
    if (saved['kind']!='position_readout_v1' or saved['config_digest']!=inputs['config_digest']
            or saved['inputs_digest']!=digest(inputs) or saved['position_scale']!=position.SCALE
            or saved['position_center']!=list(position.CENTER)):
        raise ValueError('position checkpoint provenance differs')
    model=position.PositionReadout(**saved['shape']);model.load_state_dict(saved['state_dict']);model.eval()
    return model,read(folder/'report.json')


def _render(output,design,report):
    from src.mylonas_ijepa_wm.radial_position_plots import plot_position
    plot_position(output,design,report)


def position_stage(output,cache,index,config,design,cache_index,*,device='cpu',render=True):
    """Called by shared stage() after root/config/cache validation."""
    incoming=None;incoming_sha=None;actual=set()
    def inputs_for(i,sha):
        return dict(config_digest=digest(config),stage=i,seed=config['stage_seeds'][i-1],
            incoming_sha256=sha,training_ids=design['stages'][i-1]['training_ids'])
    for earlier in range(1,index):
        folder=output/f'stages/s{earlier}'
        incoming,prior=_completed(folder,inputs_for(earlier,incoming_sha))
        incoming_sha=file_sha256(folder/'model.pt');actual.update(prior['actual_fitting_ids'])
    inputs=inputs_for(index,incoming_sha);folder=output/f'stages/s{index}'
    if (folder/'complete.json').exists():
        _,report=_completed(folder,inputs)
        if render and any(not (folder/'figures'/f'{name}.{ext}').is_file() for name in FIGURES for ext in ('png','pdf')):
            _render(output,design,report)
        return report
    ids=inputs['training_ids'];data=torch.load(output/'training_data.pt',map_location='cpu',weights_only=True)
    lookup={point_id:i for i,point_id in enumerate(data['point_ids'])};take=[lookup[i] for i in ids]
    # Select coordinate labels before calling the fitter; probes never enter it.
    xy=torch.tensor([design['points'][i] for i in ids],dtype=torch.float32)
    model,training=position.fit(data['tokens'][take],xy,seed=inputs['seed'],initial_model=incoming,
        device=device,**config['fitting'],progress=lambda row:print(
            f"Position stage {index}, R={design['stages'][index-1]['outer_radius']}: "
            f"update {row['update']} loss={row['position_loss']:.6f}",flush=True))
    actual.update(ids[i] for i,n in enumerate(training['sample_counts']) if n)
    predictions=[]
    with torch.no_grad():
        for entry in cache_index['shards']:
            shard=torch.load(cache/entry['path'],map_location='cpu',weights_only=True)
            if shard['point_ids']!=list(range(entry['start'],entry['stop'])):
                raise ValueError('position cached point IDs differ')
            predictions.extend(model.predict(shard['tokens'].to(device)).cpu().tolist())
    predictions=np.asarray(predictions)
    if predictions.shape!=(len(design['points']),2) or not np.isfinite(predictions).all():
        raise ValueError('invalid position predictions')
    radius=design['stages'][index-1]['outer_radius']
    measured=position.summarize_positions(design,predictions,cache_index['frame_sha256'],ids,actual,radius)
    exposure=measured.pop('exposure')
    checks=summarize(design,read(output/'raw_costs.json'),
        np.asarray(measured['predicted_goal_distances'])/position.SCALE,
        cache_index['frame_sha256'],ids,actual,radius,include_goal_reference=False)
    folder.mkdir(parents=True,exist_ok=True)
    save_tensors(folder/'model.pt',dict(kind='position_readout_v1',shape=model.shape,
        state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
        config_digest=digest(config),inputs_digest=digest(inputs),
        position_center=list(position.CENTER),position_scale=position.SCALE))
    report=dict(stage=index,objective='position',outer_radius=radius,training_positions=len(ids),
        offered_training_ids=ids,actual_fitting_ids=sorted(actual),initialization=training['initialization'],
        incoming_sha256=incoming_sha,checkpoint_sha256=file_sha256(folder/'model.pt'),training=training,
        predicted_positions=predictions.tolist(),position=measured,exposure=exposure,spatial_checks=checks,
        derived_cost_definition='Euclidean distance(predicted_xy,known_goal)/34; not latent MSE',primitive_interactions=0)
    write_json(folder/'training.json',training);write_json(folder/'report.json',report)
    if render:_render(output,design,report)
    write_json(folder/'complete.json',dict(inputs=inputs,files=_hashes(folder)))
    return report
