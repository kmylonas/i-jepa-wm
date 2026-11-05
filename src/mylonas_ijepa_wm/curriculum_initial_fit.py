"""Fresh near-band fitting and immutable v0 references for the no-anchor ablation."""

from src.mylonas_ijepa_wm import curriculum_multiband_protocol as protocol
from src.mylonas_ijepa_wm.curriculum_protocol import digest, write_json
from src.mylonas_ijepa_wm.curriculum_replay import fit_adapter, save_adapter
from src.mylonas_ijepa_wm.curriculum_runtime import file_sha256


def enabled(curriculum):
    return curriculum.config.get('ablation') == protocol.NO_ANCHOR


def inputs(curriculum, recipe):
    return dict(recipe=recipe, version=0, initialization='identity', seed=0,
        fitting=curriculum.manifest['fitting'], pair_margin=curriculum.config['pair_rules'][recipe],
        goal_sha256=curriculum.goal_sha, experiment_digest=digest(curriculum.config),
        replay=curriculum.replay_entries(recipe, 0))


def completed(root):
    path = root / 'updates/v0/status.json'
    return path.exists() and protocol.read(path).get('status') == 'complete'


def reference(curriculum, recipe):
    root = protocol.runtime_root(curriculum.root, recipe)
    path = root / 'checkpoints/v0.pt'
    status = protocol.read(root / 'updates/v0/status.json')
    if (status.get('status') != 'complete' or status.get('inputs') != inputs(curriculum, recipe)
            or status.get('checkpoint_sha256') != file_sha256(path)
            or status.get('training_sha256') != file_sha256(root / 'updates/v0/training.json')):
        raise ValueError('initial checkpoint inputs/hash/provenance differ')
    rid = protocol.check_runtime(curriculum.root, recipe, curriculum.config, curriculum.manifest)
    return path, status['checkpoint_sha256'], rid


def validate_loaded(curriculum, recipe, saved):
    expected = inputs(curriculum, recipe)
    if (saved['provenance'].get('inputs_digest') != digest(expected)
            or saved['provenance'].get('experiment_digest') != digest(curriculum.config)
            or saved['training'].get('initialization') != 'identity'
            or saved['training'].get('anchor_weight') != 0
            or saved['training'].get('anchoring_enabled') is not False):
        raise ValueError('initial checkpoint fitting/provenance differs')


def ensure(curriculum, recipe, runtime):
    root = protocol.runtime_root(curriculum.root, recipe)
    status_path = root / 'updates/v0/status.json'
    expected = inputs(curriculum, recipe)
    if completed(root):
        curriculum.checkpoint(recipe, 0, runtime.device)
        return None
    if status_path.exists() and protocol.read(status_path).get('inputs') != expected:
        raise ValueError('interrupted initial fit inputs changed')
    if ((root / 'episodes/eval/v0').exists()
            or (root / 'heatmaps/v0.commit.json').exists()):
        raise ValueError('initial evaluations exist without a completed initial checkpoint')
    replay = curriculum.load_replay(recipe, 0, expected['replay'])
    write_json(status_path, dict(status='training', inputs=expected))
    try:
        adapter, training = fit_adapter(replay, curriculum.goal, seed=0,
            device=runtime.device, min_goal_progress=expected['pair_margin'],
            **expected['fitting'], progress=lambda row: print(
                f"{recipe}/near initial update {row['update']} rank_loss={row['rank_loss']:.5f}; anchor disabled",
                flush=True) if row['update'] == 1 or row['update'] % 50 == 0 else None)
    except (ValueError, RuntimeError) as error:
        if any(text in str(error).lower() for text in ('no eligible', 'no successful', 'nonfinite', 'non-finite')):
            return curriculum.stopped(recipe, 0, str(error))
        raise
    training['initialization'] = 'identity'
    checkpoint = root / 'checkpoints/v0.pt'
    save_adapter(checkpoint, adapter, training, dict(runtime_id=runtime.runtime_id,
        experiment_digest=digest(curriculum.config), inputs_digest=digest(expected)))
    training_path = root / 'updates/v0/training.json'
    write_json(training_path, {k: v for k, v in training.items() if k != 'optimizer'})
    write_json(status_path, dict(status='complete', inputs=expected,
        checkpoint_sha256=file_sha256(checkpoint), training_sha256=file_sha256(training_path)))
    return None
