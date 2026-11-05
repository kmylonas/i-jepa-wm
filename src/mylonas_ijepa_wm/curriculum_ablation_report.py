"""Read historical anchored cases without rewriting the baseline experiment."""

from copy import deepcopy
from pathlib import Path

import torch

from src.mylonas_ijepa_wm import curriculum_multiband_protocol as protocol
from src.mylonas_ijepa_wm.curriculum_protocol import write_json
from src.mylonas_ijepa_wm.curriculum_runtime import file_sha256


def validated_heatmaps(curriculum):
    """Select numeric maps bound to completed, currently valid checkpoints."""
    report = curriculum.report()
    grid = curriculum.require_grid()
    signature = curriculum.grid_signature()
    maps, errors = {}, {}
    for recipe in protocol.RECIPES:
        maps[recipe] = {}
        branch = report['branches'][recipe]
        if branch['status'] == 'invalid':
            errors[recipe] = branch['error']
            continue
        try:
            for version, band in enumerate(protocol.BANDS):
                if branch['versions'][str(version)][band] is None:
                    continue
                folder = curriculum.root / f'branches/{recipe}/heatmaps'
                path = folder / f'v{version}.json'
                if not path.exists():
                    continue
                commit = protocol.read(folder / f'v{version}.commit.json')
                data = protocol.read(path)
                expected = dict(recipe=recipe, version=version,
                    checkpoint_sha256=curriculum.reference(recipe, version)[1],
                    grid_sha256=grid['sha256'], signature=signature)
                if (commit.get('data_sha256') != file_sha256(path)
                        or commit.get('image_sha256') != file_sha256(folder / f'v{version}.png')
                        or any(commit.get('inputs', {}).get(key) != value for key, value in expected.items())
                        or any(data.get(key) != value for key, value in commit['inputs'].items())
                        or data.get('points') != grid['points'].tolist()
                        or data.get('raw') != grid['raw_costs'].tolist()):
                    raise ValueError(f'{recipe} v{version}: map checkpoint/grid provenance differs')
                adapted = torch.as_tensor(data['adapted'])
                if (adapted.shape != grid['raw_costs'].shape
                        or not torch.isfinite(adapted).all() or (adapted < 0).any()):
                    raise ValueError(f'{recipe} v{version}: invalid adapted costs')
                maps[recipe][version] = data
        except (ValueError, OSError, KeyError) as error:
            maps[recipe], errors[recipe] = {}, str(error)
    return dict(maps=maps, errors=errors)


def compare_historical(curriculum, baseline_dir):
    from src.mylonas_ijepa_wm.curriculum_multiband import compare, summarize
    baseline_dir = Path(baseline_dir).resolve()
    if (baseline_dir == curriculum.root or baseline_dir in curriculum.root.parents
            or curriculum.root in baseline_dir.parents):
        raise ValueError('historical baseline and new output must be separate directories')
    baseline = type(curriculum)(curriculum.parent, curriculum.clean, curriculum.adjacent, baseline_dir)
    expected = deepcopy(curriculum.manifest)
    expected['fitting']['anchor_weight'] = baseline.manifest['fitting']['anchor_weight']
    if (baseline.config.get('ablation') is not None or baseline.manifest['fitting']['anchor_weight'] != 1
            or baseline.manifest != expected or baseline.config['sources'] != curriculum.config['sources']
            or not torch.equal(baseline.goal, curriculum.goal)):
        raise ValueError('historical comparison cases/goal/settings differ')
    # check_runtime can create execution.json in a genuinely new runtime;
    # historical reporting must never take that creation path.
    if (not (baseline.root / 'execution.json').is_file()
            and any((baseline.root / root / 'runtime.json').exists()
                    for root in ('raw', 'branches/original', 'branches/clean'))):
        raise ValueError('historical baseline execution metadata is missing')
    baseline.guard()
    if (baseline.root / 'execution.json').exists() and (curriculum.root / 'execution.json').exists():
        # Manifest and source fingerprints necessarily differ for this ablation.
        # Precision, model and environment settings must still match exactly.
        execution = lambda root: {key: value for key, value in protocol.read(root / 'execution.json').items()
                                  if key not in ('manifest', 'source')}
        if execution(baseline.root) != execution(curriculum.root):
            raise ValueError('historical comparison execution settings/precision differ')
    current = curriculum.report()
    branches = {}
    for recipe in protocol.RECIPES:
        empty = lambda: {str(v): dict.fromkeys(protocol.BANDS) for v in range(4)}
        cells = empty()
        root = protocol.runtime_root(baseline.root, recipe)
        error = None
        try:
            for version in range(4):
                path = root / f'updates/v{version}/status.json'
                ready = (root / 'runtime.json').exists() and (version == 0 or
                    path.exists() and protocol.read(path).get('status') == 'complete')
                if ready:
                    baseline.checkpoint(recipe, version)
                for band in protocol.BANDS:
                    before = baseline.rows(recipe, version, band, optional=True) if ready else None
                    has_after = current['branches'][recipe]['versions'][str(version)][band] is not None
                    after = curriculum.rows(recipe, version, band) if has_after else None
                    if before is not None and after is not None:
                        cells[str(version)][band] = dict(anchored=summarize(before),
                            without_anchor=summarize(after), paired=compare(before, after))
        except (ValueError, OSError, KeyError) as exception:
            cells, error = empty(), str(exception)
        branches[recipe] = dict(versions=cells, error=error)
    result = dict(schema_version=1, baseline=str(baseline.root), without_anchor=str(curriculum.root),
        branches=branches, note='Historical reference on identical cases and shared goal. Initial training and policy-dependent collections differ; this is not a fixed-data causal estimate of the anchor term.')
    write_json(curriculum.root / 'anchor_comparison.json', result)
    return result
