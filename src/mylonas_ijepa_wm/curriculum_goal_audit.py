"""Read-only diagnosis of cached-goal, rendering and batch-encoding consistency."""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.mylonas_ijepa_wm import curriculum_heatmaps as heatmaps
from src.mylonas_ijepa_wm import curriculum_multiband_protocol as protocol
from src.mylonas_ijepa_wm import plan_pointmaze as visual
from src.mylonas_ijepa_wm.curriculum_protocol import digest, write_json
from src.mylonas_ijepa_wm.curriculum_runtime import (
    DINO_COMMIT, PilotRuntime, file_sha256, git_output, save_tensors,
)
from src.mylonas_ijepa_wm.environment_adapters import WallAdapter
from src.mylonas_ijepa_wm.environment_configs import EnvironmentConfig


def comparison(left, right):
    a, b = torch.as_tensor(left).cpu(), torch.as_tensor(right).cpu()
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('comparison requires matching finite tensors')
    delta = a.float() - b.float()
    return dict(mse=float(delta.square().mean()), max_abs=float(delta.abs().max()),
                exact=bool(torch.equal(a, b)))


def load_saved(curriculum, clean):
    """Validate and inspect existing caches without opening an experiment runtime."""
    root, clean = Path(curriculum), Path(clean)
    config = protocol.read(root / 'experiment.json')
    source = clean / 'run_0/goal_tokens.pt'
    expected = config['sources']['clean']['files']['run_0/goal_tokens.pt']
    if file_sha256(source) != expected:
        raise ValueError('source goal cache hash differs from the curriculum reference')
    goal, goal_sha = protocol.load_goal(root, config)
    source_tokens = torch.load(source, map_location='cpu', weights_only=True)['tokens']
    meta = protocol.read(root / 'cache/grid.json')
    if (meta['signature']['goal_sha256'] != goal_sha
            or meta['signature']['experiment_digest'] != digest(config)
            or meta['signature']['execution'] != protocol.read(root / 'execution.json')):
        raise ValueError('grid goal/configuration/execution provenance differs')
    grid = heatmaps.load_grid(root, meta['signature'])
    indices = torch.where((grid['points'] == torch.tensor([50., 42.])).all(dim=1))[0]
    if len(indices) != 1:
        raise ValueError('grid must contain exactly one goal at (50,42)')
    index = int(indices.item())
    grid_goal = grid['tokens'][index].clone()
    cost = comparison(grid_goal, goal)
    stored = float(grid['raw_costs'][index])
    if not np.isclose(stored, cost['mse'], rtol=1e-6, atol=1e-8):
        raise ValueError('stored raw goal cost differs from recomputed MSE')
    batch_start = index // 8 * 8
    report = dict(goal_position=[50., 42.], goal_grid_index=index,
        goal_self=comparison(goal, goal), goal_copy=comparison(goal, source_tokens),
        grid_target_copy=comparison(grid['goal'], goal), grid_goal_vs_cached_goal=cost,
        stored_raw_goal_cost=stored, raw_grid_max=float(grid['raw_costs'].max()),
        source_goal_sha256=expected, copied_goal_sha256=goal_sha, grid_sha256=grid['sha256'],
        previous_execution=meta['signature']['execution'])
    return dict(goal=goal, grid_goal=grid_goal,
        grid_batch_points=grid['points'][batch_start:batch_start+8].tolist(),
        batch_goal_offset=index-batch_start, report=report, config=config)


def probe(saved, env, encode_single, encode_batch):
    """Hold pixels fixed while comparing actual single and eight-image paths."""
    reset = dict(layout=[32, 30], start=[44, 34], goal=[50, 42])
    env.reset(0, **reset)
    frames = dict(render_goal=env.render_goal().copy(), render_goal_repeat=env.render_goal().copy(),
                  grid_route=env._render_at([50, 42]).copy())
    mixed = np.stack([env._render_at(point) for point in saved['grid_batch_points']])
    offset = saved['batch_goal_offset']
    frames['grid_batch_goal'] = mixed[offset].copy()
    frames['observation_at_goal'] = env.reset(
        0, layout=[32, 30], start=[50, 42], goal=[50, 42]).frame.copy()
    env.reset(1, **reset)
    frames['other_reset_seed'] = env.render_goal().copy()
    base = frames['render_goal']
    pixels = {f'render_goal_vs_{name}': comparison(base, frame)
              for name, frame in frames.items() if name != 'render_goal'}
    frame_meta = {name: dict(sha256=hashlib.sha256(frame.tobytes()).hexdigest(),
        shape=list(frame.shape), dtype=str(frame.dtype)) for name, frame in frames.items()}
    tokens = dict(cached_goal=saved['goal'], saved_grid_goal=saved['grid_goal'],
        fresh_single=encode_single(base).detach().cpu(),
        fresh_repeat=encode_single(base).detach().cpu(),
        fresh_grid_single=encode_single(frames['grid_batch_goal']).detach().cpu())
    repeated = encode_batch(np.stack([base]*8)).detach().cpu()
    tokens['repeated_image_batch8'] = repeated[offset]
    tokens['grid_batch8'] = encode_batch(mixed).detach().cpu()[offset]
    pairs = dict(
        fresh_single_vs_cached_goal=('fresh_single', 'cached_goal'),
        fresh_grid_batch_vs_saved_grid_goal=('grid_batch8', 'saved_grid_goal'),
        fresh_single_vs_saved_grid_goal=('fresh_single', 'saved_grid_goal'),
        fresh_grid_batch_vs_cached_goal=('grid_batch8', 'cached_goal'),
        fresh_single_vs_fresh_repeat=('fresh_single', 'fresh_repeat'),
        single_vs_repeated_image_batch8=('fresh_single', 'repeated_image_batch8'),
        single_vs_grid_batch8=('fresh_single', 'grid_batch8'),
        identical_vs_mixed_batch_goal_slot=('repeated_image_batch8', 'grid_batch8'),
        single_vs_grid_route_single=('fresh_single', 'fresh_grid_single'))
    comparisons = {name: comparison(tokens[a], tokens[b]) for name, (a, b) in pairs.items()}
    comparisons['within_identical_image_batch8'] = comparison(repeated[0].expand_as(repeated), repeated)
    return dict(pixel_comparisons=pixels, frames=frame_meta, token_comparisons=comparisons,
                primitive_interactions=0), dict(frames={k: torch.from_numpy(v) for k, v in frames.items()}, tokens=tokens)


def arithmetic_settings():
    return dict(matmul_precision=torch.get_float32_matmul_precision(),
                cuda_matmul_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                cudnn_tf32=bool(torch.backends.cudnn.allow_tf32))


@contextmanager
def strict_fp32():
    before = arithmetic_settings()
    try:
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        # The legacy flag aliases the precision setting on some torch versions;
        # restore it first so it cannot overwrite a saved 'medium' precision.
        torch.backends.cuda.matmul.allow_tf32 = before['cuda_matmul_tf32']
        torch.set_float32_matmul_precision(before['matmul_precision'])
        torch.backends.cudnn.allow_tf32 = before['cudnn_tf32']


def run(curriculum, clean, output, encoder_checkpoint, dino_root, *, device='cuda'):
    root, clean, output = (Path(p).resolve() for p in (curriculum, clean, output))
    if any(output == p or p in output.parents or output in p.parents for p in (root, clean)):
        raise ValueError('audit output must be separate from both source directories')
    if output.exists() and (any(output.iterdir()) or output.is_symlink()):
        raise ValueError('use a new empty audit output directory')
    if any(path.is_symlink() for path in output.parents):
        raise ValueError('audit output cannot use symlinked ancestors')
    saved = load_saved(root, clean)
    source_paths = [root / p for p in ('cache/goal.pt', 'cache/goal.json', 'cache/grid.pt',
                    'cache/grid.json', 'experiment.json', 'execution.json')]
    source_paths.append(clean / 'run_0/goal_tokens.pt')
    hashes = {str(path): file_sha256(path) for path in source_paths}
    encoder_sha = file_sha256(encoder_checkpoint)
    if encoder_sha != saved['config']['sources']['models']['encoder']:
        raise ValueError('encoder checkpoint differs from the saved experiment')
    if (git_output(dino_root, 'rev-parse', 'HEAD') != DINO_COMMIT
            or git_output(dino_root, 'status', '--porcelain', '--untracked-files=no')):
        raise ValueError('DINO-WM must be the unmodified pinned source')
    print('SAVED CACHE CHECKS\n' + json.dumps(saved['report'], indent=2), flush=True)
    target_device = torch.device(device)
    encoder = visual.load_frozen_ijepa_encoder(encoder_checkpoint, target_device)
    env = WallAdapter(EnvironmentConfig(env='wall', wall_layouts=[[32, 30]], dino_wm_root=str(dino_root)), 100)
    # Reuse the experiment's real methods, without constructing a runtime that
    # would write runtime.json or load the world model.
    runtime = SimpleNamespace(device=target_device, precision='fp32',
        args=SimpleNamespace(encoder_batch_size=8), load_encoder=lambda: encoder)
    one = lambda frame: visual.encode_frame(encoder, frame, target_device, 'fp32')
    many = lambda frames: PilotRuntime.encode_frames(runtime, frames)
    modes, payloads = {}, {}
    try:
        for mode in ('current_fp32', 'strict_fp32'):
            def capture():
                print(f'PROBING {mode}: ' + json.dumps(arithmetic_settings()), flush=True)
                report, payload = probe(saved, env, one, many)
                report['arithmetic'] = arithmetic_settings()
                return report, payload
            if mode == 'strict_fp32':
                with strict_fp32():
                    modes[mode], payloads[mode] = capture()
            else:
                modes[mode], payloads[mode] = capture()
    finally:
        env.close()
    cross = {name: comparison(payloads['current_fp32']['tokens'][name], payloads['strict_fp32']['tokens'][name])
             for name in ('fresh_single', 'repeated_image_batch8', 'grid_batch8')}
    if any(file_sha256(path) != sha for path, sha in hashes.items()):
        raise ValueError('source artifacts changed while auditing')
    report = dict(schema_version=1, saved=saved['report'], probes=modes, current_vs_strict=cross,
        encoder_sha256=encoder_sha, torch_version=str(torch.__version__),
        device=str(target_device), device_name=torch.cuda.get_device_name(target_device)
        if target_device.type == 'cuda' else str(target_device),
        cuda_version=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
        audit_source_commit=git_output(Path(__file__).resolve().parents[2], 'rev-parse', 'HEAD'),
        source_hashes_unchanged=True, source_hashes=hashes, primitive_interactions=0,
        note='Diagnostic only. No target/cache replacement, planning or training. Original cached goal pixels were not saved; fresh-pixel agreement cannot establish their historical identity.')
    output.mkdir(parents=True, exist_ok=True)
    save_tensors(output / 'probe_tensors.pt', payloads)
    write_json(output / 'report.json', report)
    for mode, data in modes.items():
        print('\n' + mode.upper(), flush=True)
        for name, values in data['pixel_comparisons'].items():
            print(f"pixels {name}: exact={values['exact']}; max difference={values['max_abs']:.0f}", flush=True)
        for name, values in data['token_comparisons'].items():
            print(f"tokens {name}: MSE={values['mse']:.9g}; max difference={values['max_abs']:.9g}", flush=True)
    print('CURRENT VS STRICT\n' + json.dumps(cross, indent=2), flush=True)
    print('Report:', output / 'report.json', flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('curriculum-dir', 'clean-dir', 'output-dir', 'encoder-checkpoint', 'dino-wm-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cuda')
    args = parser.parse_args(argv)
    run(args.curriculum_dir, args.clean_dir, args.output_dir, args.encoder_checkpoint,
        args.dino_wm_root, device=args.device)


if __name__ == '__main__':
    main()
