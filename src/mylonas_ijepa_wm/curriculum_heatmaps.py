"""Fixed real-image diagnostic grid, immutable cost maps and curriculum figures."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.mylonas_ijepa_wm.cost_adapter import token_goal_cost
from src.mylonas_ijepa_wm.curriculum_protocol import digest, write_json
from src.mylonas_ijepa_wm.curriculum_runtime import file_sha256, save_tensors
from src.mylonas_ijepa_wm.curriculum_multiband_protocol import BANDS, BOUNDS, RECIPES, read


def physical_bounds(env):
    """Rollout border walls, distinct from the more conservative reset bounds."""
    config = getattr(env, 'wall_config', None)
    if config is None:
        return float(env.border_padding)+.001, float(env.img_size-1-env.border_padding)-.001
    return float(config.border_wall_loc-1)+.001, float(env.img_size-config.border_wall_loc)-.001


def points(right_edge=54.):
    # No-argument geometry is the immutable legacy cache layout.
    if not np.isfinite(right_edge) or not 54 <= right_edge <= 64:
        raise ValueError('invalid right diagnostic boundary')
    xs = np.arange(40, right_edge, 2).tolist()
    if not xs or abs(xs[-1]-right_edge)>1e-5:
        xs.append(right_edge)
    return torch.tensor([(x,y) for y in range(8,51,2) for x in xs],dtype=torch.float32)


def _safe(root):
    root = Path(root)
    if root.exists() and any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("diagnostic output cannot contain descendant symlinks")
    return root


def load_grid(output, signature):
    root = _safe(output)
    path, meta_path = root / "cache/grid.pt", root / "cache/grid.json"
    meta = read(meta_path)
    domain = meta.get('domain')
    locations = points(domain['right_edge']) if domain else points()
    if (domain and domain.get('kind') != 'physical_right_v1'
            or meta.get("signature") != signature or meta.get("points_digest") != digest(locations.tolist())):
        raise ValueError("grid signature/provenance differs")
    if meta.get("sha256") != file_sha256(path):
        raise ValueError("diagnostic grid cache hash differs")
    grid = torch.load(path, map_location="cpu", weights_only=True)
    tokens, raw, goal = grid["tokens"], grid["raw_costs"], grid["goal"]
    if (grid.get("signature") != signature or grid.get('domain') != domain
            or not torch.equal(grid["points"], locations)
            or tokens.dtype != torch.float32 or tokens.ndim != 3 or len(tokens) != len(locations)
            or goal.shape != tokens.shape[1:] or raw.shape != (len(locations),)
            or not all(torch.isfinite(tensor).all() for tensor in (tokens, raw, goal))
            or float(raw.max()) <= 0 or meta.get("display_max") != float(raw.max())):
        raise ValueError("diagnostic grid shape/values/provenance differs")
    grid["sha256"] = meta["sha256"]
    grid["display_max"] = meta["display_max"]
    return grid


def prepare_grid(runtime, output, goal, signature):
    root = _safe(output)
    if (root / "cache/grid.json").exists():
        grid = load_grid(root, signature)
        if not torch.equal(grid["goal"], goal.detach().cpu().float()):
            raise ValueError("grid goal differs from shared frozen target")
        return grid
    runtime.env.reset(0, layout=[32, 30], start=[44, 34], goal=[50, 42])
    lower, upper = physical_bounds(runtime.env.env)
    locations = points(upper)
    if float(locations.min()) < lower-1e-5 or float(locations.max()) > upper+1e-5:
        raise ValueError("diagnostic grid exceeds valid pinned environment reset bounds")
    frames = np.stack([runtime.env._render_at(point) for point in locations])
    # PilotRuntime.encode_frames chunks its encoder calls to the fixed batch size.
    tokens = runtime.encode_frames(frames).detach().cpu().float()
    goal = goal.detach().cpu().float().clone()
    if tokens.ndim != 3 or tokens.shape[1:] != goal.shape or not torch.isfinite(tokens).all():
        raise ValueError("nonfinite or misaligned diagnostic tokens")
    raw = torch.cat([(chunk-goal).square().mean(dim=(1, 2)) for chunk in tokens.split(16)])
    if not torch.isfinite(raw).all() or float(raw.max()) <= 0:
        raise ValueError("invalid diagnostic raw scale")
    path = root / "cache/grid.pt"
    domain = dict(kind='physical_right_v1',right_edge=upper)
    save_tensors(path, dict(signature=signature,domain=domain, points=locations, tokens=tokens, goal=goal, raw_costs=raw))
    write_json(root / "cache/grid.json", dict(signature=signature, points_digest=digest(locations.tolist()),
        domain=domain,sha256=file_sha256(path), display_max=float(raw.max()), primitive_interactions=0,
        note="Real-image rendering probes; excluded from training and progression decisions."))
    return load_grid(root, signature)


def checkpoint_heatmap(output, recipe, version, adapter, replay_positions, signature, *, checkpoint_sha256):
    root = _safe(output)
    if recipe not in RECIPES or version not in range(4):
        raise ValueError("unknown heatmap recipe/version")
    grid = load_grid(root, signature)
    positions = np.asarray(replay_positions, dtype=np.float32).reshape(-1, 2)
    if not np.isfinite(positions).all():
        raise ValueError("nonfinite replay overlay positions")
    expected = dict(recipe=recipe, version=version, checkpoint_sha256=checkpoint_sha256,
        grid_sha256=grid["sha256"], signature=signature, replay_positions_digest=digest(positions.tolist()))
    folder = root / f"branches/{recipe}/heatmaps"
    data_path, image_path, commit_path = (folder / f"v{version}.{suffix}" for suffix in ("json", "png", "commit.json"))
    if commit_path.exists():
        commit = read(commit_path)
        if (commit.get("inputs") != expected or commit.get("data_sha256") != file_sha256(data_path)
                or commit.get("image_sha256") != file_sha256(image_path)):
            raise ValueError("completed heatmap hash/provenance differs")
        return read(data_path)
    device = next(iter(adapter.parameters()), torch.empty(0)).device
    with torch.no_grad():
        adapted = torch.cat([token_goal_cost(adapter, chunk.to(device), grid["goal"].to(device)).cpu()
                             for chunk in grid["tokens"].split(16)])
    if not torch.isfinite(adapted).all():
        raise ValueError("nonfinite adapted grid costs")
    raw = grid["raw_costs"]
    difference = adapted-raw
    limit = grid["display_max"]
    data = dict(**expected, points=grid["points"].tolist(), raw=raw.tolist(), adapted=adapted.tolist(),
        difference=difference.tolist(), display_max=limit, difference_limit=limit,
        adapted_max=float(adapted.max()), difference_min=float(difference.min()), difference_max=float(difference.max()),
        saturated_adapted_points=int((adapted > limit).sum()),
        saturated_difference_points=int((difference.abs() > limit).sum()),
        primitive_interactions=0, note="Real rendered image costs, not predicted-terminal costs. Values saved without clipping.")
    folder.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 7), constrained_layout=True)
    for axis, title, values, is_difference in zip(axes,
            ("Raw latent MSE", "Adapted latent MSE", "Adapted minus raw"),
            (raw.numpy(), adapted.numpy(), difference.numpy()), (False, False, True)):
        xs, ys = (np.unique(grid['points'][:,i].numpy()) for i in (0,1))
        graphic = axis.pcolormesh(xs, ys, values.reshape(len(ys),len(xs)),
            shading="nearest", cmap="RdBu_r" if is_difference else "viridis",
            vmin=-limit if is_difference else 0, vmax=limit)
        for boundary in (8, 14, 22, 30, 36):
            axis.axhline(boundary, color="white", alpha=.65, linestyle="--", linewidth=.6)
        if version:
            lo, hi = BOUNDS[BANDS[version]]
            axis.axhspan(lo, hi, facecolor="none", edgecolor="#f97316", linewidth=2)
        if len(positions):
            axis.scatter(positions[:, 0], positions[:, 1], s=2, color="white", alpha=.12, edgecolors="none")
        axis.scatter(50, 42, marker="*", s=100, c="#f97316", edgecolors="black", linewidths=.5)
        axis.set(title=title, xlabel="x", ylabel="y", xlim=(39, float(xs[-1])+1), ylim=(7, 51), aspect="equal")
        extension = "both" if is_difference else "max" if np.max(values) > limit else "neither"
        figure.colorbar(graphic, ax=axis, shrink=.8, extend=extension)
    figure.suptitle(f"{recipe}: adapter v{version} — fixed real-image grid\n"
        f"White points: training replay; orange outline: latest band. Display max={limit:.4g}", fontsize=11)
    temporary = image_path.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=140)
    plt.close(figure)
    temporary.replace(image_path)
    write_json(data_path, data)
    write_json(commit_path, dict(inputs=expected, data_sha256=file_sha256(data_path), image_sha256=file_sha256(image_path)))
    print(f"heatmap {recipe}/v{version}: max adapted={data['adapted_max']:.5g}; "
          f"above shared display scale={data['saturated_adapted_points']}/{len(raw)}; "
          f"difference range=[{data['difference_min']:.5g},{data['difference_max']:.5g}], "
          f"outside difference scale={data['saturated_difference_points']}/{len(raw)}", flush=True)
    return data


def plot_curriculum_report(output, manifest, report):
    root = _safe(output)
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    action_max = max([1] + [cell["primitive_steps"] for recipe in RECIPES
        for row in report["branches"][recipe]["versions"].values() for cell in row.values() if cell is not None])
    for row_index, recipe in enumerate(RECIPES):
        for column, field, title in ((0, "successes", "Successes / 20"), (1, "primitive_steps", "Primitive actions")):
            matrix = np.full((4, 4), np.nan)
            for version in range(4):
                for band_index, band in enumerate(BANDS):
                    cell = report["branches"][recipe]["versions"][str(version)][band]
                    if cell is not None:
                        matrix[version, band_index] = cell[field]
            axis = axes[row_index, column]
            cmap = plt.get_cmap("viridis" if column == 0 else "Blues").with_extremes(bad="#e2e8f0")
            axis.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0, vmax=20 if column == 0 else action_max, aspect="auto")
            for vi in range(4):
                for bi in range(4):
                    value = matrix[vi, bi]
                    dark = not np.isnan(value) and (value > action_max*.6 if column else value < 12)
                    axis.text(bi, vi, "N/A" if np.isnan(value) else str(int(value)), ha="center", va="center",
                              color="white" if dark else "black")
            axis.set(title=f"{recipe}: {title}", yticks=range(4), yticklabels=[f"adapter v{i}" for i in range(4)],
                     xticks=range(4), xticklabels=["near\n30–36", "band 1\n22–30", "band 2\n14–22", "band 3\n8–14"])
    figure.suptitle("Current and earlier bands: N/A means no completed evaluation; cases repeat across versions.", fontsize=11)
    figure.savefig(root / "curriculum_matrix.png", dpi=150)
    plt.close(figure)
    for recipe in RECIPES:
        for version in range(1, 4):
            _plot_evaluations(root, recipe, version, report)


def _plot_evaluations(root, recipe, version, report):
    branch = report["branches"][recipe]
    bands = [band for band in BANDS[:version+1] if branch["per_case"][str(version)][band] is not None]
    if not bands:
        return
    figure, axes = plt.subplots(len(bands), 3, figsize=(10, 3.4*len(bands)), squeeze=False, constrained_layout=True)
    for band_index, band in enumerate(bands):
        choices = (report["raw_per_case"][band], branch["per_case"][str(version-1)][band],
                   branch["per_case"][str(version)][band])
        for column, (label, records) in enumerate(zip(("Raw", f"Incoming v{version-1}", f"Updated v{version}"), choices)):
            axis = axes[band_index, column]
            lo, hi = BOUNDS[band]
            axis.axhspan(lo, hi, color="#93c5fd", alpha=.2)
            axis.scatter(50, 42, marker="*", c="black", s=60, zorder=5)
            successes, actions = 0, 0
            if records is not None:
                for record in records:
                    path = np.asarray(record["achieved_positions"])
                    axis.plot(path[:, 0], path[:, 1], color="#16a34a" if record["success"] else "#dc2626", alpha=.4)
                    successes += bool(record["success"])
                    actions += record["primitive_steps"]
                title = f"{band} — {label}: {successes}/{len(records)}, {actions} actions"
            else:
                title = f"{band} — {label}: N/A"
            axis.set(title=title, xlabel="x", ylabel="y", xlim=(36, 62), ylim=(5, 56), aspect="equal")
            axis.title.set_fontsize(9)
    folder = root / f"branches/{recipe}/updates/v{version}"
    folder.mkdir(parents=True, exist_ok=True)
    figure.suptitle(f"{recipe}: paired evaluation trajectories, macro-boundary observations", fontsize=11)
    figure.savefig(folder / "evaluation_paths.png", dpi=140)
    plt.close(figure)


def plot_update(output, recipe, index, training, collection):
    root = _safe(output)
    folder = root / f"branches/{recipe}/updates/v{index}"
    folder.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for key in ("rank_loss", "anchor_loss"):
        axes[0].plot([row["update"] for row in training["history"]], [row[key] for row in training["history"]], label=key)
    axes[0].set(xlabel="Optimizer update", ylabel="Loss", title=f"{recipe}, update in {BANDS[index]}")
    if training["history"]:
        axes[0].legend()
    else:
        axes[0].text(.5, .5, "No usable new pairs; no fitting", ha="center", transform=axes[0].transAxes)
    for record in collection:
        path = np.asarray(record["achieved_positions"])
        axes[1].plot(path[:, 0], path[:, 1], color="#16a34a" if record["success"] else "#dc2626", alpha=.4)
    lo, hi = BOUNDS[BANDS[index]]
    axes[1].axhspan(lo, hi, color="#93c5fd", alpha=.2)
    axes[1].scatter(50, 42, marker="*", c="black", s=70)
    axes[1].set(xlabel="x", ylabel="y", title="Actual collection paths (macro observations)",
                xlim=(36, 62), ylim=(5, 52), aspect="equal")
    figure.savefig(folder / "training_and_experience.png", dpi=140)
    plt.close(figure)
