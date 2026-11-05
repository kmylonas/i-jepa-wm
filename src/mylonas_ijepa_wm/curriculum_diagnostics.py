"""Pilot figures and explicitly budgeted simulator candidate audits."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

from src.mylonas_ijepa_wm import plan_pointmaze as visual, plan_pointmaze_proprio as planner
from src.mylonas_ijepa_wm.cost_adapter import adapter_goal_cost, token_goal_cost
from src.mylonas_ijepa_wm.curriculum_protocol import load_stage, write_json
from src.mylonas_ijepa_wm.curriculum_runtime import file_sha256, InteractionJournal, interaction_budget
from src.mylonas_ijepa_wm.evaluate_wall_candidate_rankings import select_candidates
from src.mylonas_ijepa_wm.validate_planning_costs_proprio import _spearman


def replay_audit(runtime, case, candidates, metric=None, *, stage="diagnostics"):
    """Replay full horizons even after success, to match predicted terminal time."""
    goal = runtime.goal_tokens()
    frames, outcomes = [], []
    interactions = 0
    for candidate in candidates:
        runtime.env.reset(case["seed"], layout=case["layout"], start=case["start"], goal=case["goal"])
        reached_goal = False
        actions = np.asarray(candidate["actions"], dtype=np.float32)
        with InteractionJournal(runtime.root, runtime.runtime_id, stage, case["case_id"]) as journal:
            for step, action in enumerate(actions.reshape(-1, 2), start=1):
                outcome = runtime.env.step(action)
                journal.record(step, action)
                reached_goal |= bool(outcome.success)
                interactions += 1
        frames.append(runtime.env.render().copy())
        outcomes.append(dict(source=candidate["source"], predicted_cost=candidate["predicted_cost"],
                             actions=actions.tolist(), reached_goal=reached_goal,
                             terminal_state=np.asarray(outcome.state).tolist(),
                             actual_distance=float(np.linalg.norm(np.asarray(outcome.state) - case["goal"]))))
    tokens = runtime.encode_frames(frames).to(runtime.device)
    with torch.no_grad():
        costs = ((tokens - goal).square().mean(dim=(1, 2)) if metric is None
                 else token_goal_cost(metric, tokens, goal))
    for row, cost in zip(outcomes, costs.cpu().tolist()):
        row["true_cost"] = cost
    return outcomes, interactions


def _ranking_summary(rows):
    return dict(candidates=len(rows),
                predicted_vs_real_rho=_spearman([row["predicted_cost"] for row in rows], [row["true_cost"] for row in rows]),
                real_vs_distance_rho=_spearman([row["true_cost"] for row in rows], [row["actual_distance"] for row in rows]))


def run_diagnostics(runtime, adapter, run_seed, sample_count):
    root = runtime.root / "diagnostics" / f"run_{run_seed}"
    root.mkdir(parents=True, exist_ok=True)
    goal = runtime.goal_tokens()
    x_values, y_values = np.arange(40, 55, 2), np.arange(10, 55, 2)
    points = np.array([(x, y) for y in y_values for x in x_values], dtype=np.float32)
    frames = np.stack([runtime.env._render_at(point) for point in points])
    tokens = runtime.encode_frames(frames)
    raw_costs, adapted_costs = [], []
    with torch.no_grad():
        for chunk in tokens.split(16):
            chunk = chunk.to(runtime.device)
            raw_costs.extend((chunk - goal).square().mean(dim=(1, 2)).cpu().tolist())
            adapted_costs.extend(token_goal_cost(adapter, chunk, goal).cpu().tolist())
    distances = np.linalg.norm(points - np.array([50, 42]), axis=1)
    write_json(root / "real_cost_map.json", dict(runtime_id=runtime.runtime_id,
                samples=[dict(position=point.tolist(), distance=float(distance), raw=raw, adapted=adapted)
                         for point, distance, raw, adapted in zip(points, distances, raw_costs, adapted_costs)],
                raw_distance_rho=_spearman(raw_costs, distances), adapted_distance_rho=_spearman(adapted_costs, distances)))
    figure, axes = plt.subplots(1, 2, figsize=(9, 6), constrained_layout=True)
    for axis, title, costs in zip(axes, ("Raw real-frame cost", "Adapted real-frame cost"), (raw_costs, adapted_costs)):
        graphic = axis.pcolormesh(x_values, y_values, np.array(costs).reshape(len(y_values), -1), shading="nearest")
        axis.scatter(50, 42, marker="*", s=100, color="red")
        axis.set(title=title, xlabel="x", ylabel="y", aspect="equal")
        figure.colorbar(graphic, ax=axis)
    figure.savefig(root / "real_cost_map.png", dpi=150)
    plt.close(figure)

    runtime.load_world_model()
    p, metadata = runtime.manifest["planning"], runtime.metadata
    adapter_sha = file_sha256(runtime.root / f"run_{run_seed}/adapter.pt")
    counts = min(sample_count, p["population"])
    audit_files = []
    for band in ("near", "far"):
        case = next(item for item in runtime.manifest["evaluation"] if item["band"] == band)
        for label, metric in (("raw", None), ("adapted", adapter)):
            path = root / f"audit_{label}_{case['case_id']}.json"
            signature = dict(runtime_id=runtime.runtime_id, adapter_sha256=adapter_sha,
                             samples_per_subset=counts, case=case, cost=label)
            if path.exists():
                saved = json.loads(path.read_text())
                if any(saved.get(key) != value for key, value in signature.items()):
                    raise ValueError("candidate audit provenance mismatch")
                audit_files.append(saved)
                continue
            initial = runtime.env.reset(case["seed"], layout=case["layout"], start=case["start"], goal=case["goal"])
            latent = visual.encode_frame(runtime.encoder, initial.frame, runtime.device, runtime.precision)
            state = torch.tensor(initial.state, device=runtime.device)
            state = (state - metadata["state_mean"].to(runtime.device)) / metadata["state_std"].to(runtime.device)
            rounds = []

            def observe(iteration, actions, costs):
                sample = (actions.detach().cpu(), costs.detach().cpu())
                if iteration == 0:
                    rounds.append(sample)
                if len(rounds) == 1:
                    rounds.append(sample)
                else:
                    rounds[-1] = sample

            result = planner.plan_action(
                runtime.model, latent.unsqueeze(0), state.unsqueeze(0), torch.zeros(0, 10, device=runtime.device),
                goal, torch.full((10,), -torch.inf, device=runtime.device), torch.full((10,), torch.inf, device=runtime.device),
                runtime.device, p["planning_horizon"], p["population"], p["num_elites"], p["cem_iterations"], p["candidate_batch_size"],
                action_mean=metadata.get("action_mean"), action_std=metadata.get("action_std"),
                precision=runtime.precision, min_std=p["min_std"],
                generator=visual._create_generator(runtime.device, case["seed"]),
                max_action_length=p["max_action_length"], cem_selection=p["cem_selection"],
                candidate_observer=observe,
                latent_cost_fn=adapter_goal_cost(metric, goal) if metric is not None else None,
            )
            candidates = select_candidates(rounds, counts, counts, case["seed"])
            candidates.append(dict(source="returned_elite_mean", actions=result.action_sequence.detach().cpu().numpy(), predicted_cost=result.cost))
            rows, interactions = replay_audit(runtime, case, candidates, metric,
                                              stage=f"run_{run_seed}/diagnostics/{label}/{case['case_id']}")
            saved = dict(**signature, primitive_interactions=interactions, rows=rows,
                         subsets={source: _ranking_summary([row for row in rows if row["source"] == source])
                                  for source in ("initial_random", "final_elite", "returned_elite_mean")})
            write_json(path, saved)
            audit_files.append(saved)
            print(f"audit {label}/{band}: {interactions} extra primitive interactions", flush=True)
    budget = interaction_budget(runtime.root, runtime.runtime_id)
    diagnostic_steps = sum(count for stage, count in budget["by_stage"].items()
                           if stage.startswith(f"run_{run_seed}/diagnostics/"))
    result = dict(runtime_id=runtime.runtime_id, adapter_sha256=adapter_sha,
                  extra_diagnostic_primitive_interactions=diagnostic_steps,
                  completed_audit_primitive_interactions=sum(item["primitive_interactions"] for item in audit_files),
                  journal_count_is_lower_bound=budget["count_is_lower_bound"],
                  note="Separate from adaptation budget; full terminal horizons, no early success truncation.")
    write_json(root / "summary.json", result)
    return result


def _draw_layout(axis):
    axis.add_patch(Rectangle((29, 7), 6, 19, color="black", alpha=.4))
    axis.add_patch(Rectangle((29, 34), 6, 23, color="black", alpha=.4))
    axis.scatter(50, 42, marker="*", c="red", s=120, zorder=5)
    axis.set(xlim=(25, 57), ylim=(7, 57), xlabel="x", ylabel="y", aspect="equal")


def plot_report(root, manifest, report):
    root = Path(root)
    raw = load_stage(root, "baseline", manifest["evaluation"], report["runtime_id"])
    if not report["runs"]:
        return
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    seeds = list(report["runs"])
    for axis, band in zip(axes, ("near", "far")):
        values = [report["runs"][seeds[0]][band]["raw_successes"] / report["runs"][seeds[0]][band]["episodes"]]
        values += [report["runs"][seed][band]["adapted_successes"] / report["runs"][seed][band]["episodes"] for seed in seeds]
        axis.bar(["Raw"] + [f"Run {seed}" for seed in seeds], values)
        axis.set(title=band, ylim=(0, 1), ylabel="Success fraction (20 cases by default)")
    figure.savefig(root / "success_by_band.png", dpi=150)
    plt.close(figure)
    for seed in seeds:
        adapted = load_stage(root, f"run_{seed}/evaluate", manifest["evaluation"], report["runtime_id"])
        figure, axes = plt.subplots(1, 2, figsize=(10, 6), constrained_layout=True)
        for axis, band in zip(axes, ("near", "far")):
            _draw_layout(axis)
            for label, records, color in (("Raw", raw, "#666666"), ("Adapted", adapted, "#1b9e77")):
                for index, record in enumerate(item for item in records if item["case"]["band"] == band):
                    path = np.asarray(record["achieved_positions"])
                    axis.plot(path[:, 0], path[:, 1], color=color, alpha=.45, label=label if index == 0 else None)
            axis.set_title(f"{band}, adapter run {seed}")
            axis.legend()
        figure.savefig(root / f"run_{seed}/trajectories.png", dpi=150)
        plt.close(figure)
        history_path = root / f"run_{seed}/training.json"
        if history_path.exists():
            history = json.loads(history_path.read_text())["history"]
            figure, axis = plt.subplots(figsize=(7, 3), constrained_layout=True)
            for key in ("rank_loss", "anchor_loss"):
                axis.plot([row["update"] for row in history], [row[key] for row in history], label=key)
            axis.set(xlabel="Adapter update", ylabel="Loss")
            axis.legend()
            figure.savefig(root / f"run_{seed}/training_loss.png", dpi=150)
            plt.close(figure)
