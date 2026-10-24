"""Run the released DINO-WM Wall model on our saved failed start–goal pairs.

The search matches the released Wall CEM settings. Candidate trajectories are
evaluated in chunks so the comparison can also run on smaller GPUs or MPS.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torchvision import transforms

from src.mylonas_ijepa_wm.environment_adapters import WallAdapter
from src.mylonas_ijepa_wm.environment_configs import EnvironmentConfig


REPO = Path(__file__).resolve().parents[2]
REFERENCE = REPO / "data/dino_wm_reference"
DEFAULT_CASES = (REPO / "src/mylonas_ijepa_wm/wall/wall_fixed_2_hist1"
                 / "world_models/k1/planning_position_app5.json")


def load_failed_cases(path: Path) -> list[dict]:
    """Select the saved failures by their exact initial and goal coordinates."""
    episodes = json.loads(Path(path).read_text())["episodes"]
    cases = []
    for episode in episodes:
        if not episode["success"]:
            cases.append({
                "seed": int(episode["seed"]),
                "start": episode["achieved_positions"][0],
                "goal": episode["goal_position"],
            })
    if not cases or len({case["seed"] for case in cases}) != len(cases):
        raise ValueError("Expected at least one failed episode with unique seeds")
    return cases


def _add_dino_source(source_root: Path) -> None:
    source = str(Path(source_root).resolve())
    if source not in sys.path:
        sys.path.insert(0, source)


def load_dino_model(checkpoint_path: Path | None = None,
                    source_root: Path | None = None):
    """Load the authors' released ViT-S/14 and Wall world-model checkpoint."""
    source_root = Path(source_root or REFERENCE / "source")
    checkpoint_path = Path(checkpoint_path or REFERENCE / "outputs/wall_single/checkpoints/model_latest.pth")
    _add_dino_source(source_root)
    from models.visual_world_model import VWorldModel

    class DinoEncoder(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.base_model = backbone
            self.name = "dinov2_vits14"
            self.emb_dim = backbone.num_features
            self.patch_size = backbone.patch_size

        def forward(self, image):
            return self.base_model.forward_features(image)["x_norm_patchtokens"]

    hub_checkout = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    if hub_checkout.is_dir():
        backbone = torch.hub.load(str(hub_checkout), "dinov2_vits14", source="local")
    else:
        backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = VWorldModel(
        image_size=224, num_hist=1, num_pred=1,
        encoder=DinoEncoder(backbone),
        proprio_encoder=checkpoint["proprio_encoder"],
        action_encoder=checkpoint["action_encoder"],
        decoder=checkpoint["decoder"], predictor=checkpoint["predictor"],
        proprio_dim=10, action_dim=10, concat_dim=1,
        num_action_repeat=1, num_proprio_repeat=1,
        train_encoder=False, train_predictor=False, train_decoder=False,
    )
    model.eval()
    return model, checkpoint["epoch"]


def place_model(model, device: str):
    """Move the released model, including its unregistered attention masks."""
    model.to(device)
    for module in model.modules():
        if type(module).__module__ == "models.vit" and type(module).__name__ == "Attention":
            module.bias = module.bias.to(device)
    return model.eval()


def make_wall_adapter(seed: int, source_root: Path | None = None) -> WallAdapter:
    config = EnvironmentConfig(env="wall", dino_wm_root=str(source_root or REFERENCE / "source"))
    adapter = WallAdapter(config, max_micro_steps=100)
    adapter.reset(seed)
    return adapter


def render_observation_at(adapter: WallAdapter, position) -> torch.Tensor:
    """Return the released DINO-WM preprocessing of a 65-pixel Wall render."""
    _add_dino_source(Path(adapter.config.dino_wm_root))
    from datasets.img_transforms import default_transform

    dot = adapter.env._render_dot(torch.as_tensor(position, dtype=torch.float32))
    raw = adapter.env.channels_to_img(adapter.env.wall_img, dot).float()
    resized = transforms.Resize((224, 224))(raw)
    return default_transform(224)(resized / 255.0)


def rollout_encoded(model, encoded_obs: dict, actions: torch.Tensor) -> dict:
    """DINO-WM rollout from one encoded observation, avoiding repeated ViT calls."""
    if model.num_hist != 1 or model.concat_dim != 1:
        raise ValueError("This comparison requires the released Wall model (history=1, concat_dim=1)")
    batch, horizon, action_dim = actions.shape
    if horizon < 1 or action_dim != model.action_encoder.in_chans:
        raise ValueError("Expected at least one 10-dimensional macro action")
    visual = encoded_obs["visual"].expand(batch, -1, -1, -1)
    proprio = encoded_obs["proprio"].expand(batch, -1, -1)
    patches = visual.shape[2]
    proprio = proprio.unsqueeze(2).expand(-1, -1, patches, -1)
    first_action = model.encode_act(actions[:, :1]).unsqueeze(2).expand(-1, -1, patches, -1)
    z = torch.cat((visual, proprio, first_action), dim=-1)
    for step in range(horizon):
        z = model.predict(z)
        if step + 1 < horizon:
            z = model.replace_actions_from_z(z[:, -1:].clone(), actions[:, step + 1:step + 2])
    observations, _ = model.separate_emb(z[:, -1:])
    return {key: value[:, 0] for key, value in observations.items()}


def score_candidates(model, encoded_start: dict, encoded_goal: dict,
                     candidates: torch.Tensor, batch_size: int, alpha: float) -> torch.Tensor:
    target_visual = encoded_goal["visual"][:, -1]
    target_proprio = encoded_goal["proprio"][:, -1]
    costs = []
    with torch.inference_mode():
        for chunk in candidates.split(batch_size):
            predicted = rollout_encoded(model, encoded_start, chunk)
            visual = (predicted["visual"] - target_visual).square().mean(dim=(1, 2))
            proprio = (predicted["proprio"] - target_proprio).square().mean(dim=1)
            costs.append(visual + alpha * proprio)
    return torch.cat(costs)


def limit_normalized_actions(candidates: torch.Tensor, action_mean: torch.Tensor,
                             action_std: torch.Tensor, limit: float) -> torch.Tensor:
    """Constrain Wall actions in simulator units and return normalized actions."""
    shape = candidates.shape
    raw = candidates.reshape(*shape[:-1], 5, 2) * action_std + action_mean
    return ((raw.clamp(-limit, limit) - action_mean) / action_std).reshape(shape)


def cem_plan(model, encoded_start: dict, encoded_goal: dict, *,
             horizon: int, population: int, elites: int, iterations: int,
             candidate_batch_size: int, alpha: float, generator: torch.Generator,
             action_mean: torch.Tensor, action_std: torch.Tensor,
             action_limit: float | None):
    device = encoded_start["visual"].device
    mean = torch.zeros(horizon, 10, device=device)
    std = torch.ones_like(mean)
    best_cost = float("inf")
    for _ in range(iterations):
        candidates = torch.randn(population, horizon, 10, generator=generator, device=device)
        candidates = candidates * std + mean
        candidates[0] = mean
        if action_limit is not None:
            candidates = limit_normalized_actions(
                candidates, action_mean, action_std, action_limit
            )
        costs = score_candidates(model, encoded_start, encoded_goal, candidates,
                                 candidate_batch_size, alpha)
        indices = costs.argsort()[:elites]
        chosen = candidates[indices]
        mean = chosen.mean(dim=0)
        std = chosen.std(dim=0)
        best_cost = float(costs[indices[0]].item())
    return mean, best_cost


def evaluate_case(model, stats: dict, case: dict, args) -> dict:
    adapter = make_wall_adapter(case["seed"], args.source_root)
    initial = adapter.state()
    if not np.allclose(initial, case["start"], atol=1e-5):
        raise ValueError(f"Seed {case['seed']} reset to {initial}, expected {case['start']}")
    if not np.allclose(adapter.goal, case["goal"], atol=1e-5):
        raise ValueError(f"Seed {case['seed']} goal is {adapter.goal}, expected {case['goal']}")
    if adapter.layout != (32, 30):
        raise ValueError(f"Unexpected Wall layout: {adapter.layout}")

    device = next(model.parameters()).device
    state_mean = torch.tensor(stats["state_mean"], device=device)
    state_std = torch.tensor(stats["state_std"], device=device)
    action_mean = torch.tensor(stats["action_mean"], device=device)
    action_std = torch.tensor(stats["action_std"], device=device)
    goal = torch.tensor(case["goal"], device=device)
    goal_obs = {
        "visual": render_observation_at(adapter, case["goal"]).to(device)[None, None],
        "proprio": ((goal - state_mean) / state_std)[None, None],
    }
    with torch.inference_mode():
        encoded_goal = model.encode_obs(goal_obs)
    generator = torch.Generator(device=device).manual_seed(case["seed"])
    frames = [adapter.render_video()] if args.video_dir else None
    states = [initial.tolist()]
    macro_actions = []
    costs = []
    micro_steps = 0
    success = bool(np.linalg.norm(initial - np.asarray(case["goal"])) < 4.5)
    try:
        while len(macro_actions) < args.max_macro_steps and not success:
            position = adapter.state()
            start_obs = {
                "visual": render_observation_at(adapter, position).to(device)[None, None],
                "proprio": ((torch.as_tensor(position, device=device) - state_mean) / state_std)[None, None],
            }
            with torch.inference_mode():
                encoded_start = model.encode_obs(start_obs)
                plan, cost = cem_plan(
                    model, encoded_start, encoded_goal, horizon=args.horizon,
                    population=args.population, elites=args.elites,
                    iterations=args.iterations, candidate_batch_size=args.candidate_batch_size,
                    alpha=args.alpha, generator=generator,
                    action_mean=action_mean, action_std=action_std,
                    action_limit=args.action_limit,
                )
            costs.append(cost)
            for normalized_macro in plan[:min(args.actions_per_plan,
                                              args.max_macro_steps - len(macro_actions))]:
                micro_actions = normalized_macro.reshape(5, 2) * action_std + action_mean
                raw_macro = micro_actions.detach().cpu().numpy().astype(np.float32)
                for action in raw_macro:
                    outcome = adapter.step(action)
                    micro_steps += 1
                    if frames is not None:
                        frames.append(adapter.render_video())
                    success = outcome.success
                    if success:
                        break
                macro_actions.append(raw_macro.reshape(-1).tolist())
                states.append(adapter.state().tolist())
                if success:
                    break
            print(f"seed {case['seed']}: {len(macro_actions)}/{args.max_macro_steps} "
                  f"macro steps, goal distance {np.linalg.norm(adapter.state() - case['goal']):.2f}, "
                  f"success={success}", flush=True)
        video_path = None
        if frames is not None:
            import imageio.v2 as imageio
            args.video_dir.mkdir(parents=True, exist_ok=True)
            video_path = args.video_dir / f"dino_seed_{case['seed']}_{'success' if success else 'failed'}.mp4"
            imageio.mimsave(video_path, frames, fps=10)
        return {
            "seed": case["seed"], "start": case["start"], "goal_position": case["goal"],
            "success": success, "macro_steps": len(macro_actions), "micro_steps": micro_steps,
            "final_goal_distance": float(np.linalg.norm(adapter.state() - case["goal"])),
            "achieved_positions": states, "macro_actions": macro_actions,
            "planner_best_costs": costs, "video_path": str(video_path) if video_path else None,
        }
    finally:
        adapter.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--checkpoint", type=Path,
                        default=REFERENCE / "outputs/wall_single/checkpoints/model_latest.pth")
    parser.add_argument("--source-root", type=Path, default=REFERENCE / "source")
    parser.add_argument("--stats", type=Path, default=REFERENCE / "wall_stats.json")
    parser.add_argument("--output", type=Path, default=REFERENCE / "matched_wall_planning.json")
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--max-macro-steps", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--actions-per-plan", type=int, default=5)
    parser.add_argument("--population", type=int, default=300)
    parser.add_argument("--elites", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--candidate-batch-size", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--action-limit", type=float,
                        help="Optional symmetric limit in raw Wall action units; use 1 for our planner's range")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if min(args.max_macro_steps, args.horizon, args.actions_per_plan, args.population,
           args.elites, args.iterations, args.candidate_batch_size) < 1:
        parser.error("Planning counts must be positive")
    if args.elites > args.population or args.elites < 2:
        parser.error("Elites must be between 2 and population")
    if args.action_limit is not None and args.action_limit <= 0:
        parser.error("Action limit must be positive")
    return args


def settings_match(saved: dict, requested: dict) -> bool:
    saved = dict(saved)
    saved.setdefault("action_limit", None)
    return saved == requested


def main(argv=None):
    args = parse_args(argv)
    cases = load_failed_cases(args.cases)
    if args.seeds:
        cases = [case for case in cases if case["seed"] in set(args.seeds)]
    if not cases:
        raise ValueError("No matching failed cases")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    stats = json.loads(args.stats.read_text())
    model, epoch = load_dino_model(args.checkpoint, args.source_root)
    place_model(model, args.device)
    print(f"DINO-WM Wall checkpoint epoch {epoch}; device {args.device}; "
          f"{len(cases)} matched failed cases", flush=True)
    output = {"checkpoint": str(args.checkpoint), "checkpoint_epoch": epoch,
              "cases": str(args.cases), "settings": {
                  "horizon": args.horizon, "actions_per_plan": args.actions_per_plan,
                  "population": args.population, "elites": args.elites,
                  "iterations": args.iterations, "alpha": args.alpha,
                  "action_limit": args.action_limit,
                  "max_macro_steps": args.max_macro_steps,
              }, "episodes": []}
    existing = json.loads(args.output.read_text()) if args.output.is_file() else None
    if existing is not None:
        if not settings_match(existing["settings"], output["settings"]):
            raise ValueError("Existing output has different planning settings")
        output["episodes"] = existing["episodes"]
    completed = {row["seed"] for row in output["episodes"]}
    for case in cases:
        if case["seed"] in completed:
            continue
        result = evaluate_case(model, stats, case, args)
        output["episodes"].append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(output, indent=2))
        temporary.replace(args.output)
    selected = [row for row in output["episodes"] if row["seed"] in {case["seed"] for case in cases}]
    print(f"Matched cases: {sum(row['success'] for row in selected)}/{len(selected)} successes. "
          f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
