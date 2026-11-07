"""Frozen model runtime and real-observation capture for the pilot."""

from datetime import datetime, timezone
import errno
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

import numpy as np
import torch

from src.mylonas_ijepa_wm import plan_pointmaze as visual, plan_proprio, wm, wm_proprio
from src.mylonas_ijepa_wm.cost_adapter import adapter_goal_cost
from src.mylonas_ijepa_wm.curriculum_protocol import digest, write_json
from src.mylonas_ijepa_wm.embed_frames import preprocess_frames
from src.mylonas_ijepa_wm.environment_adapters import WallAdapter
from src.mylonas_ijepa_wm.environment_configs import EnvironmentConfig


EXPECTED_WM = "b0f3ccfefe0534d9f51af2d0767619913f48d0c323ebb323f0bcd4b800206c3f"
EXPECTED_ENCODER = "3f37d8a7572e25c2dce9a3072c2fd8cd55ce32dc6edb92a8996b0e4cb140bf98"
DINO_COMMIT = "0a9492fa12044b852ae9e001cc74604b79c8bb0c"


def file_sha256(path):
    path = Path(path)
    for delay in (1, 3, None):
        result = hashlib.sha256()
        try:
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    result.update(block)
            return result.hexdigest()
        except OSError as error:
            if error.errno != errno.EIO:
                raise
            if delay is None:
                raise OSError(errno.EIO, "SHA-256 file read failed after three attempts", str(path)) from error
            print(f"I/O error reading {path}; retrying SHA-256 from the beginning in {delay}s",
                  file=sys.stderr, flush=True)
            time.sleep(delay)


def source_digest(root):
    root = Path(root)
    files = sorted(root.rglob("*.py"))
    if not files:
        raise ValueError(f"no Python source under {root}")
    return digest([(str(path.relative_to(root)), file_sha256(path)) for path in files])


def git_output(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def repository_provenance(root):
    root = Path(root)
    if (root / ".git").exists():
        return dict(source_commit=git_output(root, "rev-parse", "HEAD"),
                    source_dirty=bool(git_output(root, "status", "--porcelain")), source_kind="git")
    archive = json.loads((root / "pilot_source.json").read_text())
    return dict(source_commit=archive["commit"], source_kind="archive",
                source_dirty=archive["source_digest"] != source_digest(root / "src"))


def save_tensors(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


class InteractionJournal:
    """Line-flushed primitive actions survive interrupted, later repeated attempts."""

    def __init__(self, root, runtime_id, stage, case_id):
        self.path = Path(root) / "interactions.jsonl"
        self.identity = dict(runtime_id=runtime_id, stage=stage, case_id=case_id, attempt_id=uuid.uuid4().hex)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Separate any truncated final line from a previous process crash.
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open("rb") as stream:
                stream.seek(-1, 2)
                needs_newline = stream.read(1) != b"\n"
            if needs_newline:
                with self.path.open("ab") as stream:
                    stream.write(b"\n")
        self.stream = self.path.open("a", buffering=1)
        self._write(event="start")
        return self

    def _write(self, **data):
        self.stream.write(json.dumps({**self.identity, **data}) + "\n")

    def record(self, step, action):
        self._write(event="primitive", step=int(step), action=np.asarray(action).tolist())

    def __exit__(self, exception_type, exception, traceback):
        self._write(event="finish", status="complete" if exception_type is None else "interrupted")
        self.stream.close()


def interaction_budget(root, runtime_id):
    path = Path(root) / "interactions.jsonl"
    result = dict(primitive_interactions=0, by_stage={}, interrupted_attempts=0, incomplete_journal_lines=0)
    started, completed = set(), set()
    if not path.exists():
        return result
    with path.open() as stream:
        for line in stream:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                result["incomplete_journal_lines"] += 1
                continue
            if entry["runtime_id"] != runtime_id:
                raise ValueError("interaction journal provenance mismatch")
            if entry["event"] == "start":
                started.add(entry["attempt_id"])
            elif entry["event"] == "primitive":
                result["primitive_interactions"] += 1
                stage = entry["stage"]
                result["by_stage"][stage] = result["by_stage"].get(stage, 0) + 1
            elif entry.get("status") == "complete":
                completed.add(entry["attempt_id"])
    result["interrupted_attempts"] = len(started - completed)
    result["count_is_lower_bound"] = result["incomplete_journal_lines"] > 0
    return result


class PilotRuntime:
    def __init__(self, args, manifest):
        self.args, self.manifest, self.root = args, manifest, Path(args.output_dir)
        self.device = wm.resolve_device(args.device)
        self.precision = wm.resolve_precision(args.precision, self.device)
        self.dino_root = Path(args.dino_wm_root).resolve()
        if git_output(self.dino_root, "rev-parse", "HEAD") != DINO_COMMIT:
            raise ValueError("DINO-WM checkout is not the pinned commit")
        if git_output(self.dino_root, "status", "--porcelain", "--untracked-files=no"):
            raise ValueError("pinned DINO-WM source has tracked modifications")
        identities = dict(world_model=file_sha256(args.world_model_checkpoint),
                          encoder=file_sha256(args.encoder_checkpoint))
        if identities["world_model"] != args.expected_world_model_sha256:
            raise ValueError("world-model checkpoint SHA256 differs from selected pilot artifact")
        if identities["encoder"] != args.expected_encoder_sha256:
            raise ValueError("encoder checkpoint SHA256 differs from selected pilot artifact")
        repo = Path(__file__).resolve().parents[2]
        signature = dict(manifest=digest(manifest), models=identities,
                         source=source_digest(repo / "src"), dino_source=source_digest(self.dino_root / "env/wall"),
                         dino_commit=DINO_COMMIT, torch_version=str(torch.__version__),
                         device_type=self.device.type, precision=self.precision,
                         encoder_batch_size=args.encoder_batch_size)
        runtime_variant = getattr(args, "runtime_signature_extra", None)
        if runtime_variant is not None:
            if not isinstance(runtime_variant, dict):
                raise ValueError("runtime signature extension must be a dictionary")
            signature["runtime_variant"] = runtime_variant
        self.runtime_id = digest(signature)
        path = self.root / "runtime.json"
        if path.exists() and json.loads(path.read_text())["runtime_id"] != self.runtime_id:
            raise ValueError("runtime provenance changed; use a new output directory")
        if not path.exists():
            write_json(path, dict(runtime_id=self.runtime_id, signature=signature,
                                  **repository_provenance(repo),
                                  created_utc=datetime.now(timezone.utc).isoformat(),
                                  world_model_path=str(args.world_model_checkpoint), encoder_path=str(args.encoder_checkpoint)))
        self.config = EnvironmentConfig(env="wall", wall_layouts=[[32, 30]],
                                        dino_wm_root=str(self.dino_root))
        self.env = WallAdapter(self.config, manifest["planning"]["max_macro_steps"] * 5)
        self.encoder = self.model = self.metadata = self.goal = None

    def load_encoder(self):
        if self.encoder is None:
            self.encoder = visual.load_frozen_ijepa_encoder(self.args.encoder_checkpoint, self.device)
        return self.encoder

    def load_world_model(self):
        if self.model is None:
            self.model, self.metadata, _ = wm_proprio.load_model_and_metadata(self.args.world_model_checkpoint, self.device)
            shape = self.metadata["model_configuration"]
            if (self.metadata.get("environment") != "wall"
                    or self.metadata.get("layout_splits", {}).get("train") != [[32, 30]]
                    or shape["num_patches"] != 256 or shape["ijepa_dim"] != 1280
                    or shape["action_dim"] != 10 or shape["proprio_dim"] != 2
                    or shape["num_hist"] != 1):
                raise ValueError("pilot requires the selected fixed-layout 256x1280, history-1 Wall model")
        return self.model

    def goal_tokens(self):
        if self.goal is None:
            self.env.reset(0, layout=(32, 30), start=(44, 34), goal=(50, 42))
            self.goal = visual.encode_frame(self.load_encoder(), self.env.render_goal(), self.device, self.precision).clone()
        return self.goal

    @torch.no_grad()
    def encode_frames(self, frames):
        encoder = self.load_encoder()
        encoded = []
        for chunk in torch.as_tensor(frames).split(self.args.encoder_batch_size):
            images = preprocess_frames(chunk).to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.precision == "bf16"):
                encoded.append(encoder(images).float().cpu())
        return torch.cat(encoded)

    def run_case(self, case, *, metric=None, capture=False, cost="latent", latent_cost_fn=None,
                 journal_stage=None):
        if latent_cost_fn is not None and (metric is not None or cost != "latent"):
            raise ValueError("explicit latent scorer requires latent cost without an adapter")
        self.load_world_model()
        self.load_encoder()
        # Compute/cache goal before case reset, since goal_tokens itself resets.
        scorer = adapter_goal_cost(metric, self.goal_tokens()) if metric is not None else latent_cost_fn
        initial = self.env.reset(case["seed"], layout=case["layout"], start=case["start"], goal=case["goal"])
        states, frames, steps = [], [], []

        def observe(step, outcome, action):
            if step:
                journal.record(step, action)
            if capture:
                steps.append(step)
                states.append(outcome.state.copy())
                frames.append(outcome.frame.copy())

        planning_args = SimpleNamespace(**self.manifest["planning"], cost=cost)
        video_frames = [] if self.args.videos else None
        stage = (f"run_{self.args.run_seed}/collect" if capture else
                 f"run_{self.args.run_seed}/evaluate" if metric is not None else
                 "position" if cost == "position" else "baseline")
        if journal_stage is not None:
            stage = journal_stage
        with InteractionJournal(self.root, self.runtime_id, stage, case["case_id"]) as journal:
            record = plan_proprio.run_episode(
                self.env, self.config, self.encoder, self.model, self.metadata,
                planning_args, self.device, self.precision, case["seed"], video_frames,
                initial=initial, latent_cost_fn=scorer, observation_observer=observe,
            )
        record["attempt_id"] = journal.identity["attempt_id"]
        record["raw_path"] = None
        if capture:
            path = Path("raw") / f"run_{case['run_seed']}" / f"{case['case_id']}.pt"
            save_tensors(self.root / path, dict(
                runtime_id=self.runtime_id, case=case, success=record["success"],
                frames=torch.from_numpy(np.stack(frames)), states=torch.tensor(np.array(states)),
                primitive_steps=torch.tensor(steps, dtype=torch.int64),
                primitive_actions=torch.tensor(record["primitive_actions"], dtype=torch.float32).reshape(-1, 2),
                macro_durations=record["macro_durations"],
            ))
            record["raw_path"] = str(path)
        if video_frames:
            condition = "adapted" if metric is not None else cost
            video_run = self.args.run_seed if metric is not None else case["run_seed"]
            record["video_path"] = str(visual.save_episode_video(
                video_frames, self.root / "videos" / condition / str(video_run),
                int(case["case_id"].split("_")[-1]), case["seed"], record["success"], 10))
        return record

    def encode_episode(self, record):
        case = record["case"]
        if case["split"] != "train" or record["runtime_id"] != self.runtime_id or not record["success"]:
            raise ValueError("only this pilot's successful training episodes may be encoded for replay")
        path = self.root / "replay" / f"run_{case['run_seed']}" / f"{case['case_id']}.pt"
        if path.exists():
            cached = torch.load(path, map_location="cpu", weights_only=True)
            if cached["runtime_id"] != self.runtime_id or cached["case"] != case:
                raise ValueError("replay provenance mismatch")
            return cached
        raw = torch.load(self.root / record["raw_path"], map_location="cpu", weights_only=True)
        if (raw["runtime_id"] != self.runtime_id or raw["case"] != case or not raw["success"]
                or len(raw["frames"]) != record["micro_steps"] + 1):
            raise ValueError("raw training episode provenance/timing mismatch")
        cached = dict(runtime_id=self.runtime_id, case=case, case_id=case["case_id"],
                      split="train", success=True, states=raw["states"], primitive_steps=raw["primitive_steps"],
                      latents=self.encode_frames(raw["frames"]).half(), latent_storage_dtype="float16")
        save_tensors(path, cached)
        return cached

    def close(self):
        self.env.close()
