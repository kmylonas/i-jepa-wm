"""Independent original/clean adapter curricula over contiguous downward bands."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from src.mylonas_ijepa_wm import curriculum_multiband_protocol as protocol
from src.mylonas_ijepa_wm import curriculum_heatmaps as heatmaps
from src.mylonas_ijepa_wm import curriculum_initial_fit as initial_fit
from src.mylonas_ijepa_wm.curriculum_protocol import digest, write_json
from src.mylonas_ijepa_wm.curriculum_replay import eligible_pairs, fit_adapter, load_adapter, save_adapter
from src.mylonas_ijepa_wm.curriculum_runtime import (
    EXPECTED_ENCODER, EXPECTED_WM, PilotRuntime, file_sha256, interaction_budget, save_tensors,
)


prepare = protocol.prepare


def summarize(records):
    return dict(successes=sum(bool(row["success"]) for row in records), episodes=len(records),
                mean_final_distance=float(np.mean([row["final_goal_distance"] for row in records])),
                primitive_steps=sum(row["micro_steps"] for row in records),
                failed_case_ids=[row["case_id"] for row in records if not row["success"]])


def case_results(records):
    return [dict(case=row["case"], case_id=row["case_id"], success=row["success"],
                 final_goal_distance=row["final_goal_distance"], primitive_steps=row["micro_steps"],
                 achieved_positions=row["achieved_positions"]) for row in records]


def compare(before, after):
    a, b = ({row["case_id"]: row for row in rows} for rows in (before, after))
    if (len(a) != len(before) or len(b) != len(after) or a.keys() != b.keys()
            or any(a[key]["case"] != b[key]["case"] for key in a)):
        raise ValueError("paired curriculum comparisons require identical cases and CEM seeds")
    wins = [key for key in a if not a[key]["success"] and b[key]["success"]]
    losses = [key for key in a if a[key]["success"] and not b[key]["success"]]
    competent = summarize(before)["successes"] >= 18
    return dict(wins=len(wins), losses=len(losses), won_case_ids=wins, lost_case_ids=losses,
                success_change=len(wins)-len(losses), reference_competent=competent,
                retention_passed=len(losses) <= 2 if competent else None)


class Curriculum:
    def __init__(self, parent, clean, adjacent_dir, output):
        self.parent, self.clean, self.adjacent, self.root = protocol.safe_paths(parent, clean, adjacent_dir, output)
        self.config, self.manifest = protocol.load(self.parent, self.clean, self.adjacent, self.root)
        self.goal, self.goal_sha = protocol.load_goal(self.root, self.config)
        self.near_replay = protocol.read(self.clean / "ablation.json")["replay_files"]

    def guard(self):
        protocol.safe_paths(self.parent, self.clean, self.adjacent, self.root)
        if (protocol.read(self.root / "experiment.json") != self.config
                or protocol.read(self.root / "manifest.json") != self.manifest):
            raise ValueError("curriculum configuration changed during execution")
        for key in ("original", "clean", "adjacent"):
            source = self.config["sources"][key]
            for name, sha in source["files"].items():
                if file_sha256(Path(source["directory"]) / name) != sha:
                    raise ValueError("frozen curriculum source hash changed")
        for item in self.near_replay:
            if file_sha256(self.parent / item["path"]) != item["sha256"]:
                raise ValueError("near source replay hash changed")
        protocol.load_goal(self.root, self.config)
        if any((self.root / f"cache/grid.{suffix}").exists() for suffix in ("pt", "json")):
            self.require_grid()

    def attach(self, runtime, recipe):
        self.guard()
        root = protocol.runtime_root(self.root, recipe)
        if Path(runtime.root).resolve() != root or runtime.manifest != self.manifest:
            raise ValueError("runtime root or curriculum manifest differs")
        protocol.check_runtime(self.root, recipe, self.config, self.manifest, runtime.runtime_id)
        runtime.goal = self.goal.to(runtime.device).clone()
        return root

    def identity(self, recipe, version, sha):
        return dict(recipe=recipe, version=version, checkpoint_sha256=sha, goal_sha256=self.goal_sha,
                    experiment_digest=digest(self.config))

    def grid_signature(self):
        return dict(experiment_digest=digest(self.config), goal_sha256=self.goal_sha,
                    execution=protocol.read(self.root / "execution.json"),
                    preprocessing="preprocess_frames/frozen_ijepa/224/v1")

    def grid(self, runtime):
        self.attach(runtime, "raw")
        grid = heatmaps.prepare_grid(runtime, self.root, self.goal, self.grid_signature())
        return dict(points=len(grid["points"]), grid_sha256=grid["sha256"],
                    display_max=grid["display_max"], diagnostic_primitive_interactions=0)

    def require_grid(self):
        if not (self.root / "cache/grid.json").exists():
            raise ValueError("run the fixed grid stage before curriculum branches")
        return heatmaps.load_grid(self.root, self.grid_signature())

    def heatmap(self, recipe, version, adapter, sha):
        entries = self.replay_entries(recipe, version)
        positions = []
        for item in entries:
            saved = torch.load(item["path"], map_location="cpu", weights_only=True)
            positions.append(torch.as_tensor(saved["states"]).clone())
        return heatmaps.checkpoint_heatmap(self.root, recipe, version, adapter, torch.cat(positions).numpy(),
                                          self.grid_signature(), checkpoint_sha256=sha)

    def reference(self, recipe, version):
        if recipe not in protocol.RECIPES or version not in range(4):
            raise ValueError("invalid recipe or adapter version")
        if version == 0:
            if initial_fit.enabled(self):
                return initial_fit.reference(self, recipe)
            source = self.config["sources"][recipe]
            return Path(source["directory"]) / "run_0/adapter.pt", source["adapter_sha256"], source["runtime_id"]
        root = protocol.runtime_root(self.root, recipe)
        status = protocol.read(root / f"updates/v{version}/status.json")
        path = root / f"checkpoints/v{version}.pt"
        if status.get("status") != "complete" or status.get("checkpoint_sha256") != file_sha256(path):
            raise ValueError("curriculum checkpoint incomplete or hash differs")
        rid = protocol.check_runtime(self.root, recipe, self.config, self.manifest)
        return path, status["checkpoint_sha256"], rid

    def checkpoint(self, recipe, version, device="cpu"):
        path, sha, rid = self.reference(recipe, version)
        adapter, saved = load_adapter(path, device, expected_runtime_id=rid)
        if version == 0 and initial_fit.enabled(self):
            initial_fit.validate_loaded(self, recipe, saved)
        if version:
            root = protocol.runtime_root(self.root, recipe)
            status = protocol.read(root / f"updates/v{version}/status.json")
            inputs = status["inputs"]
            _, incoming_sha, _ = self.reference(recipe, version-1)
            if (inputs.get("recipe") != recipe or inputs.get("version") != version
                    or inputs.get("incoming_sha256") != incoming_sha
                    or inputs.get("goal_sha256") != self.goal_sha
                    or inputs.get("grid_sha256") != protocol.read(self.root / "cache/grid.json")["sha256"]
                    or inputs.get("fitting") != self.manifest["fitting"]
                    or inputs.get("seed") != 20261106 + version
                    or inputs.get("pair_margin") != self.config["pair_rules"][recipe]
                    or saved["provenance"].get("inputs_digest") != digest(inputs)
                    or saved["provenance"].get("experiment_digest") != digest(self.config)
                    or status.get("training_sha256") != file_sha256(root / f"updates/v{version}/training.json")):
                raise ValueError("curriculum checkpoint lineage/provenance differs")
            if inputs["replay"] != self.replay_entries(recipe, version):
                raise ValueError("curriculum training replay hash changed")
        return adapter, saved, sha

    def rows(self, recipe, version, band, *, optional=False):
        root = protocol.runtime_root(self.root, recipe)
        rid = protocol.check_runtime(self.root, recipe, self.config, self.manifest)
        sha = None if recipe == "raw" else self.reference(recipe, version)[1]
        method = protocol.optional_stage if optional else protocol.completed_stage
        return method(root, f"eval/v{version}/{band}", self.manifest["evaluation"][band], rid,
                      self.identity(recipe, version, sha))

    def evaluate(self, recipe, version, band, runtime, adapter=None, sha=None):
        stage = f"eval/v{version}/{band}"
        target = runtime.goal

        @torch.no_grad()
        def raw_score(tokens):
            return (tokens-target.unsqueeze(0)).square().mean(dim=(1, 2))

        return protocol.execute_stage(runtime.root, stage, self.manifest["evaluation"][band], runtime.runtime_id,
            self.identity(recipe, version, sha), lambda case: runtime.run_case(case, metric=adapter,
                journal_stage=stage, **({"latent_cost_fn": raw_score} if recipe == "raw" else {})))

    def baseline(self, runtime):
        self.attach(runtime, "raw")
        # Preflight all four bands before any missing baseline case is executed.
        for band in protocol.BANDS:
            protocol.preflight(runtime.root, f"eval/v0/{band}", self.manifest["evaluation"][band],
                               runtime.runtime_id, self.identity("raw", 0, None))
        return {band: summarize(self.evaluate("raw", 0, band, runtime)) for band in protocol.BANDS}

    def require_raw(self):
        for band in protocol.BANDS:
            self.rows("raw", 0, band)

    def initial(self, recipe, runtime):
        self.attach(runtime, recipe)
        self.require_grid()
        self.require_raw()
        root = protocol.runtime_root(self.root, recipe)
        if (root / 'stopped.json').exists():
            return protocol.read(root / 'stopped.json')
        if initial_fit.enabled(self):
            stopped = initial_fit.ensure(self, recipe, runtime)
            if stopped is not None:
                return stopped
        adapter, _, sha = self.checkpoint(recipe, 0, runtime.device)
        result = summarize(self.evaluate(recipe, 0, "near", runtime, adapter, sha))
        self.heatmap(recipe, 0, adapter, sha)
        return result

    def collection_rows(self, recipe, index):
        root = protocol.runtime_root(self.root, recipe)
        rid = protocol.check_runtime(self.root, recipe, self.config, self.manifest)
        sha = self.reference(recipe, index-1)[1]
        band = protocol.BANDS[index]
        return protocol.completed_stage(root, f"collect/{band}", self.manifest["collection"][band], rid,
                                        self.identity(recipe, index-1, sha), capture=True)

    def ensure_replay(self, runtime, recipe, record):
        case = record["case"]
        path = Path(runtime.root) / "replay/run_0" / f"{case['case_id']}.pt"
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if (saved.get("runtime_id") != runtime.runtime_id or saved.get("case") != case
                    or saved.get("recipe") != recipe or saved.get("raw_sha256") != record["raw_sha256"]):
                raise ValueError("cached curriculum replay provenance differs")
            return path
        raw = torch.load(Path(runtime.root) / record["raw_path"], map_location="cpu", weights_only=True)
        if (raw.get("runtime_id") != runtime.runtime_id or raw.get("case") != case or not raw.get("success")
                or len(raw["frames"]) != record["micro_steps"]+1
                or len(raw["states"]) != len(raw["frames"]) or len(raw["primitive_steps"]) != len(raw["frames"])
                or raw["macro_durations"] != record["macro_durations"]):
            raise ValueError("collection raw provenance/timing differs")
        tokens = runtime.encode_frames(raw["frames"]).half()
        save_tensors(path, dict(runtime_id=runtime.runtime_id, case=case, case_id=case["case_id"],
            split="train", success=True, recipe=recipe, raw_sha256=record["raw_sha256"],
            states=raw["states"], primitive_steps=raw["primitive_steps"], latents=tokens,
            latent_storage_dtype="float16"))
        return path

    def replay_entries(self, recipe, version):
        entries = [dict(band="near", case_id=item["case_id"], path=str(self.parent / item["path"]),
                        sha256=item["sha256"]) for item in self.near_replay]
        root = protocol.runtime_root(self.root, recipe)
        for index in range(1, version+1):
            band = protocol.BANDS[index]
            for record in self.collection_rows(recipe, index):
                if record["success"]:
                    path = root / "replay/run_0" / f"{record['case_id']}.pt"
                    entries.append(dict(band=band, case_id=record["case_id"], path=str(path), sha256=file_sha256(path)))
        return entries

    def load_replay(self, recipe, version, entries):
        replay = []
        root = protocol.runtime_root(self.root, recipe)
        rid = protocol.check_runtime(self.root, recipe, self.config, self.manifest)
        near_cases = {case["case_id"]: case for case in protocol.read(self.parent / "manifest.json")["training"]["0"]}
        own_cases = {case["case_id"]: case for band in protocol.BANDS[1:version+1]
                     for case in self.manifest["collection"][band]}
        for item in entries:
            if file_sha256(item["path"]) != item["sha256"]:
                raise ValueError("curriculum replay hash changed")
            saved = torch.load(item["path"], map_location="cpu", weights_only=True)
            near = item["band"] == "near"
            expected_case = (near_cases if near else own_cases)[item["case_id"]]
            if (saved.get("case") != expected_case or saved.get("case_id") != item["case_id"]
                    or saved.get("runtime_id") != (self.config["sources"]["original"]["runtime_id"] if near else rid)
                    or saved.get("split") != "train" or not saved.get("success")
                    or (not near and saved.get("recipe") != recipe)):
                raise ValueError("curriculum replay recipe/case provenance differs")
            replay.append(saved)
        return replay

    def stopped(self, recipe, index, reason):
        result = dict(status="stopped", recipe=recipe, version=index, reason=reason)
        root = protocol.runtime_root(self.root, recipe)
        write_json(root / "stopped.json", result)
        return result

    def band(self, recipe, index, runtime):
        if index not in range(1, 4):
            raise ValueError("curriculum band index must be 1, 2 or 3")
        root = self.attach(runtime, recipe)
        grid = self.require_grid()
        grid_sha = grid["sha256"]
        del grid
        self.require_raw()
        if (root / "stopped.json").exists():
            return protocol.read(root / "stopped.json")
        self.rows(recipe, 0, "near")
        if index > 1:
            self.checkpoint(recipe, index-1)
            completion = protocol.read(root / f"updates/v{index-1}/band_complete.json")
            if completion.get("checkpoint_sha256") != self.reference(recipe, index-1)[1]:
                raise ValueError("previous band evaluation/diagnostics are incomplete")
            for previous_band in protocol.BANDS[:index]:
                self.rows(recipe, index-1, previous_band)
        adapter, _, incoming_sha = self.checkpoint(recipe, index-1, runtime.device)
        band, stage = protocol.BANDS[index], f"collect/{protocol.BANDS[index]}"
        incoming_identity = self.identity(recipe, index-1, incoming_sha)
        protocol.preflight(root, f"eval/v{index-1}/{band}", self.manifest["evaluation"][band],
                           runtime.runtime_id, incoming_identity)
        protocol.preflight(root, stage, self.manifest["collection"][band], runtime.runtime_id,
                           incoming_identity, capture=True)
        status_path = root / f"updates/v{index}/status.json"
        checkpoint_path = root / f"checkpoints/v{index}.pt"
        done = status_path.exists() and protocol.read(status_path).get("status") == "complete"
        if done:
            updated, _, updated_sha = self.checkpoint(recipe, index, runtime.device)
            for key in protocol.BANDS[:index+1]:
                protocol.preflight(root, f"eval/v{index}/{key}", self.manifest["evaluation"][key],
                                   runtime.runtime_id, self.identity(recipe, index, updated_sha))
        elif any((root / "episodes" / f"eval/v{index}/{key}").exists() for key in protocol.BANDS[:index+1]):
            raise ValueError("post-update records exist without a committed checkpoint")
        self.evaluate(recipe, index-1, band, runtime, adapter, incoming_sha)

        def collect(case):
            record = runtime.run_case(case, metric=adapter, capture=True, journal_stage=stage)
            record["raw_sha256"] = file_sha256(root / record["raw_path"])
            return record

        collection = protocol.execute_stage(root, stage, self.manifest["collection"][band], runtime.runtime_id,
                                             incoming_identity, collect, capture=True)
        for record in collection:
            if record["success"]:
                self.ensure_replay(runtime, recipe, record)
        entries = self.replay_entries(recipe, index)
        inputs = dict(recipe=recipe, version=index, incoming_sha256=incoming_sha, goal_sha256=self.goal_sha,
                      grid_sha256=grid_sha,
                      fitting=self.manifest["fitting"], seed=20261106+index,
                      pair_margin=self.config["pair_rules"][recipe], replay=entries)
        if done:
            status = protocol.read(status_path)
            if status["inputs"] != inputs:
                raise ValueError("completed curriculum update inputs changed")
        else:
            if status_path.exists() and protocol.read(status_path).get("inputs") != inputs:
                raise ValueError("interrupted curriculum update inputs changed")
            replay = self.load_replay(recipe, index, entries)
            new_pairs = sum(len(eligible_pairs(row["states"], row["primitive_steps"], goal=row["case"]["goal"],
                min_goal_progress=inputs["pair_margin"])[1]) for row in replay if row["case"]["band"] == band)
            if not new_pairs:
                heatmaps.plot_update(self.root, recipe, index, {"history": []}, collection)
                return self.stopped(recipe, index, "no usable current-band pairs from successful collection")
            write_json(status_path, dict(status="training", inputs=inputs))
            try:
                updated, training = fit_adapter(replay, self.goal, seed=inputs["seed"], device=runtime.device,
                    initial_adapter=adapter, balance_bands=True, min_goal_progress=inputs["pair_margin"],
                    **inputs["fitting"], progress=lambda row: print(
                        f"{recipe}/{band} update {row['update']} loss={row['loss']:.5f}", flush=True)
                        if row["update"] == 1 or row["update"] % 50 == 0 else None)
            except (ValueError, RuntimeError) as error:
                if "nonfinite" in str(error).lower() or "non-finite" in str(error).lower():
                    return self.stopped(recipe, index, str(error))
                raise
            updated.eval().requires_grad_(False)
            provenance = dict(runtime_id=runtime.runtime_id, experiment_digest=digest(self.config), inputs_digest=digest(inputs))
            save_adapter(checkpoint_path, updated, training, provenance)
            training_path = root / f"updates/v{index}/training.json"
            write_json(training_path, {key: value for key, value in training.items() if key != "optimizer"})
            updated_sha = file_sha256(checkpoint_path)
            status = dict(status="complete", inputs=inputs, checkpoint_sha256=updated_sha,
                          training_sha256=file_sha256(training_path),
                          collection_successes=sum(bool(row["success"]) for row in collection),
                          collection_attempts=len(collection), new_usable_pairs=new_pairs,
                          old_episodes=sum(row["band"] != band for row in entries),
                          new_episodes=sum(row["band"] == band for row in entries))
            write_json(status_path, status)
            del replay
        for key in protocol.BANDS[:index+1]:
            self.evaluate(recipe, index, key, runtime, updated, updated_sha)
        self.heatmap(recipe, index, updated, updated_sha)
        heatmaps.plot_update(self.root, recipe, index,
            protocol.read(root / f"updates/v{index}/training.json"), collection)
        write_json(root / f"updates/v{index}/band_complete.json", {"checkpoint_sha256": updated_sha, "status": "complete"})
        return status

    def branch_report(self, recipe, raw):
        root = protocol.runtime_root(self.root, recipe)
        versions, comparisons, updates, collections, per_case = {}, {}, {}, {}, {}
        for version in range(4):
            versions[str(version)], comparisons[str(version)] = {}, {}
            per_case[str(version)] = {}
            initial_ready = not initial_fit.enabled(self) or initial_fit.completed(root)
            exists = (root / "runtime.json").exists() and ((version == 0 and initial_ready) or
                version > 0 and
                (root / f"updates/v{version}/status.json").exists() and
                protocol.read(root / f"updates/v{version}/status.json").get("status") == "complete")
            if exists:
                self.checkpoint(recipe, version)
            for band in protocol.BANDS:
                rows = self.rows(recipe, version, band, optional=True) if exists else None
                versions[str(version)][band] = summarize(rows) if rows is not None else None
                per_case[str(version)][band] = case_results(rows) if rows is not None else None
                if rows is not None:
                    block = {"vs_raw": compare(raw[band], rows) if raw[band] is not None else None}
                    earlier = self.rows(recipe, version-1, band, optional=True) if version else None
                    block["vs_incoming"] = compare(earlier, rows) if earlier is not None else None
                    comparisons[str(version)][band] = block
            if version and (root / f"updates/v{version}/status.json").exists():
                updates[str(version)] = protocol.read(root / f"updates/v{version}/status.json")
        rid = protocol.check_runtime(self.root, recipe, self.config, self.manifest) if (root / "runtime.json").exists() else None
        for index, band in enumerate(protocol.BANDS[1:], 1):
            stage_path = root / "stages" / f"collect/{band}.json"
            if rid and stage_path.exists():
                sha = self.reference(recipe, index-1)[1]
                records = protocol.optional_stage(root, f"collect/{band}", self.manifest["collection"][band], rid,
                    self.identity(recipe, index-1, sha), capture=True)
                collections[band] = summarize(records) if records is not None else None
            else:
                collections[band] = None
        state = "not_started" if all(row is None for rows in versions.values() for row in rows.values()) else "active"
        if (root / "stopped.json").exists():
            state = "stopped"
        elif (root / "updates/v3/band_complete.json").exists():
            state = "complete"
        return dict(status=state, versions=versions, comparisons=comparisons, updates=updates,
            collections=collections, per_case=per_case,
            stopped=protocol.read(root / "stopped.json") if state == "stopped" else None)

    def report_budget(self, recipe):
        root = protocol.runtime_root(self.root, recipe)
        try:
            if not (root / "runtime.json").exists():
                if (root / "interactions.jsonl").exists() or any((root / "episodes").rglob("*.json")):
                    raise ValueError("interaction accounting has no runtime identity")
                return dict(status="not_started", primitive_interactions=0, count_is_lower_bound=False)
            rid = protocol.check_runtime(self.root, recipe, self.config, self.manifest)
            return protocol.validated_budget(root, rid)
        except (ValueError, OSError, KeyError) as error:
            return dict(status="unavailable", primitive_interactions=None,
                        count_is_lower_bound=True, error=str(error))

    def report(self, *, render=False):
        self.guard()
        raw_error = None
        try:
            raw_root = protocol.runtime_root(self.root, "raw")
            raw = {band: self.rows("raw", 0, band, optional=True)
                   if (raw_root / "runtime.json").exists() else None for band in protocol.BANDS}
        except (ValueError, OSError, KeyError) as error:
            raw, raw_error = dict.fromkeys(protocol.BANDS), str(error)
        branches = {}
        for recipe in protocol.RECIPES:
            try:
                branch = self.branch_report(recipe, raw)
            except (ValueError, OSError, KeyError) as error:
                empty = lambda: {str(version): dict.fromkeys(protocol.BANDS) for version in range(4)}
                branch = dict(status="invalid", error=str(error), versions=empty(), per_case=empty(),
                              comparisons={}, updates={}, collections=dict.fromkeys(protocol.BANDS[1:]), stopped=None)
            branch["budget"] = self.report_budget(recipe)
            branches[recipe] = branch
        raw_budget = self.report_budget("raw")
        budgets = [raw_budget, *(branch["budget"] for branch in branches.values())]
        known = sum(budget["primitive_interactions"] for budget in budgets
                    if budget["primitive_interactions"] is not None)
        unavailable = any(budget["primitive_interactions"] is None for budget in budgets)
        result = dict(schema_version=1, experiment=self.config, bounds=self.manifest["bounds"],
                      raw={key: summarize(rows) if rows is not None else None for key, rows in raw.items()},
                      raw_per_case={key: case_results(rows) if rows is not None else None for key, rows in raw.items()},
                      raw_budget=raw_budget, raw_error=raw_error, branches=branches,
                      total_primitive_interactions=None if unavailable else known,
                      known_primitive_interactions=known,
                      total_count_is_lower_bound=any(budget["count_is_lower_bound"] for budget in budgets),
                      diagnostic_primitive_interactions=0,
                      note="Repeated monitoring cases; operational screens are not significance claims. Each branch learns from its own policy's successes.")
        write_json(self.root / "report.json", result)
        if render:
            heatmaps.plot_curriculum_report(self.root, self.manifest, result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "baseline", "grid", "initial", "band", "report"])
    for name in ("parent-dir", "clean-dir", "adjacent-dir", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--recipe", choices=protocol.RECIPES)
    parser.add_argument("--band-index", type=int, choices=[1, 2, 3])
    parser.add_argument('--without-anchor', action='store_true', help='prepare: retrain v0 from identity and disable anchoring in all fits')
    for name in ("world-model-checkpoint", "encoder-checkpoint", "dino-wm-root"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--expected-world-model-sha256", default=EXPECTED_WM)
    parser.add_argument("--expected-encoder-sha256", default=EXPECTED_ENCODER)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16"], default="fp32")
    parser.add_argument("--encoder-batch-size", type=int, default=8)
    parser.set_defaults(run_seed=0, videos=False)
    args = parser.parse_args(argv)
    paths = (args.parent_dir, args.clean_dir, args.adjacent_dir, args.output_dir)
    if args.without_anchor and args.stage != 'prepare':
        raise ValueError('--without-anchor is a prepare option; later stages use the saved configuration')
    if args.stage == "prepare":
        result = prepare(*paths, without_anchor=args.without_anchor)
    else:
        curriculum = Curriculum(*paths)
        if args.stage == "report":
            result = curriculum.report(render=True)
        else:
            if any(getattr(args, name) is None for name in ("world_model_checkpoint", "encoder_checkpoint", "dino_wm_root")):
                raise ValueError("frozen model, encoder and pinned DINO paths are required")
            recipe = "raw" if args.stage in ("baseline", "grid") else args.recipe
            if recipe is None or args.stage == "band" and args.band_index is None:
                raise ValueError("recipe and band index are required for branch stages")
            args.output_dir = protocol.runtime_root(curriculum.root, recipe)
            args.runtime_signature_extra = protocol.variant(curriculum.config, recipe)
            runtime = PilotRuntime(args, curriculum.manifest)
            try:
                if args.stage == "baseline":
                    result = curriculum.baseline(runtime)
                elif args.stage == "grid":
                    result = curriculum.grid(runtime)
                elif args.stage == "initial":
                    result = curriculum.initial(recipe, runtime)
                else:
                    result = curriculum.band(recipe, args.band_index, runtime)
            finally:
                runtime.close()
    if args.stage == "prepare":
        console = dict(prepared=str(args.output_dir), evaluation_cases=80, collection_attempts_per_band=40,
                       maximum_scheduled_episodes=840, bands=protocol.BOUNDS)
    elif args.stage == "band":
        console = {key: value for key, value in result.items() if key != "inputs"}
        if "inputs" in result:
            console.update(recipe=result["inputs"]["recipe"], version=result["inputs"]["version"],
                           replay_episodes=len(result["inputs"]["replay"]))
    elif args.stage == "report":
        console = dict(raw=result["raw"], total_primitive_interactions=result["total_primitive_interactions"],
            branches={key: {field: row[field] for field in ("status", "versions", "collections", "budget", "stopped")}
                      for key, row in result["branches"].items()}, report=str(curriculum.root / "report.json"))
    else:
        console = result
    print(json.dumps(console, indent=2))
    return result


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        print(f"Curriculum stopped: {error}", file=sys.stderr)
        raise SystemExit(2)
