"""Rank sampled failed observations by their similarity to successful routes."""
from itertools import combinations

import numpy as np
import torch

from src.mylonas_ijepa_wm import experience_curriculum as experience
from src.mylonas_ijepa_wm.experience_half_square import _right_room
from src.mylonas_ijepa_wm.curriculum_multiband_protocol import read
from src.mylonas_ijepa_wm.curriculum_runtime import save_tensors
from src.mylonas_ijepa_wm.ranking_planning import _frame_hash


def eligible_observations(states, frames, case, design, goal_ids):
    xy = np.asarray(states)[:, :2]
    inside = _right_room(xy)
    hashes = experience._exclude_protected(xy, inside, dict(goal_id=goal_ids[case["goal_index"]]), design)
    return [int(i) for i in np.flatnonzero(inside) if _frame_hash(frames[i]) not in hashes]


def sample_frames(states, frames, case, design, goal_ids):
    unique = {}
    for i in eligible_observations(states, frames, case, design, goal_ids):
        unique.setdefault(_frame_hash(frames[i]), i)
    ids = list(unique.values())
    return [ids[j] for j in np.linspace(0, len(ids) - 1, min(5, len(ids)), dtype=int)]


def route_scores(failed, successful, *, top=3):
    """Score each query by the nearest observation from each of its top routes."""
    minima, indices = [], []
    for reference in successful:
        distances = torch.cdist(failed, reference, compute_mode="use_mm_for_euclid_dist").square() / failed.shape[-1]
        values, frames = distances.min(dim=1)
        minima.append(values.cpu())
        indices.append(frames.tolist())
    if len(minima) < top:
        return None
    values, routes = torch.stack(minima, dim=1).sort(dim=1, stable=True)
    matches = [[(route, indices[route][query]) for route in row[:top]]
               for query, row in enumerate(routes.tolist())]
    return values[:, :top].mean(dim=1).tolist(), matches


def ranking_pairs(scores, *, min_gap=1e-7):
    pairs = []
    for a, b in combinations(range(len(scores)), 2):
        delta = scores[a] - scores[b]
        if abs(delta) > min_gap:
            pairs.append((a, b) if delta > 0 else (b, a))
    return pairs


def _reference(runtime, record, design, goal_ids):
    case = record["case"]
    path = runtime.root / "route_references" / f"{case['case_id']}.pt"
    lineage = dict(runtime_id=runtime.runtime_id, case=case, raw_sha256=record["raw_sha256"])
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["inputs"] != lineage:
            raise ValueError(f"successful route provenance differs for {case['case_id']}")
        return saved
    raw = experience._read_replay_raw(runtime, record)
    if not raw["success"] or len(raw["frames"]) != record["micro_steps"] + 1:
        raise ValueError("expected the full observations of a successful rollout")
    ids = eligible_observations(raw["states"], raw["frames"], case, design, goal_ids)
    saved = dict(inputs=lineage, positions=raw["states"][ids, :2].tolist(),
                 steps=raw["primitive_steps"][ids].tolist(),
                 tokens=runtime.encode_frames(raw["frames"][ids]).half() if ids else None)
    save_tensors(path, saved)
    return saved


@torch.no_grad()
def label_failures(runtime, records, design, manifest, adapter, index):
    """Generate new labels with the incoming cost and freeze them in local replay."""
    failures = [row for row in records if not row["success"] and row["raw_path"] is not None]
    if not failures:
        return {}
    goals = manifest["goal_ids"]
    rounds = manifest["rounds_per_band"]
    prior_successes = []
    for i in range(index):
        stage = experience._stage_name(i, rounds)
        for row in read(runtime.root / stage / "complete.json")["collection"]:
            if row["success"]:
                prior_successes.append((stage, row["case"]))
    stage = experience._stage_name(index, rounds)
    prior_successes.extend((stage, row["case"]) for row in records if row["success"])

    def embed(tokens):
        batches = []
        for part in tokens.split(8):
            part = part.to(runtime.device, dtype=torch.float32)
            batches.append((adapter(part) if adapter is not None else part).flatten(1))
        return torch.cat(batches)

    result = {}
    for goal_index in sorted({row["case"]["goal_index"] for row in failures}):
        group = [row for row in failures if row["case"]["goal_index"] == goal_index]
        references = [(stage, case) for stage, case in prior_successes if case["goal_index"] == goal_index]
        raw = [experience._read_replay_raw(runtime, row) for row in group]
        for payload, row in zip(raw, group):
            steps = payload["primitive_steps"].tolist()
            if (payload["success"] or not payload.get("sampled_failure")
                    or not 1 <= len(steps) <= 5 or steps != sorted(set(steps))
                    or steps[0] < 0 or steps[-1] > row["micro_steps"]
                    or len(steps) != len(payload["frames"]) or len(steps) != len(payload["states"])):
                raise ValueError("invalid sampled observations for failure route matching")
        queries = runtime.encode_frames(torch.cat([row["frames"] for row in raw])).half()
        metadata = []

        def reference_embeddings():
            for source_stage, case in references:
                record = read(runtime.root / "episodes" / source_stage / f"{case['case_id']}.json")
                if record["case"] != case or record["runtime_id"] != runtime.runtime_id:
                    raise ValueError("successful route record differs from the collection report")
                payload = _reference(runtime, record, design, goals)
                if payload["tokens"] is not None:
                    metadata.append((case["case_id"], payload["steps"], payload["positions"]))
                    yield embed(payload["tokens"])

        scored = route_scores(embed(queries), reference_embeddings()) if len(references) >= 3 else None
        print(f"Goal {goal_index}: scored {len(queries) if scored else 0} failed observations "
              f"against {len(metadata)} usable successful routes", flush=True)
        offset = 0
        for row, payload in zip(group, raw):
            case = row["case"]
            end = offset + len(payload["frames"])
            scores = scored[0][offset:end] if scored else None
            pairs = ranking_pairs(scores) if scores is not None else []
            matches = []
            if scored:
                for matched in scored[1][offset:end]:
                    matches.append([dict(case_id=metadata[i][0], step=metadata[i][1][j],
                                         position=metadata[i][2][j]) for i, j in matched])
            ids = sorted({i for pair in pairs for i in pair})
            local = {i: j for j, i in enumerate(ids)}
            positions = payload["states"][:, :2]
            stats = dict(experience._failed_temporal_stats(np.asarray(row["primitive_positions"])),
                         skipped_failed=not pairs, eligible_pairs=len(pairs), retained_pairs=len(pairs),
                         route_feedback=dict(positions=positions.tolist(), steps=payload["primitive_steps"].tolist(),
                                             scores=scores, matches=matches, pairs=[list(p) for p in pairs]))
            lineage = dict(runtime_id=runtime.runtime_id, case=case, raw_sha256=row["raw_sha256"])
            saved = dict(inputs=lineage, stats=stats, positions=positions[ids],
                         pairs=[[local[a], local[b]] for a, b in pairs],
                         tokens=queries[offset:end][ids] if ids else None)
            save_tensors(runtime.root / "replay" / f"{case['case_id']}.pt", saved)
            result[case["case_id"]] = saved
            offset = end
    return result
