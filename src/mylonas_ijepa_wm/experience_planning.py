"""Benchmark the final collected ranking adapter on the original planning cases."""
import argparse
from pathlib import Path
from types import SimpleNamespace

from src.mylonas_ijepa_wm import ranking_planning as planning
from src.mylonas_ijepa_wm.curriculum_multiband_protocol import read
from src.mylonas_ijepa_wm.curriculum_protocol import digest, write_json
from src.mylonas_ijepa_wm.curriculum_replay import load_adapter
from src.mylonas_ijepa_wm.curriculum_runtime import file_sha256
from src.mylonas_ijepa_wm.radial_experiment import locations
from src.mylonas_ijepa_wm.ranking_planning_protocol import summarize


def load_inputs(historical, collected):
    config = read(historical / "experiment.json")
    manifest = read(historical / "manifest.json")
    if config["kind"] != "ranking_planning_v1" or config["manifest_digest"] != digest(manifest):
        raise ValueError("historical benchmark configuration differs")
    refs, reference_runtime = planning._references(historical, config, manifest)
    reference_signature = reference_runtime["signature"]
    cases = manifest["evaluation"]
    baselines = {}
    for role, label in (("raw", "raw"), ("adapted", "handcrafted")):
        folder, saved = planning._runtime(historical, config, manifest, role)
        expected_variant = dict(reference_signature["runtime_variant"], condition=role)
        if saved["signature"] != dict(reference_signature, runtime_variant=expected_variant):
            raise ValueError(f"historical {role} runtime differs from shared reference")
        inputs = planning._condition_inputs(config, historical / "reference", saved)
        stage = read(folder / "stage.json")
        if stage["inputs"] != inputs or stage["status"] != "complete":
            raise ValueError(f"historical {role} benchmark is incomplete")
        baselines[label] = [planning._episode(folder, case, inputs) for case in cases]
        if any(record is None for record in baselines[label]):
            raise ValueError(f"missing historical {role} episode")

    curriculum = read(collected / "experiment.json")
    runtime = read(collected / "runtime.json")
    signature = runtime["signature"]
    if (curriculum["kind"] != "experience_curriculum_v1" or curriculum["handcrafted"] != config["source"]
            or len(curriculum["manifest"]["stages"]) != 4
            or runtime["runtime_id"] != digest(signature)
            or signature["manifest"] != digest(curriculum["manifest"])
            or signature["models"] != config["models"]
            or signature["runtime_variant"]["experiment_digest"] != digest(curriculum)):
        raise ValueError("collected run differs from the historical benchmark inputs")
    checkpoint = collected / "band_4" / "adapter.pt"
    marker = read(checkpoint.with_name("complete.json"))
    sha = file_sha256(checkpoint)
    if (marker["band"] != 4 or marker["runtime_id"] != runtime["runtime_id"]
            or marker["checkpoint_sha256"] != sha):
        raise ValueError("final collected checkpoint hash or provenance differs")
    adapter, payload = load_adapter(checkpoint, "cpu", expected_runtime_id=runtime["runtime_id"])
    if payload["provenance"]["band"] != 4 or adapter.dim != refs["tokens"].shape[-1]:
        raise ValueError("final collected checkpoint does not match the benchmark")
    inputs = dict(historical_run=str(historical), collected_run=str(collected),
                  historical_config=digest(config), curriculum_config=digest(curriculum),
                  baseline_episodes=digest(baselines), reference_runtime=reference_runtime["runtime_id"],
                  reference_tokens_sha256=file_sha256(historical / "reference" / "goals.pt"),
                  checkpoint_sha256=sha)
    return inputs, manifest, refs, baselines, adapter, reference_signature


def evaluate(runtime, inputs, manifest, refs, baselines, adapter, old_signature, new_signature):
    # Different code revisions are expected. Physical setup and model artifacts must match.
    for key in ("manifest", "models", "dino_source", "dino_commit", "precision", "encoder_batch_size"):
        if old_signature[key] != new_signature[key]:
            raise ValueError(f"planning runtime differs from the historical benchmark for {key}")
    old_variant = old_signature["runtime_variant"]
    new_variant = new_signature["runtime_variant"]
    numeric_runtime = dict(torch_version=[old_signature["torch_version"], new_signature["torch_version"]],
                           device_name=[old_variant["device_name"], new_variant["device_name"]],
                           matmul_precision=[old_variant["matmul_precision"], new_variant["matmul_precision"]])
    differences = {key: pair for key, pair in numeric_runtime.items() if pair[0] != pair[1]}
    if differences:
        print(f"Numerical runtime differs (historical, collected): {differences}", flush=True)

    cases = manifest["evaluation"]
    records = planning._run_cases(runtime.root, cases, "collected", runtime, refs,
                                  adapter.to(runtime.device), dict(inputs, runtime_id=runtime.runtime_id))
    result = dict(inputs=inputs, runtime_differences=differences, numeric_runtime=numeric_runtime,
                  handcrafted_vs_collected=summarize(baselines["handcrafted"], records),
                  raw_vs_collected=summarize(baselines["raw"], records),
                  records=dict(baselines, collected=records))
    write_json(runtime.root / "report.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("historical-run", "collected-run", "output-dir", "world-model-checkpoint",
                  "encoder-checkpoint", "dino-wm-root"):
        parser.add_argument(f"--{field}", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    output, historical = locations(args.output_dir, args.historical_run,
                                   (args.collected_run, args.world_model_checkpoint,
                                    args.encoder_checkpoint, args.dino_wm_root))
    inputs, manifest, refs, baselines, adapter, old_signature = load_inputs(historical, args.collected_run.resolve())
    models = old_signature["models"]
    runtime_args = SimpleNamespace(output_dir=output, world_model_checkpoint=args.world_model_checkpoint,
        encoder_checkpoint=args.encoder_checkpoint, dino_wm_root=args.dino_wm_root, device=args.device,
        precision=old_signature["precision"], encoder_batch_size=old_signature["encoder_batch_size"],
        expected_world_model_sha256=models["world_model"], expected_encoder_sha256=models["encoder"],
        run_seed=0, videos=False, runtime_signature_extra=planning.runtime_variant(inputs, "collected", args.device))
    runtime = planning.PlanningRuntime(runtime_args, manifest)
    try:
        new_signature = read(output / "runtime.json")["signature"]
        return evaluate(runtime, inputs, manifest, refs, baselines, adapter, old_signature, new_signature)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
