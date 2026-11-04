import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.mylonas_ijepa_wm.wall_layout_manifest import (
    manifest_digest, prepare_manifest, validate_checkpoint_manifest,
)


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/wall_matched_fixed_multi_colab.ipynb"


def test_uploadable_notebook_contains_sequential_runtime_only_data_flow():
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook["nbformat"] == 4
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "from google.colab import drive" in source
    assert 'LOCAL_ROOT = Path("/content/wall_matched_fixed_multi_v3")' in source
    assert 'DRIVE_ROOT = Path("/content/drive/MyDrive/datasets/wall_matched_fixed_multi_v3")' in source
    assert 'MODEL_DIRS = {mode: DRIVE_ROOT / mode for mode in ("fixed", "multi")}' in source
    assert 'prepare_manifest(train_layout_mode=mode, collection_profile="matched_alternating",' in source
    assert 'validation_layout_mode="training"' in source
    assert '"--validation-layout-mode", "training"' in source
    assert 'fixed_manifest["val_layouts"] == fixed_manifest["train_layouts"]' in source
    assert 'multi_manifest["val_layouts"] == multi_manifest["train_layouts"]' in source
    assert 'if e["split"] == "test"' in source
    assert 'collection_profile="matched_alternating"' in source
    assert '"--max-action-length", "3.5"' in source
    assert '"--matched"' in source and '"--fixed-manifest"' in source
    assert source.index('run_model("fixed")') < source.index('run_model("multi")')
    assert source.index('cleanup_model("fixed")') < source.index('run_model("multi")')
    assert "shutil.rmtree(local_model_root)" in source
    assert "verify_model_outputs" in source
    assert '"--embedding-frame-stride", "1"' in source
    assert 'pilot_model("fixed")' in source
    assert 'pilot_model("multi")' in source
    assert 'assert audit["coverage_pass"]' in source


def test_notebook_checks_out_pushed_source_commit_without_embedded_patch():
    notebook = json.loads(NOTEBOOK.read_text())
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert 'SOURCE_COMMIT = "b493bf89dd67473a94b3cf4681f952e5f6a7318d"' in source
    assert '["git", "fetch", "origin", "wall-layout-generalization"]' in source
    assert '"git", "checkout", SOURCE_COMMIT' in source
    assert "PATCH_B64" not in source
    assert "PATCH_SHA256" not in source
    assert '"git", "apply"' not in source


def test_notebook_resume_checks_remaining_cache_and_repairs_bad_pairs():
    notebook = json.loads(NOTEBOOK.read_text())
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "missing_cache_count * LATENT_EPISODE_BYTES" in source
    assert "EPISODES * LATENT_EPISODE_BYTES" not in source
    assert 'latent_path.unlink(missing_ok=True)' in source
    assert 'trajectory_path.unlink(missing_ok=True)' in source


def test_notebook_caps_training_at_20_epochs_and_completes_before_cleanup():
    notebook = json.loads(NOTEBOOK.read_text())
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "MAX_EPOCHS = 20" in source
    assert 'epochs=MAX_EPOCHS' in source
    assert '"--epochs", str(MAX_EPOCHS)' in source
    assert 'reconcile_run_identity(mode, paths)' in source
    execution = "".join(notebook["cells"][7]["source"])
    assert execution.index('run_model("fixed")') < execution.index('cleanup_model("fixed")')


def test_notebook_adopts_incomplete_100_epoch_identity_with_matching_checkpoint(tmp_path):
    notebook = json.loads(NOTEBOOK.read_text())
    tree = ast.parse("".join(notebook["cells"][3]["source"]))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"expected_identity", "reconcile_run_identity"}]
    manifest = prepare_manifest(train_episodes=18, val_episodes=18, test_episodes=8,
                                train_layout_mode="fixed", collection_profile="matched_alternating",
                                validation_layout_mode="training")
    namespace = {
        "Path": Path, "json": json, "torch": torch, "manifest_digest": manifest_digest,
        "validate_checkpoint_manifest": validate_checkpoint_manifest,
        "manifest_for": lambda mode: manifest,
        "ENCODER_SHA256": "encoder-hash", "SOURCE_COMMIT": "source-hash",
        "MAX_EPOCHS": 20,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    expected = namespace["expected_identity"]("fixed")
    drive = tmp_path / "fixed"
    checkpoint_dir = drive / "world_model"
    checkpoint_dir.mkdir(parents=True)
    paths = {"drive": drive, "checkpoint_dir": checkpoint_dir, "cache": tmp_path / "cache"}
    identity_path = drive / "run_identity.json"
    identity_path.write_text(json.dumps({**expected, "epochs": 100}))
    metadata = {
        "environment": "wall", "experiment_manifest_sha256": manifest_digest(manifest),
        "layout_splits": {split: manifest[f"{split}_layouts"] for split in ("train", "val", "test")},
        **{f"{split}_ids": [entry["episode_id"] for entry in manifest["episodes"]
                            if entry["split"] == split] for split in ("train", "val", "test")},
    }
    for name in ("best.pt", "last.pt"):
        torch.save({"epoch": 19, "metadata": metadata}, checkpoint_dir / name)
    namespace["reconcile_run_identity"]("fixed", paths)
    assert json.loads(identity_path.read_text()) == expected
    namespace["reconcile_run_identity"]("fixed", paths)
    assert json.loads(identity_path.read_text()) == expected


def test_notebook_rejects_identity_adoption_after_20_epochs(tmp_path):
    notebook = json.loads(NOTEBOOK.read_text())
    tree = ast.parse("".join(notebook["cells"][3]["source"]))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"expected_identity", "reconcile_run_identity"}]
    manifest = prepare_manifest(train_episodes=18, val_episodes=18, test_episodes=8,
                                train_layout_mode="fixed", collection_profile="matched_alternating",
                                validation_layout_mode="training")
    namespace = {
        "Path": Path, "json": json, "torch": torch, "manifest_digest": manifest_digest,
        "validate_checkpoint_manifest": validate_checkpoint_manifest,
        "manifest_for": lambda mode: manifest,
        "ENCODER_SHA256": "encoder-hash", "SOURCE_COMMIT": "source-hash",
        "MAX_EPOCHS": 20,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    drive = tmp_path / "fixed"
    checkpoint_dir = drive / "world_model"
    checkpoint_dir.mkdir(parents=True)
    paths = {"drive": drive, "checkpoint_dir": checkpoint_dir, "cache": tmp_path / "cache"}
    identity_path = drive / "run_identity.json"
    identity_path.write_text(json.dumps({**namespace["expected_identity"]("fixed"), "epochs": 100}))
    torch.save({"epoch": 20, "metadata": {}}, checkpoint_dir / "last.pt")
    with pytest.raises(AssertionError, match="20 epochs"):
        namespace["reconcile_run_identity"]("fixed", paths)
    assert json.loads(identity_path.read_text())["epochs"] == 100


def test_notebook_checks_current_identity_and_pilot_actions():
    notebook = json.loads(NOTEBOOK.read_text())
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert 'assert json.loads(identity_path.read_text()) == expected_identity(mode)' in source
    assert 'assert complete["identity"] == expected_identity(mode)' in source
    assert 'compare_matched_action_hashes(' in source
    assert 'shared_seeds=range(100000, 100010)' in source
    assert 'pilot_coverage.json' in source


def test_notebook_requires_encoder_only_checkpoint():
    notebook = json.loads(NOTEBOOK.read_text())
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert 'ENCODER_DRIVE = Path("/content/drive/MyDrive/checkpoints/IN1K-vit.h.14-300e.encoder-only.pth.tar")' in source
    assert 'IN1K-vit.h.14-300e.pth.tar' not in source


def test_notebook_streams_child_logs_and_reports_training_steps():
    notebook = json.loads(NOTEBOOK.read_text())
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert 'subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE,' in source
    assert 'stderr=subprocess.STDOUT' in source
    assert 'print(line, end="", flush=True)' in source
    assert '"--log-every", "100"' in source


def test_notebook_command_runner_relays_both_streams_and_failures(capsys):
    notebook = json.loads(NOTEBOOK.read_text())
    setup = ast.parse("".join(notebook["cells"][1]["source"]))
    shell_node = next(node for node in setup.body
                      if isinstance(node, ast.FunctionDef) and node.name == "shell")
    namespace = {"subprocess": subprocess}
    exec(compile(ast.Module(body=[shell_node], type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    namespace["shell"]([sys.executable, "-u", "-c",
                        "import sys; print('stdout-line'); print('stderr-line', file=sys.stderr)"])
    output = capsys.readouterr().out
    assert "stdout-line" in output and "stderr-line" in output
    with pytest.raises(subprocess.CalledProcessError):
        namespace["shell"]([sys.executable, "-u", "-c", "raise SystemExit(7)"])


def test_notebook_accepts_embedded_wall_layout_tuple(tmp_path):
    notebook = json.loads(NOTEBOOK.read_text())
    functions = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            functions.extend(node for node in ast.parse("".join(cell["source"])).body
                             if isinstance(node, ast.FunctionDef)
                             and node.name in {"episode_files", "valid_cache_pair"})
    namespace = {"np": np, "torch": torch, "manifest_digest": manifest_digest}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    manifest = prepare_manifest(train_episodes=18, val_episodes=4, test_episodes=8,
                                train_layout_mode="fixed", collection_profile="matched_alternating")
    paths = {"raw": tmp_path / "raw", "cache": tmp_path / "cache"}
    _, latent_path, trajectory_path = namespace["episode_files"](paths, 0)
    latent_path.parent.mkdir(parents=True)
    trajectory_path.parent.mkdir(parents=True)
    latent = np.lib.format.open_memmap(latent_path, mode="w+", dtype=np.float16,
                                       shape=(51, 256, 1280))
    del latent
    torch.save({"manifest_sha256": manifest_digest(manifest), "seed": 0,
                "wall_layout": (32, 30), "embedding_frame_stride": 1}, trajectory_path)
    assert namespace["valid_cache_pair"](paths, manifest, 0)
