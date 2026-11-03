import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/wall_matched_fixed_multi_colab.ipynb"


def test_uploadable_notebook_contains_sequential_runtime_only_data_flow():
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook["nbformat"] == 4
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "from google.colab import drive" in source
    assert 'LOCAL_ROOT = Path("/content/wall_matched_fixed_multi_v2")' in source
    assert 'DRIVE_ROOT = Path("/content/drive/MyDrive/datasets/wall_matched_fixed_multi_v2")' in source
    assert 'MODEL_DIRS = {mode: DRIVE_ROOT / mode for mode in ("fixed", "multi")}' in source
    assert 'prepare_manifest(train_layout_mode=mode, collection_profile="matched_alternating")' in source
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
    assert 'SOURCE_COMMIT = "82b08453f3ee8f0913da851cfeb98b296c9e90af"' in source
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
