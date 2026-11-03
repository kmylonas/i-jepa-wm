import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/wall_matched_fixed_multi_colab.ipynb"


def test_uploadable_notebook_contains_sequential_runtime_only_data_flow():
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook["nbformat"] == 4
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "from google.colab import drive" in source
    assert 'LOCAL_ROOT = Path("/content/wall_matched_fixed_multi_v1")' in source
    assert 'DRIVE_ROOT = Path("/content/drive/MyDrive/datasets/wall_matched_fixed_multi_v1")' in source
    assert 'MODEL_DIRS = {mode: DRIVE_ROOT / mode for mode in ("fixed", "multi")}' in source
    assert 'prepare_manifest(train_layout_mode=mode, collection_profile="wide_only")' in source
    assert 'collection_profile="wide_only"' in source
    assert '"--max-action-length", "3.5"' in source
    assert '"--matched"' in source and '"--fixed-manifest"' in source
    assert source.index('run_model("fixed")') < source.index('run_model("multi")')
    assert source.index('cleanup_model("fixed")') < source.index('run_model("multi")')
    assert "shutil.rmtree(local_model_root)" in source
    assert "verify_model_outputs" in source
    assert '"--embedding-frame-stride", "1"' in source
    assert 'pilot_model("fixed")' in source
    assert 'pilot_model("multi")' in source


def test_notebook_embeds_pinned_source_patch():
    notebook = json.loads(NOTEBOOK.read_text())
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "BASE_COMMIT =" in source
    assert "PATCH_SHA256 =" in source
    assert '"git", "apply"' in source
