# Wall candidate ranking replay

This diagnostic asks whether the trained I-JEPA 1K world model preserves the
ranking of candidate actions by visual goal cost. It uses the existing multi
Wall checkpoint and frozen encoder. It does not need training episodes or
cached latents.

For each selected start and goal, the script runs the normal latent-cost CEM
search. It saves 32 uniformly sampled candidates from the first CEM round and
the 30 best candidates from the final round. Each sequence is then executed
from a fresh simulator reset. The terminal real frame and goal frame are
encoded with the *same* I-JEPA 1K encoder used for model training. The saved
predicted and real latent MSE values can therefore be compared directly.

The default test uses the seen Wall layout `(32, 30)`, three far start–goal
pairs on opposite sides of the wall, horizons 1 and 5, and a Euclidean action
length cap of 3.5 per primitive action. This matches the multi Wall model's
training action limit. The planner uses 300 candidates, 30 elites, and 10 CEM
rounds. The selected cases are evaluation starts generated from fresh seeds;
they are not training episodes.

## Colab after mounting Drive and cloning both repositories

The following cell assumes `/content/i-jepa-wm` is the
`wall-layout-generalization` branch and `/content/dino_wm` is checked out at
`0a9492fa12044b852ae9e001cc74604b79c8bb0c`. Keep the same package setup
used for the earlier Wall experiment.

```python
from pathlib import Path
import json
import shutil
import subprocess
import sys

REPO = Path("/content/i-jepa-wm")
DINO_ROOT = Path("/content/dino_wm")
DRIVE_ROOT = Path("/content/drive/MyDrive/datasets/wall_layout_generalization")
CHECKPOINT = DRIVE_ROOT / "world_model/best.pt"
ENCODER_DRIVE = Path(
    "/content/drive/MyDrive/checkpoints/"
    "IN1K-vit.h.14-300e.encoder-only.pth.tar"
)
ENCODER_LOCAL = Path("/content/ijepa_1k_encoder.pth.tar")
ENV_CONFIG = DRIVE_ROOT / "environment.json"
LOCAL_CONFIG = Path("/content/wall_candidate_rankings_environment.json")
OUTPUT_DIR = DRIVE_ROOT / "results/candidate_rankings_wall32_door30_v1"

for path in (
    REPO / "src/mylonas_ijepa_wm/evaluate_wall_candidate_rankings.py",
    DINO_ROOT / "env/wall/envs/wall.py",
    CHECKPOINT,
    ENCODER_DRIVE,
    ENV_CONFIG,
):
    assert path.is_file(), f"Missing: {path}"

if not ENCODER_LOCAL.is_file():
    shutil.copy2(ENCODER_DRIVE, ENCODER_LOCAL)
settings = json.loads(ENV_CONFIG.read_text())
settings["dino_wm_root"] = str(DINO_ROOT)
LOCAL_CONFIG.write_text(json.dumps(settings, indent=2))

subprocess.run([
    sys.executable, "-u", "-m",
    "src.mylonas_ijepa_wm.evaluate_wall_candidate_rankings",
    "--world-model-checkpoint", str(CHECKPOINT),
    "--encoder-checkpoint", str(ENCODER_LOCAL),
    "--env-config", str(LOCAL_CONFIG),
    "--output-dir", str(OUTPUT_DIR),
    "--wall-x", "32", "--door-y", "30",
    "--device", "cuda", "--precision", "auto",
], cwd=REPO, check=True)
```

The output directory contains `summary.json`, `candidates.csv`, one PNG per
start and horizon, and an action archive for each PNG. The left plot compares
world-model predicted cost with real terminal-frame cost. The right plot
checks whether real latent cost tracks actual straight-line distance to the
goal. Read `predicted_vs_true_rho` separately for the random candidates and
final CEM elites: a pooled correlation can conceal incorrect ordering within
the elites. The true cost of the predicted best candidate versus the best
available real cost is the other main diagnostic. Straight-line distance is
supporting context because a wall can require a detour.

To inspect the summary and first figure in Colab:

```python
from IPython.display import Image, display

results = json.loads((OUTPUT_DIR / "summary.json").read_text())
def fmt_rho(value):
    return "undefined" if value is None else f"{value:.3f}"

for case in results["cases"]:
    print(f"seed {case['seed']} horizon {case['horizon']}")
    for group in ("initial_random", "final_elite", "all"):
        metrics = case["rankings"][group]
        print(
            f"  {group}: "
            f"predicted/real rho={fmt_rho(metrics['predicted_vs_true_rho'])}, "
            f"true/distance rho={fmt_rho(metrics['true_vs_distance_rho'])}, "
            f"predicted-best real cost={metrics['predicted_best_true_cost']:.4f}, "
            f"best available real cost={metrics['oracle_best_true_cost']:.4f}"
        )
first = results["cases"][0]
display(Image(filename=str(
    OUTPUT_DIR / f"seed_{first['seed']}_h{first['horizon']}.png"
)))
```

The three starts and each candidate sequence are fixed by saved seeds and
action archives. A large difference between predicted and real rankings
indicates that world-model predictions distort CEM's choices. A weak
relationship between *real* latent cost and actual progress points to the
encoder goal cost instead. If both relationships are strong yet planning
still fails, inspect CEM search and replanning limits.
