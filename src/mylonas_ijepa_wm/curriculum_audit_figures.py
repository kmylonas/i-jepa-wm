"""Static figures for the saved curriculum-experience audit."""

from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
import torch


def _save(fig, path):
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _world(ax, goal):
    ax.add_patch(Rectangle((29, 0), 6, 26, color="0.55", alpha=.3))
    ax.add_patch(Rectangle((29, 34), 6, 30, color="0.55", alpha=.3))
    ax.add_patch(Rectangle((40, 30), 14, 6, fill=False, edgecolor="#3b82f6", linestyle="--", label="collection starts"))
    ax.add_patch(Rectangle((40, 14), 14, 10, fill=False, edgecolor="#f59e0b", linestyle="--", label="far test starts"))
    ax.add_patch(Circle(goal, 4.5, fill=False, color="#16a34a", linewidth=1.4))
    ax.scatter(*goal, marker="*", s=80, color="#16a34a", zorder=4)
    ax.set_xlim(0, 64); ax.set_ylim(0, 64); ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")


def _fraction(row, label):
    value = row[label]["episode_uniform"]
    return [value[key + "_fraction"] for key in ("progress", "ambiguous", "contradictory")]


def _ranked_examples(rows):
    def badness(row):
        pairs = row["pairs_after"]
        wrong = pairs["contradictory_fraction"] or 0
        away = row["away_fraction"] or 0
        efficiency = row["success_path_efficiency"] or 0
        return (wrong, away, -efficiency)
    ordered = sorted(rows, key=badness, reverse=True)
    chosen = ordered[:2]
    best = min(rows, key=badness)
    if best not in chosen:
        chosen.append(best)
    return chosen[:3]


def _load_frames(root, row):
    raw = torch.load(root / row["raw_path"], map_location="cpu", weights_only=True)
    return np.asarray(raw["frames"])


def _frame_with_goal(frame, goal):
    image = np.asarray(frame).copy()
    h, w = image.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    gx = goal[0] / 64 * (w - 1)
    gy = (64 - goal[1]) / 64 * (h - 1)
    radius = 4.5 / 64 * min(h, w)
    edge = np.abs(np.sqrt((xx - gx) ** 2 + (yy - gy) ** 2) - radius) <= 1.5
    image[edge] = [30, 200, 80]
    return image


def _placeholder(path, title, message):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off"); ax.set_title(title)
    ax.text(.5, .5, message, ha="center", va="center", wrap=True, transform=ax.transAxes)
    _save(fig, path)


def render_audit(root, rows, report, output):
    """Render all audit plots from validated rows and selected saved RGB frames."""
    root, output = Path(root), Path(output)
    goal = np.asarray(rows[0]["trace"]["goal"])

    cols = min(5, len(rows)); nrows = math.ceil(len(rows) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(3.2 * cols, 2.5 * nrows), squeeze=False)
    for ax, row in zip(axes.flat, rows):
        d = row["trace"]["distances"]
        ax.plot(row["trace"]["steps"], d, color="#334155", linewidth=1.3)
        retained = row["trace"]["retained_indices"]
        ax.scatter(retained, np.asarray(d)[retained], s=8, color="#2563eb")
        ax.axhline(4.5, color="#16a34a", linestyle="--", linewidth=1)
        away = row["away_fraction"]
        ax.set_title(f"{row['case_id']}\naway {0 if away is None else 100*away:.0f}%", fontsize=8)
        ax.set_xlabel("primitive step"); ax.set_ylabel("goal distance")
    for ax in axes.flat[len(rows):]: ax.axis("off")
    fig.suptitle("Distance to goal during every collected episode", y=.995)
    fig.tight_layout(rect=(0, 0, 1, .97))
    _save(fig, output / "progress.png")

    if report["pair_quality"]["after"]["usable_successful_episodes"] == 0:
        _placeholder(output / "pair_quality.png", "Does temporal order give a correct closer-to-goal label?",
                     "No successful episode had an eligible pair after the revisit and separation filters.")
    else:
        before, after = _fraction(report["pair_quality"], "before"), _fraction(report["pair_quality"], "after")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        labels = ["useful: later closer", "unclear within tolerance", "wrong: later farther"]
        colors = ["#16a34a", "#94a3b8", "#dc2626"]
        bottom = np.zeros(2)
        for i, label in enumerate(labels):
            values = [before[i], after[i]]
            ax.bar([0, 1], values, bottom=bottom, color=colors[i], label=label)
            bottom += values
        ax.set_xticks([0, 1], ["before revisit filter", "pairs actually used"])
        ax.set_ylim(0, 1); ax.set_ylabel("fraction under trainer sampling")
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.14), ncol=3)
        ax.set_title("Does temporal order give a correct closer-to-goal label?")
        _save(fig, output / "pair_quality.png")

    fig, ax = plt.subplots(figsize=(7, 7)); _world(ax, goal)
    for row in rows:
        states = np.asarray(row["trace"]["states"])
        ax.plot(states[:, 0], states[:, 1], alpha=.45, linewidth=1)
        ax.scatter(states[0, 0], states[0, 1], s=8, color="#2563eb")
    ax.set_title("State coverage from collection trajectories")
    ax.legend(loc="lower left", fontsize=8)
    _save(fig, output / "coverage.png")

    examples = _ranked_examples(rows)
    fig, axes = plt.subplots(len(examples), 2, figsize=(10, 4 * len(examples)), squeeze=False)
    for axis_row, row in zip(axes, examples):
        states = np.asarray(row["trace"]["states"]); d = row["trace"]["distances"]
        _world(axis_row[0], goal); axis_row[0].plot(states[:, 0], states[:, 1], "-o", ms=2)
        axis_row[0].set_title(row["case_id"])
        axis_row[1].plot(row["trace"]["steps"], d); axis_row[1].axhline(4.5, color="#16a34a", linestyle="--")
        axis_row[1].set_xlabel("primitive step"); axis_row[1].set_ylabel("goal distance")
        axis_row[1].set_title(f"efficiency {row['success_path_efficiency']:.2f}" if row["success_path_efficiency"] is not None else "efficiency unavailable")
    fig.suptitle("Episodes selected for close inspection")
    fig.tight_layout(rect=(0, 0, 1, .97))
    _save(fig, output / "episode_examples.png")

    fig, axes = plt.subplots(len(examples), 3, figsize=(10, 3.4 * len(examples)), squeeze=False)
    for axis_row, row in zip(axes, examples):
        frames = _load_frames(root, row); picks = [0, len(frames) // 2, len(frames) - 1]
        distances = row["trace"]["distances"]
        for ax, index, label in zip(axis_row, picks, ("start", "middle", "end")):
            ax.imshow(_frame_with_goal(frames[index], goal)); ax.axis("off")
            ax.set_title(f"{row['case_id']} {label}\nt={index}, distance={distances[index]:.1f}", fontsize=9)
    fig.suptitle("Real observations from selected episodes (green ring marks success region)")
    fig.tight_layout(rect=(0, 0, 1, .97))
    _save(fig, output / "frames.png")

    pairs = []
    for row in rows:
        for pair in row["contradictory_examples"]:
            pairs.append((pair["distance_increase"], row, pair))
    pairs.sort(key=lambda item: item[0], reverse=True)
    if not pairs:
        _placeholder(output / "training_pair_examples.png", "Surviving contradictory training pairs",
                     "None of the retained pairs moved farther from the goal by more than the diagnostic tolerance.")
    else:
        shown = pairs[:3]
        fig, axes = plt.subplots(len(shown), 2, figsize=(7, 3.4 * len(shown)), squeeze=False)
        for axis_row, (_, row, pair) in zip(axes, shown):
            frames = _load_frames(root, row)
            for ax, index, when in zip(axis_row, pair["indices"], ("earlier", "later")):
                ax.imshow(_frame_with_goal(frames[index], goal)); ax.axis("off")
                ax.set_title(f"{row['case_id']} {when} t={index}\ndistance {row['trace']['distances'][index]:.1f}")
        fig.suptitle("Pairs used for training where the later frame is farther from goal")
        fig.tight_layout(rect=(0, 0, 1, .97))
        _save(fig, output / "training_pair_examples.png")

    lost = report["evaluation"]["lost_cases"]
    if not lost:
        _placeholder(output / "lost_cases.png", "Cases lost after adaptation", "No raw-success/adapted-failure case was available.")
    else:
        fig, axes = plt.subplots(len(lost), 2, figsize=(10, 4 * len(lost)), squeeze=False)
        for axis_row, item in zip(axes, lost):
            _world(axis_row[0], np.asarray(item["case"]["goal"]))
            for label, color in (("raw", "#334155"), ("adapted", "#dc2626")):
                trace = item["traces"][label]; states = np.asarray(trace["states"])
                axis_row[0].plot(states[:, 0], states[:, 1], "-o", ms=3, label=label, color=color)
                axis_row[1].plot(trace["steps"], trace["distances"], "-o", ms=3, label=label, color=color)
            axis_row[0].set_title(item["case_id"]); axis_row[0].legend()
            axis_row[1].axhline(4.5, color="#16a34a", linestyle="--")
            axis_row[1].set_xlabel("actual primitive step"); axis_row[1].set_ylabel("goal distance")
            axis_row[1].set_title("coarse observations at replanning boundaries"); axis_row[1].legend()
        fig.suptitle("Evaluation cases that succeeded before and failed after adaptation")
        fig.tight_layout(rect=(0, 0, 1, .97))
        _save(fig, output / "lost_cases.png")

