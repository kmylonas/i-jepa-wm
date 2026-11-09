"""Remove leave-and-return loops from successful trajectories."""

import numpy as np


def loop_erased_steps(positions, radius):
    """Keep the latest visit when the path leaves a neighborhood and returns."""
    xy = np.asarray(positions, dtype=float)
    delta = xy[:, None] - xy[None, :]
    near = (delta ** 2).sum(axis=2) <= radius ** 2
    retained = []
    for i in range(len(xy)):
        revisit = next((j for j, step in enumerate(retained[:-1])
                        if near[step, i] and not near[step, step + 1:i].all()), None)
        if revisit is not None:
            del retained[revisit:]
        retained.append(i)
    return retained
