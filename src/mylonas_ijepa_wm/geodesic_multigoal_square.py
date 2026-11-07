"""Entry point for multi-goal geodesic ranking on expanding square grids."""
from src.mylonas_ijepa_wm import geodesic_multigoal, geodesic_radial


def run(output, env, encode, device, signature, *, updates=500):
    """Fit and evaluate one adapter on per-goal square grids."""
    return geodesic_multigoal.run(output, env, encode, device, signature, updates=updates, geometry="square")


if __name__ == "__main__":
    geodesic_radial.main(runner=run)
