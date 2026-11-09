import numpy as np

from src.mylonas_ijepa_wm import experience_failure_cap as cap


def test_cap_only_replaces_excess_failure_draws_and_keeps_goal_when_possible():
    triples = [[0, 2, 3], [0, 4, 5], [1, 6, 7], [1, 8, 9]]
    failed = np.array([True, False, True, False])
    source = np.array([[0, 2, 1, 3, 0, 2, 0, 2, 0, 2]] * 2)
    draws = cap.capped_schedule(source, triples, failed, .15, seed=0)
    assert failed[draws].sum(axis=1).tolist() == [1, 2]
    assert np.array_equal(draws, cap.capped_schedule(source, triples, failed, .15, seed=0))
    assert np.array_equal(draws[:, 2:4], source[:, 2:4])
    groups = np.asarray(triples)[:, 0]
    assert np.array_equal(groups[draws], groups[source])


def test_cap_never_adds_failure_draws_to_a_success_batch():
    source = [[1, 1], [1, 1]]
    assert cap.capped_schedule(source, [[0, 1, 2], [0, 3, 4]],
                               np.array([True, False]), .15, seed=0).tolist() == source
