import numpy as np

from diffpump.restarts.perturb import (
    find_indices_to_perturb,
)


def test_find_indices_to_perturb():
    """
    Test the function to find indices to perturb.

    The indices to perturb are the ones with largest difference
    between x_lp and x_round.
    """
    x_lp = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 1.0])
    binary_mask = np.array([0, 1, 1, 1, 0, 1])
    # Test 1: very large difference to force selection
    x_round = np.array([0, 0, 0, 0, 1, 2])
    idxs_to_flip = find_indices_to_perturb(x_lp, x_round, binary_mask)
    assert idxs_to_flip.dtype == np.int64
    assert 5 in idxs_to_flip
