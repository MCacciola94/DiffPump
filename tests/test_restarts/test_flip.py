import numpy as np
import pytest
import torch

from diffpump.restarts.flip import (
    find_indices_largest_loss,
    flip_theta,
)


def test_find_indices_largest_loss():
    x_lp = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 1.0])
    binary_idxs = [1, 2, 3, 5]
    x_binary = x_lp[binary_idxs]
    # test 1: vary nb flips
    idxs_to_flip = find_indices_largest_loss(x_binary, binary_idxs, 1)
    assert (idxs_to_flip == [3]).all()
    idxs_to_flip = find_indices_largest_loss(x_binary, binary_idxs, 2)
    assert (idxs_to_flip == [2, 3]).all()
    idxs_to_flip = find_indices_largest_loss(x_binary, binary_idxs, 3)
    assert (idxs_to_flip == [1, 2, 3]).all()
    idxs_to_flip = find_indices_largest_loss(x_binary, binary_idxs, 4)
    idxs_to_flip.sort()
    assert (idxs_to_flip == [1, 2, 3, 5]).all()
    # test 2: nb_flips is outside possible range
    with pytest.raises(ValueError):
        find_indices_largest_loss(x_binary, binary_idxs, 0)
    with pytest.warns(UserWarning):
        find_indices_largest_loss(x_binary, binary_idxs, 5)


def test_flip_theta():
    theta = torch.DoubleTensor(np.array([0.123, 0.0, 0.0, 0.0]))
    idxs_to_flip = [1, 2, 3]
    # Test 1: raises warning error
    with pytest.warns(UserWarning):
        flip_theta(theta, idxs_to_flip)
    # Test 2: no warning, flip is correct
    theta = torch.DoubleTensor(np.array([0.123, 1.0, -1.0, 0.0]))
    flip_theta(theta, idxs_to_flip)
    assert np.isclose(
        theta.detach().clone().numpy(), np.array([0.123, -1.0, 1.0, 0.0])
    ).all()
