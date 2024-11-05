import numpy as np
import pytest

from diffpump.feas_pump import (
    flip_top_k_binary,
    get_binary_mask,
    hamming_update,
    integrality_loss,
    is_binary,
    round_binary_vars,
)


def test_get_binary_mask():
    num_variables = 5
    # Test 1
    binary_idxs = np.array([1, 2, 4])
    mask = get_binary_mask(num_variables, binary_idxs)
    assert len(mask) == num_variables
    assert (mask == np.array([0, 1, 1, 0, 1])).all()
    # Test 2: no binary should raise error
    binary_idxs = np.array([])
    with pytest.raises(ValueError):
        mask = get_binary_mask(num_variables, binary_idxs)
    # Test 3: all binary
    binary_idxs = np.array([0, 1, 2, 3, 4])
    mask = get_binary_mask(num_variables, binary_idxs)
    assert len(mask) == num_variables
    assert (mask == np.ones(num_variables)).all()


def test_integrality_loss():
    """
    Test the behavior of the integrality loss function.
    By default, this function is:
        loss = min{x_j, 1-x_j}
    applied component wise to each x_j and then sum.
    """
    x = np.array([0.5, 0.5, 1, 0, 0, 1, 0.3])
    # test 1
    binary_idxs = np.array([0, 1, 2, 3, 4, 5])
    loss = integrality_loss(x, binary_idxs).item()
    assert loss == (0.5 + 0.5)
    # test 2
    binary_idxs = np.array([1, 2, 3, 4, 5])
    loss = integrality_loss(x, binary_idxs).item()
    assert loss == 0.5
    # test 3
    binary_idxs = np.array([0, 1, 2, 3, 4, 5, 6])
    loss = integrality_loss(x, binary_idxs).item()
    assert loss == (0.5 + 0.5 + 0.3)


def test_is_binary():
    x = np.array([0.5, 0.5, 1, 0, 0, 1, 0.3])
    # test 1
    binary_idxs = np.array([0, 1, 2, 3, 4, 5])
    assert not is_binary(x, binary_idxs)
    # test 2
    binary_idxs = np.array([1, 2, 3, 4, 5])
    assert not is_binary(x, binary_idxs)
    # test 3
    binary_idxs = np.array([0, 1, 2, 3, 4, 5, 6])
    assert not is_binary(x, binary_idxs)
    # test 4
    binary_idxs = np.array([2, 3, 4, 5])
    assert is_binary(x, binary_idxs)


def test_round_binary_vars():
    x = np.array([0.5, 0.5, 1, 0, 0, 1, 0.3])
    # test 1
    binary_idxs = np.array([0, 1, 2, 3, 4, 5])
    x_round = round_binary_vars(x, binary_idxs)
    assert (x_round == np.array([0, 0, 1, 0, 0, 1, 0.3])).all()
    # test 2
    binary_idxs = np.array([1, 2, 3, 4, 5])
    x_round = round_binary_vars(x, binary_idxs)
    assert (x_round == np.array([0.5, 0, 1, 0, 0, 1, 0.3])).all()
    # test 3: round everything
    binary_idxs = np.array([0, 1, 2, 3, 4, 5, 6])
    x_round = round_binary_vars(x, binary_idxs)
    assert (x_round == np.array([0, 0, 1, 0, 0, 1, 0])).all()


def test_flip_k():
    x_lp = np.array([0.4, 0.55, 1, 0, 0, 1, 0.3])
    binary_idxs = np.array([0, 1, 2])
    x_round = round_binary_vars(x_lp, binary_idxs)
    # test 1
    nb_flips = 2
    flipped_x = flip_top_k_binary(x_lp, x_round, binary_idxs, nb_flips)
    assert (flipped_x == np.array([1, 0, 1, 0, 0, 1, 0.3])).all()
    # test 1
    nb_flips = 1
    flipped_x = flip_top_k_binary(x_lp, x_round, binary_idxs, nb_flips)
    assert (flipped_x == np.array([0, 0, 1, 0, 0, 1, 0.3])).all()
    # test 1
    nb_flips = 3
    flipped_x = flip_top_k_binary(x_lp, x_round, binary_idxs, nb_flips)
    assert (flipped_x == np.array([1, 0, 0, 0, 0, 1, 0.3])).all()


def test_hamming_loss():
    x_round = np.array([0, 0, 1, 0, 0, 1, 0.3])
    binary_idxs = np.array([0, 1, 2, 3, 4])
    # test 1
    binary_mask = get_binary_mask(len(x_round), binary_idxs)
    hamm = hamming_update(x_round, binary_mask)
    assert (hamm == np.array([1, 1, -1, 1, 1, 0, 0])).all()
