import numpy as np

from diffpump.restarts import (
    is_x_lp_cycling,
    is_x_round_same,
)


def test_is_x_lp_cycling():
    history = [np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3])]
    # Test 1: history not long enough, should return False
    assert not is_x_lp_cycling(history, win_size=3)
    # Test 2: history long enough and cycle, should return True
    history.append(np.array([0, 1, 2, 3]))
    assert is_x_lp_cycling(history, win_size=3)
    # Test 3: different x_lp should return False
    history.append(np.array([5, 1, 2, 3]))
    history.append(np.array([6, 1, 2, 3]))
    history.append(np.array([7, 1, 2, 3]))
    assert not is_x_lp_cycling(history, win_size=4)
    assert not is_x_lp_cycling(history, win_size=3)
    assert not is_x_lp_cycling(history, win_size=2)
    # Test 4: with window of size 3, should detect cycle with one step
    history.append(np.array([6, 1, 2, 3]))
    assert is_x_lp_cycling(history, win_size=3)
    # Test 5: cycle but outside of window size
    history.append(np.array([5, 1, 2, 3]))
    assert not is_x_lp_cycling(history, win_size=3)
    assert not is_x_lp_cycling(history, win_size=4)
    assert is_x_lp_cycling(history, win_size=5)


def test_is_x_round_same():
    # Test 1: equal vectors
    x1 = np.array([0, 1, 2, 3])
    x2 = np.array([0, 1, 2, 3])
    assert is_x_round_same(x1, x2)
    # Test 2: different vectors
    x1 = np.array([0, 1, 2, 3])
    x2 = np.array([4, 1, 2, 3])
    assert not is_x_round_same(x1, x2)
    # Test 3: almost equal vectors should return True
    x1 = np.array([0, 1, 2, 3])
    x2 = np.array([0, 1, 2, 3 + 1e-11])
    assert is_x_round_same(x1, x2)
