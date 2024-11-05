import numpy as np

from diffpump.utils import integ_metric


def test_integ_metric():
    binary_idxs = [2, 3, 5]
    # Test 1: all binary indices have binary values
    x = np.array([0.5, 0.4, 0.0, 1.0, 0.6, 1.0])
    assert integ_metric(x, binary_idxs=binary_idxs) == 0
    # Test 2: binary indices have non-binary values
    x = np.array([0.0, 0.15, 0.45, 0.7, 0.8, 0.51])
    assert integ_metric(x, binary_idxs=binary_idxs) == 0.49
