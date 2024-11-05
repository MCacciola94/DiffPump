import numpy as np
import torch

from diffpump.losses import FeasibilityLoss

"""
Test for feasibility loss: ReLU(Ax-b).

Note that A and b are normalized when computing the loss.
"""


def test_feasibility_loss():
    A = np.ones((5, 5))
    b = np.array([1, 2, 3, 4, 5])
    A = torch.DoubleTensor(A)
    b = torch.DoubleTensor(b)
    norms = (A.norm(dim=1) ** 2 + b**2) ** (0.5)
    feasibility_loss = FeasibilityLoss(A=A, b=b)
    # Test 1: all variables tight.
    x = np.array([0, 0, 0, 0, 0])
    x = torch.DoubleTensor(x)
    assert np.isclose(feasibility_loss(x), 0.0)
    # Test 2: loss is negligible when constraint is tight.
    x = np.array([0, 0, 0, 0, 1])
    x = torch.DoubleTensor(x)
    assert feasibility_loss(x) <= 1e-12
    # Test 3: there is large loss when constraint is violated.
    x = np.array([1, 0, 0, 0, 1])
    x = torch.DoubleTensor(x)
    assert np.isclose(feasibility_loss(x), (1.0 / len(b)) / norms[0])
    # Test 4: there is large loss when constraint is violated.
    x = np.array([1, 1, 1, 1, 1])
    x = torch.DoubleTensor(x)
    loss = feasibility_loss(x)
    expected_result = 4 / norms[0] + 3 / norms[1] + 2 / norms[2] + 1 / norms[3]
    assert np.isclose(loss, expected_result / len(b))
