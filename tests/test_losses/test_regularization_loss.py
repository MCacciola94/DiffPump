import numpy as np
import torch

from diffpump.losses import RegularizationLoss


def test_regularization_loss():
    reg_loss = RegularizationLoss()
    # Test 1: only positive values
    theta = np.array([1, 1, 0.5, 0.0, 1.0])
    theta = torch.DoubleTensor(theta)
    assert np.isclose(reg_loss(theta).item(), 0.5 * (1 + 1 + 0.25 + 1.0))
    # Test 2: positive and negative values
    theta = np.array([1, -1, -0.5, 0.0, 1.0])
    theta = torch.DoubleTensor(theta)
    assert np.isclose(reg_loss(theta).item(), 0.5 * (1 + 1 + 0.25 + 1.0))
