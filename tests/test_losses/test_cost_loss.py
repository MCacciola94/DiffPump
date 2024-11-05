import numpy as np
import torch

from diffpump.losses import CostLoss


def test_init_cost_loss():
    c = np.array([1, 2, 3, 4, 5])
    c = torch.DoubleTensor(c)
    cost_loss = CostLoss(init_cost=c)
    # Test 1: all variables tight
    x = np.array([1, 1, 0.5, 0.0, 1.0])
    x = torch.DoubleTensor(x)
    assert cost_loss(x) == (1 + 2 + 1.5 + 0.0 + 5)
