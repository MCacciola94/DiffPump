from pathlib import Path

import numpy as np
import torch

from diffpump.losses import FeasibilitySparseLoss
from diffpump.solver.grb import MIPSparseModel

path = Path("data") / "tests" / "dummy.mps"


def test_init():
    model = MIPSparseModel(path)
    A, b = model.get_constr()
    feasibility_loss = FeasibilitySparseLoss(A=A, b=b)
    A_norm, b_norm = feasibility_loss.A, feasibility_loss.b
    assert len(b_norm) == 4
    assert A_norm.shape[1] == 3
    assert np.isclose(A_norm[0, 1], 0.7071067811865476)
    assert np.isclose(b_norm[3], -0.5773502691896257)


def test_loss_val():
    model = MIPSparseModel(path)
    A, b = model.get_constr()
    feasibility_loss = FeasibilitySparseLoss(A=A, b=b)
    x1 = torch.DoubleTensor([1, 1, 5])
    x2 = torch.DoubleTensor([0, 1, 0])
    x3 = torch.DoubleTensor([1, 0, 4.5])
    loss1 = feasibility_loss(x1).item()
    loss2 = feasibility_loss(x2).item()
    loss3 = feasibility_loss(x3).item()

    exp_val1 = 1 / (4 * 3 ** (0.5))
    exp_val2 = 1 / (4 * 2 ** (0.5))
    assert np.isclose(loss1, exp_val1)
    assert np.isclose(loss2, exp_val2)
    assert np.isclose(loss3, 0)


def test_loss_grad():
    model = MIPSparseModel(path)
    A, b = model.get_constr()
    feasibility_loss = FeasibilitySparseLoss(A=A, b=b)
    x1 = torch.DoubleTensor([1, 1, 5])
    x2 = torch.DoubleTensor([0, 1, 0])
    x3 = torch.DoubleTensor([1, 0, 4.5])
    x1 = torch.nn.Parameter(x1)
    x2 = torch.nn.Parameter(x2)
    x3 = torch.nn.Parameter(x3)
    loss1 = feasibility_loss(x1)
    loss2 = feasibility_loss(x2)
    loss3 = feasibility_loss(x3)
    loss1.backward()
    loss2.backward()
    loss3.backward()

    assert x1.grad[0] > 0 and x1.grad[1] > 0 and x1.grad[2] == 0
    assert x2.grad[0] == 0 and x2.grad[1] > 0 and x2.grad[2] < 0
    assert x3.grad[0] == 0 and x3.grad[1] == 0 and x3.grad[2] == 0
