import numpy as np
import torch
from scipy import sparse as sp

from diffpump.losses import FeasibilityArgminLoss

"""
Tests for the eq.(21) argmin feasibility loss used by DP5:
    g(x_int) = min_u (1/m) sum_j ReLU((A^int_j x + A^cont_j u - b_j))^q
The continuous variables u are re-optimized (L-BFGS-B) to the best feasible
completion; the gradient wrt x uses the envelope theorem.
"""


def _system():
    # Rows 0,1 have no continuous column -> the continuous solve cannot remove
    # their violation, so g depends on x and its gradient is non-trivial.
    A = sp.csr_matrix(np.array([[1., 0., 0., 0.],
                                [0., 1., 0., 0.],
                                [1., 1., 1., 0.],
                                [0., 1., 0., 1.]]))
    b = np.array([-0.5, -0.5, 0.0, 0.0])
    return A, b


def test_argmin_feasibility_gradient_q2():
    A, b = _system()
    loss = FeasibilityArgminLoss(A=A, b=b, int_idx=[0, 1], cont_idx=[2, 3], q=2)
    x = torch.tensor([0.8, 0.6], dtype=torch.double, requires_grad=True)
    g = loss(x)
    assert g.item() > 0            # the rounded point is infeasible
    g.backward()
    ana = x.grad.numpy().copy()

    h = 1e-6
    num = np.zeros(2)
    for i in range(2):
        xp = x.detach().clone(); xp[i] += h
        xm = x.detach().clone(); xm[i] -= h
        num[i] = (loss(xp).item() - loss(xm).item()) / (2 * h)
    assert np.max(np.abs(ana - num)) < 1e-4


def test_argmin_feasibility_feasible_is_zero():
    # A point that admits a feasible continuous completion has zero loss.
    A, b = _system()
    loss = FeasibilityArgminLoss(A=A, b=b, int_idx=[0, 1], cont_idx=[2, 3], q=2)
    x = torch.tensor([-1.0, -1.0], dtype=torch.double)  # rows 0,1 satisfied
    assert loss(x).item() == 0.0


def test_argmin_feasibility_pure_integer():
    # No continuous columns: reduces to the violation at the integer point.
    A, b = _system()
    loss = FeasibilityArgminLoss(A=A, b=b, int_idx=[0, 1, 2, 3], cont_idx=[], q=2)
    x = torch.tensor([0.8, 0.6, 0.0, 0.0], dtype=torch.double)
    assert loss(x).item() > 0
