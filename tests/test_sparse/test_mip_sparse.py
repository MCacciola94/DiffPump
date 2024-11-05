from pathlib import Path

import numpy as np
import torch
from scipy import sparse as sp

from diffpump.solver.grb import MIPSparseModel

path = Path("data") / "tests" / "dummy.mps"


def test_init():
    model = MIPSparseModel(path)
    assert type(model.A) is sp._csr.csr_matrix
    assert type(model.b) is np.ndarray
    assert type(model.sense) is np.ndarray
    assert len(model.sense) == len(model.b)
    assert len(model.b) == model.A.shape[0]
    assert model.num_cost == model.A.shape[1]


def test_get_const():
    model = MIPSparseModel(path)

    ineqs, b = model.get_ineq_constr()
    assert len(b) == 2
    assert ineqs.toarray()[1][0] == 3

    eqs, b = model.get_eq_constr()
    assert eqs.shape[0] == 1
    assert b[0] == 1

    A, b = model.get_constr()
    assert A.shape[0] == 4
    assert b[0] == 0
    assert A.toarray()[3][0] == -1


def test_check_slacks():
    model = MIPSparseModel(path)

    x1 = torch.Tensor([1, 1, 5])
    x2 = torch.Tensor([1, 0, 2])
    x3 = torch.Tensor([0, 1, -2])
    x4 = torch.Tensor([0, 1, 2])
    x5 = torch.Tensor([1, 0, 3])

    assert not model.check_feasibility(x1)
    assert not model.check_feasibility(x2)
    assert not model.check_feasibility(x3)
    assert model.check_feasibility(x4)
    assert model.check_feasibility(x5)

    A1, _ = model.get_active_constr(x1)
    _, b2 = model.get_active_constr(x2)
    _, b3 = model.get_active_constr(x3)
    A5, _ = model.get_active_constr(x5)

    assert A1.shape[0] == 0
    assert np.isclose(b2[0], 0.5773502691896257)
    assert np.isclose(b3[1], -0.5773502691896257)
    assert np.isclose(A5[0, 0], 0.9486832980505138)
