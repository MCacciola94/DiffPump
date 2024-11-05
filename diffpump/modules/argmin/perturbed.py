"""
Perturbed Jacobian estimator
"""

import numpy as np
import torch

from .base import AbstractEstimator, AbstractEstimatorFunction


class PerturbedEstimator(AbstractEstimator):
    """
    Use perturbation method to estimate Jacobian.

    Adapted from PyEPO.
    """

    def __init__(self, solver, seed=123):
        super().__init__(solver)
        self.func = PerturbedEstimatorFunc()
        self.rnd = np.random.RandomState(seed)


class PerturbedEstimatorFunc(AbstractEstimatorFunction):
    @staticmethod
    def backward(ctx, grad_output, sigma: float = 1.0, n_samples: int = 1):
        rnd = ctx.module.rnd
        cp = ctx.obj.detach().to("cpu").numpy()

        # Sample perturbations and add them to cost vector
        noises = rnd.normal(0, sigma, size=(n_samples, *cp.shape))
        ptb_c = cp + sigma * noises
        ptb_sols = PerturbedEstimatorFunc._solve_in_forward(ptb_c, ctx)

        noises = torch.DoubleTensor(noises).unsqueeze(1)
        ptb_sols = torch.DoubleTensor(ptb_sols)
        grad_output = grad_output.unsqueeze(0)
        grad = torch.einsum(
            "nbd,bn->bd",
            noises,
            torch.einsum("bnd,bd->bn", ptb_sols, grad_output),
        )
        grad /= n_samples * sigma

        return grad.squeeze(0), None, None, None, None, None, None, None, None

    @staticmethod
    def _solve_in_forward(ptb_c, ctx):
        solver = ctx.module.solver
        n_samples = ptb_c.shape[0]
        ptb_sols = []
        sols = []
        for j in range(n_samples):
            solver.setObj(ptb_c[j])
            sol, _ = solver.solve()
            sols.append(sol)
        ptb_sols.append(sols)

        return np.array(ptb_sols)
