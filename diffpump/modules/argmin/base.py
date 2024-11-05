import numpy as np
import torch
from torch import nn
from torch.autograd import Function

from ...solver.solver import Solver


class AbstractEstimator(nn.Module):
    """
    Abstract class for Jacobian estimators.

    Args:
        solver (opt.Solver): optimization model to solve.
        func (autograd.Function): perform forward and backward pass.
            Forward is always solving model with input objective and
            saving everything in ctx.

    """

    def __init__(self, solver):
        super().__init__()
        if not isinstance(solver, Solver):
            msg = "Input model should be a solver."
            raise TypeError(msg)
        self.solver = solver

    def forward(self, obj):
        return self.func.apply(obj, self)


class AbstractEstimatorFunction(Function):
    """Abstract class for the function of the jacobian estimator."""

    @staticmethod
    def forward(ctx, obj, module):
        sol = AbstractEstimatorFunction.solve_mip(obj, module.solver)
        ctx.sol = sol
        ctx.obj = obj
        ctx.module = module
        return torch.DoubleTensor(sol)

    @staticmethod
    def backward(ctx, grad_output):
        """
        An abstract method to build a model from a optimization solver.

        Returns:
            tuple: gradients

        """
        raise NotImplementedError

    @staticmethod
    def solve_mip(obj, solver):
        """
        A function to solve optimization in the forward pass.
        """
        obj_vector = obj.detach().to("cpu").numpy()
        # Set objective to input cost vector and solve model.
        solver.setObj(obj_vector)
        sol, _ = solver.solve()
        return np.array(sol)
