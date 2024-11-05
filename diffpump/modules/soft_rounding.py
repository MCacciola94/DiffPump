"""
Torch module for differentiable rounding.
"""

import torch

from ..utils import round_binary_vars


class SoftRounding(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)

    def forward(self, x_lp, binary_idxs):
        return SoftRoundFunction.apply(x_lp, binary_idxs)


class SoftRoundFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_lp, binary_idxs):
        ctx.save_for_backward(x_lp)
        x_lp = x_lp.detach().clone().numpy()
        x_round = round_binary_vars(x_lp, binary_idxs)
        return torch.DoubleTensor(x_round)

    @staticmethod
    def backward(ctx, grad_output, epsilon=0.15):
        (x_lp,) = ctx.saved_tensors
        dist = torch.distributions.normal.Normal(0, 1)
        grad_x = 1 / epsilon * torch.exp(dist.log_prob((x_lp - 0.5) / epsilon))
        grad_x = grad_x * grad_output
        return grad_x, None
