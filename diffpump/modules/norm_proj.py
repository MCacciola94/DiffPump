"""
Torch module for normalization and projection:
allow experimenting with projected gradient and/or normalized
cost vectors.

The idea behind projecting the gradient is that,
because LP are invariant to scale of input,
we only need to propagate the gradient information orthogonal
to the cost vector.

Not used in paper
"""

import numpy as np
import torch


class Normalization(torch.nn.Module):
    def __init__(self, normalize_theta, project_gradient):
        super().__init__()
        self.normalize_theta = normalize_theta
        # If use_grad_norm is True, compute gradient of normalization in
        # backward pass. Otherwise, simply propagate gradient
        self.project_gradient = project_gradient

    def forward(self, theta):
        if self.normalize_theta:
            if self.project_gradient:
                # Normalize theta and compute projected gradient
                return NormalizeFun.apply(theta)
            # Normalize but don't compute gradients
            return NormalizeFunNoGrad.apply(theta)
        if self.project_gradient:
            # Don't normalize but project gradient
            return NoNormButProjectedGrad.apply(theta)
        # Just return theta
        return theta


class NormalizeFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta):
        norm_theta = torch.linalg.norm(theta, ord=2)
        theta_normalized = theta / norm_theta
        ctx.save_for_backward(theta, norm_theta, theta_normalized)
        return theta_normalized

    @staticmethod
    def backward(ctx, grad_output):
        return gradient_of_projection(ctx) * grad_output, None


class NoNormButProjectedGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta):
        norm_theta = torch.linalg.norm(theta, ord=2)
        theta_normalized = theta / norm_theta
        ctx.save_for_backward(theta, norm_theta, theta_normalized)
        return theta

    @staticmethod
    def backward(ctx, grad_output):
        return gradient_of_projection(ctx) * grad_output, None


class NormalizeFunNoGrad(torch.autograd.Function):
    @staticmethod
    def forward(_, theta):
        theta_np = theta.detach().clone().numpy()
        norm_theta = np.linalg.norm(theta_np, ord=2)
        theta_normalized = theta_np / norm_theta
        return torch.DoubleTensor(theta_normalized)

    @staticmethod
    def backward(_, grad_output):
        # Equivalent to grad = Identity matrix
        return grad_output, None


def gradient_of_projection(ctx):
    """Project gradient orthogonal to theta."""
    theta, norm_theta, theta_normalized = ctx.saved_tensors
    # Transform vector into a column matrix
    theta_normalized = theta_normalized.unsqueeze(-1)
    theta_prod = torch.matmul(
        theta_normalized, torch.transpose(theta_normalized, 0, 1)
    )
    return (1 / norm_theta) * (torch.eye(len(theta)) - theta_prod)
