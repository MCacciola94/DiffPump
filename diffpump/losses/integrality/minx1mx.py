import numpy as np
import torch


class Minx1mx(torch.autograd.Function):
    """
    Compute the minimum between x and 1-x for each component of vec,
    elevate them at order p and sums them.

    Args:
        vec (torch.Tensor): the vector on which to compute the loss
        p (float): the exponent of the loss

    Returns:
        torch.Tensor: integrality loss

    """

    @staticmethod
    def forward(ctx, vec, p, binary_idxs):
        if p <= 0:
            msg = "Order 'p' of integrality loss should be positive."
            raise ValueError(msg)

        vec = vec.detach().clone().numpy()

        # Saving original dimension of vec
        n = len(vec)
        if binary_idxs != []:
            vec = vec[binary_idxs]

        # Compute min(x, 1-x) for each component,
        # elevate at p and sum over all components
        minvec1mvec = np.minimum(vec, 1 - vec)
        if int(p) != p:
            minvec1mvec[minvec1mvec < 0] = 0

        loss = (minvec1mvec**p).sum()

        # Save for backward pass
        ctx.p = p
        ctx.minvec1mvec = minvec1mvec
        ctx.vec = vec
        ctx.binary_idxs = binary_idxs
        ctx.n = n

        return torch.tensor(loss).to(torch.float64)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute the gradient for backward pass.

        The code assumes that all the entries in vec
        are supposed to be in [0, 1], which is a consequence of
        filtering to binary values of the forward pass.

        If it finds a component 'x' such that 'min(x, 1-x) < 0'
        the code assumes there was a numerical approximation
        error and sets 'min(x, 1-x) = 0'.

        The analytical formula for each component 'x' of the gradient is:
            p * min(x, 1 - x)^(p - 1)    if min(x, 1 - x) = x
            - p * min(x, 1 - x)^(p - 1)  if min(x, 1 - x) = 1 - x

        Args:
            ctx: information from forward pass
            grad_output (torch.Tensor): gradient from last block of computation

        Returns:
            torch.Tensor: gradient

        - - - - - - - - - - - - - - - - - - - - -
        N.B. If modified, rouding of 0.5 (up or down) should be
             consistent everywere in the project.

        """
        p = ctx.p
        minvec1mvec = ctx.minvec1mvec
        vec = ctx.vec
        binary_idxs = ctx.binary_idxs
        n = ctx.n

        # Compute the indexes that needs to change sign.
        # Notice that, in the following line, "vec < 0.5" will produce
        # negative gradient for the entries that are equal to 0.5.
        # This  corresponds to rounding up those components when computing
        # the rounded solution. Rounding down is achieved by "vec <= 0.5".
        sign_multipliers = 2 * (vec <= 0.5) - 1
        # Fix negative entries due to numerical erros
        minvec1mvec[minvec1mvec < 0] = 0

        # Compute the gradient for the binary indexes
        grad = p * minvec1mvec ** (p - 1)
        grad = sign_multipliers * grad

        # Create the gradient with right number of components
        final_grad = torch.zeros(n).to(torch.float64)
        final_grad[binary_idxs] = torch.DoubleTensor(grad)

        return final_grad * grad_output, None, None


def minx1mx(vec, p, binary_idxs):
    return Minx1mx.apply(vec, p, binary_idxs)
