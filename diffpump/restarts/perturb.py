"""
Perturbation restart operation.

In the code, we keep the terminology of the original feasibility,
i.e., using two restart operations: `flip` and `perturb`.

N.B.: not to be confused with perturbed argmin estimator.
"""
import numpy as np

from .flip import flip_theta


def perturb(theta, x_lp, x_round, binary_mask):
    """
    Perturb to escape cycles.

    Args:
        theta (torch.tensor): current cost vector
        x_lp (torch.tensor): argmin of linear relaxation
        x_round (torch.tensor): rounded argmin
        binary_mask (np.array[{0,1}]): equal to 1 if variable is binary

    Returns:
        np.array: x_round with flipped binary variables

    """
    x_lp = x_lp.detach().clone().numpy()
    x_round = x_round.detach().clone().numpy()

    # Find where to perturb
    idxs_to_flip = find_indices_to_perturb(x_lp, x_round, binary_mask)
    flip_theta(theta, idxs_to_flip)

    # Return indices of flipped variables
    return idxs_to_flip


def find_indices_to_perturb(x_lp, x_round, binary_mask):
    """
    Find indices to perturb following original feasibility pump.

    Args:
        x_lp (torch.tensor): argmin of linear relaxation
        x_round (torch.tensor): rounded argmin
        binary_mask (np.array[{0,1}]): equal to 1 if variable is binary

    Returns:
        list[int]: indices at which to flip theta

    """
    rho = np.random.uniform(-0.3, 0.7, x_lp.shape[0])
    rho[rho < 0] = 0
    # Compute auxiliary test statistic
    aux = np.abs(x_lp - x_round) + rho
    # Get variables that are binary and pass the test
    idxs_to_flip = (aux > 0.5) * (binary_mask == 1)
    return idxs_to_flip.nonzero()[0]
