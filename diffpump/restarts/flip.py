import warnings

import numpy as np
import torch


def flip(x_lp, theta, binary_idxs, T=20):
    """
    Flip a random number of integer-constrained entries.

    Args:
        x_lp (torch.tensor): argmin of linear relaxation
        theta (torch.tensor): current cost vector
        binary_idxs (list[int]): list of binary indices
        T (int, optional): hyperparameter. Defaults to 20.

    Returns:
        np.array: indices to flip

    """
    # Compute random number of indices to flip
    nb_flips = np.random.randint(0.5 * T, 1.5 * T + 1)

    # Get only binary variables of x_lp as numpy array
    x_lp = x_lp.detach().clone().numpy()
    x_binary = x_lp[binary_idxs]

    # Find indices with largest non-integrality loss
    idxs_to_flip = find_indices_largest_loss(x_binary, binary_idxs, nb_flips)

    # Flip theta coordinates
    flip_theta(theta, idxs_to_flip)
    return idxs_to_flip


def flip_theta(theta, idxs_to_flip):
    """Simply change sign of theta at given indices."""
    with torch.no_grad():
        if np.all(np.isclose(theta[idxs_to_flip], 0)):
            warnings.warn(
                "\nWarning: All components to flip are actually zero.",
                UserWarning,
            )
            print("This should be a rare edge case!\n")

            theta[idxs_to_flip] = torch.DoubleTensor(
                np.random.randint(low=-1, high=2, size=len(idxs_to_flip))
            )
        else:
            theta[idxs_to_flip] = -theta[idxs_to_flip]


def flip_x_round(history, flipped_idxs):
    """Apply flip and perturb modifications to x_round."""
    x_round_flipped = 1 - history["x_round"][-1][flipped_idxs]
    history["x_round"][-1][flipped_idxs] = x_round_flipped


def find_indices_largest_loss(x_binary, binary_idxs, nb_flips):
    """
    Return the top nb_flips indices of x_binary that have the
    the largest non-integrality loss, measured as x(1-x)
    """
    # Check input consistency
    if nb_flips < 1:
        print("Cannot flip 0 indices.")
        raise ValueError
    if nb_flips > len(x_binary):
        warnings.warn(
            "Warning: Cannot flip more indices than dimension of x_binary.",
            UserWarning,
        )
        print("Taking nb_flips = len(x_binary)")
        nb_flips = len(x_binary)
    # Find indices with largest loss
    non_integrality = x_binary * (1 - x_binary)
    idxs_binary_to_flip = np.argsort(non_integrality, kind="stable")[
        -nb_flips:
    ]
    return np.array(binary_idxs)[idxs_binary_to_flip]
