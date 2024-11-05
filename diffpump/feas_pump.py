from copy import copy

import numpy as np
import torch


def feas_pump(solver, max_iter, history_length=2):
    """
    Implementation of original feasibility pump algorithm from:
        Fischetti, M., Glover, F., & Lodi, A. (2005). The feasibility pump.
        Mathematical Programming, 104, 91-104.

    This algorithm is used in tests to verify that our implementation
    of the differentiable pump recovers the exact same iteration and values
    than the original pump when the parameters are correctly specified.

    In our experiments, we always use the `differentiable_pump`.
    """
    num_variables = solver.num_cost

    # Get indices of binary variables
    binary_idxs = solver.get_binary_vars()
    # Create mask: equal to 1 for binary variables, 0 otherwise
    binary_mask = get_binary_mask(num_variables, binary_idxs)

    hamm = torch.DoubleTensor(solver.model.obj)

    # Solve linear relaxation of original problem
    x_lp = solve(solver)

    # Round argmin to nearest integer
    x_round_prev = round_binary_vars(x_lp, binary_idxs)

    # If x_lp is already binary, algorithm terminates
    if is_binary(x_lp, binary_idxs):
        return (0, 0)

    x_lp_history = []

    for num_iters in range(max_iter):
        # Save previous iterations
        x_lp_history.append(copy(x_lp))
        if len(x_lp_history) > history_length:
            x_lp_history.pop(0)

        # Update cost vector by projecting using Hamming distance
        hamm = hamming_update(x_round_prev, binary_mask)

        # Solve linear relaxation for new cost vector
        x_lp = solve(solver, cost_vector=hamm)

        #  Measure integrality loss
        loss = integrality_loss(x_lp, binary_idxs)
        metric = integ_metric(x_lp, binary_idxs=binary_idxs)
        print(f"Iteration {num_iters}, Loss: {loss.item():.4f}")

        if np.isclose(metric, 0):
            # Check if last rounding was admissible to have exact iter numb.
            if np.all(np.isclose(x_lp, x_round_prev)):
                num_iters += -1
            break  # Terminate algorithm
        x_round = round_binary_vars(x_lp, binary_idxs)
        # Check if algorithm is in a cycle
        if need_to_perturb(
            num_iters,
            history_length,
            x_lp,
            x_lp_history,
            use_isclose=True,
        ):
            print(" - RESTART: Perturb at iter ", num_iters, " -\n")
            x_round_prev = perturb(x_lp, x_round, binary_mask, num_variables)
        elif np.all(np.isclose(x_round[binary_idxs],
                               x_round_prev[binary_idxs])):
            print(" - RESTART: Flip at iter ", num_iters, " -\n")
            x_round_prev = flip(x_lp, x_round, binary_idxs, T=20)
        else:
            x_round_prev = x_round

    return loss.item(), metric, -1, num_iters + 1, hamm


def hamming_update(x_round, binary_mask):
    """
    Update cost vector as original feasibility pump.
    """
    zeros = x_round == 0
    return (2 * zeros - 1) * binary_mask


def solve(model, cost_vector=None):
    """Solve relaxed optimization model for given cost vector."""
    if cost_vector is not None:
        model.setObj(cost_vector)
    x_lp, _ = model.solve()
    return np.array(x_lp)


def get_binary_mask(num_variables, binary_idxs):
    """Get vector of size num_variables with ones at each binary variable."""
    # Check that binary idxs is not empty vector
    if len(binary_idxs) == 0:
        print("Error: no variables are binary!")
        raise ValueError
    binary_mask = np.zeros(num_variables)
    np.put(binary_mask, binary_idxs, 1)
    return binary_mask


def is_binary(x, binary_idxs):
    """Return True if the binary variables of x_lp are binary."""
    return np.isclose(integrality_loss(x, binary_idxs), 0)


def integrality_loss(x, binary_idxs):
    """Measure non-integrality of binary variables of x."""
    x_bin = x[binary_idxs]
    return np.minimum(x_bin, 1 - x_bin).sum()


def round_binary_vars(x_lp, binary_idxs):
    """
    Round only the binary variables of x_lp.

    Warning: numpy rounds down 0.5 to 0 by default.
    """
    x_round = copy(x_lp)
    x_round[binary_idxs] = np.around(x_round[binary_idxs])
    return x_round


def perturb(x_lp, x_round, binary_mask, num_variables):
    """
    Perturb x_round to escape cycle.

    Args:
        x_lp (np.array): argmin of linear relaxation
        x_round (np.array): rounded argmin
        binary_mask (np.array[{0,1}]): equal to 1 if variable is binary
        num_variables (int): number of variables (both binary and continuous)

    Returns:
        np.array: x_round with flipped binary variables

    """
    # Sample random perturbation from uniform distribution
    rho = np.random.uniform(-0.3, 0.7, num_variables)
    rho[rho < 0] = 0

    # Compute auxiliary test statistic
    aux = np.abs(x_lp - x_round) + rho
    # Flip variables that are binary and pass the test
    flip_indices = (aux > 0.5) * (binary_mask == 1)
    x_round[flip_indices] = 1 - x_round[flip_indices]
    return x_round


def flip(x_lp, x_round, binary_idxs, T=20):
    """
    Flip a random number of integer-constrained entries.

    Args:
        x_lp (np.array): argmin of linear relaxation
        x_round (np.array): rounded argmin
        binary_idxs (list[int]): list of binary indices
        T (int, optional): hyperparameter. Defaults to 20.

    Returns:
        np.array: x_round with flipped binary variables

    """
    # Sample random integers
    nb_flips = np.random.randint(0.5 * T, 1.5 * T + 1)
    return flip_top_k_binary(x_lp, x_round, binary_idxs, nb_flips)


def flip_top_k_binary(x_lp, x_round, binary_idxs, nb_flips):
    """
    Flip the top nb_flips integer-constrained
    entries with largest non-integrality.

    Args:
        x_lp (np.array): argmin of linear relaxation
        x_round (np.array): rounded argmin
        binary_idxs (list[int]): list of binary indices
        nb_flips (int): number of flips.

    Returns:
        np.array: x_round with flipped binary variables

    """
    x_round = copy(x_round)

    # Keep only binary variables
    x_binary = x_lp[binary_idxs]
    # Find indices of binary variables with largest non-integrality loss
    non_integrality = x_binary * (1 - x_binary)
    idxs = np.argsort(non_integrality, kind="stable")[-nb_flips:]
    # Flip top nb_flips binary variables
    binary_idxs = np.array(binary_idxs)
    x_round[binary_idxs[idxs]] = 1 - x_round[binary_idxs[idxs]]
    return x_round


def need_to_perturb(
    num_iters, history_length, x_lp, x_lp_history, use_isclose=True
):
    """
    Perturb every 100 iterations or if the solution x_lp is
    in a cycle.
    """
    return num_iters >= history_length - 1 and (
        (num_iters + 1) % 100 == 0
        or is_in_cycle(x_lp, x_lp_history, use_isclose=use_isclose)
    )


def is_in_cycle(x_lp, x_lp_history, use_isclose=True):
    """
    Check if x_lp is observed in x_lp_history.

    Returns:
        boolean: True if in cycle

    """
    if use_isclose:
        return any(
            np.isclose(np.linalg.norm(x_lp - x_prev), 0)
            for x_prev in x_lp_history
        )
    # This test has a very small tolerance, will return False more often.
    return any(np.linalg.norm(x_lp - x_prev) == 0 for x_prev in x_lp_history)


def integ_metric(vec, *, binary_idxs):
    vec = vec[binary_idxs]
    # Compute min(x, 1-x) for each component,
    minvec1mvec = np.minimum(vec, 1 - vec)
    return np.max(minvec1mvec)
