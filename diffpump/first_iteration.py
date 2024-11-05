from time import time

import numpy as np
import torch


def first_iteration(
    model,
    binary_idxs,
    *,
    rounding,
    integrality_loss,
    feasibility_loss,
    init_cost_loss,
    reg_loss,
    w,
):
    """
    Perform the first iteration of the original FP algorithm.

    Args:
        model (grb.model): optimization model
        binary_idxs (list[int]): list of indices of binary-constrained vars

    Returns:
        theta, history

    """
    tick = time()

    # Save initial costs
    model.init_cost = np.array(model.model.obj)

    # Solve first iteration of feasibility pump with initial cost vector
    sol, _ = model.solve()
    # Round argmin solution
    sol_round = list(rounding.forward(torch.DoubleTensor(sol), binary_idxs))
    hamm = [(1 if val == 0 else -1) for val in sol_round]
    theta = np.array(hamm)
    x_round = np.array(sol_round)
    non_bi_idxs = [i for i in range(model.num_cost) if i not in binary_idxs]
    # Avoid unbounded problems
    theta[non_bi_idxs] = 0

    # Measure losses
    x_round_torch = torch.DoubleTensor(x_round)
    sol_torch = torch.DoubleTensor(sol)
    initcost_torch = torch.DoubleTensor(model.init_cost)
    initcostLossVal = init_cost_loss(x_round_torch).item()
    integralityLossVal = integrality_loss(sol_torch).item()
    feasibilityLossVal = feasibility_loss(x_round_torch).item()

    regularizationLossVal = reg_loss(initcost_torch).item()

    # Compute the total loss as a linear combination
    totalLossVal = (
        w[0] * initcostLossVal
        + w[1] * integralityLossVal
        + w[2] * feasibilityLossVal
        + w[3] * regularizationLossVal
    )
    elapsed = time() - tick
    print(
        f"Iter {-1}, Totloss: {totalLossVal:.4f},"
        f" costLoss: {initcostLossVal:.2f}, intLoss: {integralityLossVal:.4f}"
        f", feasLoss: {feasibilityLossVal:.4f}"
        f", regLoss: {regularizationLossVal:.2f}, elapsed: {elapsed:.1f}"
    )

    # Variables to keep information in memory
    history = {
        "totalLoss": [totalLossVal],
        "costLoss": [initcostLossVal],
        "reguLoss": [regularizationLossVal],
        "integralityLoss": [integralityLossVal],
        "feasibilityLoss": [feasibilityLossVal],
        "theta": [model.init_cost],
        "x_lp": [np.array(sol)],
        "x_round": [x_round],
        "n_flips": 0,
        "n_restarts": 0,
    }

    return theta, history
