"""
Differentiable feasibility pump
"""

from time import time

import numpy as np
import torch

from .first_iteration import first_iteration
from .losses import (
    CostLoss,
    FeasibilityLoss,
    FeasibilitySparseLoss,
    IntegralityLoss,
    RegularizationLoss,
)
from .modules import Normalization, SoftRounding
from .modules.argmin import get_gradient_estimator
from .restarts import (
    flip,
    flip_x_round,
    is_x_lp_cycling,
    is_x_round_same,
    perturb,
)
from .utils import get_binary_mask, get_optimizer, integ_metric


def diff_pump(solver, config):
    """
    Differentiable feasibility pump: gradient descent over generalized
    loss function. The loss function has four components:
        (1) cost of rounded solution according to initial cost vector,
        (2) non-integrality of argmin of LP,
        (3) infeasibility of rounded soliution,
        (4) regularization of cost vector.
    The gradient descent is performed to reach local optima, and restarts
    are applied to escape from it if needed.

    Args:
        solver (Solver): optimization model
        config (_type_): experiment parameters

    Returns:
        float: final total loss
        float: final non-integrality loss
        int: number of iterations
        np.array: solution of feasibility pump

    - - - - - - - - - - - - - - - - - - - - -

    N.B.  Make sure numpy and torch are working with the same number format.
          Numpy default is Float64 while torch default is Float32.
          Anyway, I think torch autograd functions always work with Float32.

    N.B.2 Try to avoid all checks of the kind "x == 0" and
          use "np.isclose(x,0)" instead.

    """
    tick = time()
    # - Read optimization model parameters -
    # Get indexes of MILP binary variables
    binary_idxs = solver.get_binary_vars()
    # Feasible region of the relaxed MILP is of the form Ax<=b
    A, b = solver.get_constr()

    # Convert numpy arrays to tensors
    init_cost = torch.DoubleTensor(solver.model.obj)

    # - Load loss functions and modules -
    # Load modules
    rounding = SoftRounding()
    normalization = Normalization(config.normtheta, config.proj_grad)
    # Load losses
    integrality_loss = IntegralityLoss(
        config.integ_metric, p=config.p, binary_idxs=binary_idxs
    )
    if config.denselinalg:
        feasibility_loss = FeasibilityLoss(A=A, b=b)
    else:
        feasibility_loss = FeasibilitySparseLoss(A=A, b=b)
    init_cost_loss = CostLoss(init_cost=init_cost)
    reg_loss = RegularizationLoss()

    # Set estimator for the jacobian of the argmin
    jac_estimator = get_gradient_estimator(solver, config.estim)

    # - Set pump parameters -
    # Read loss weights from confid
    w1 = config.initcost_loss
    w2 = config.integ_loss
    w3 = config.feas_loss
    w4 = config.reg_loss

    print("Built losses and Jacobian estimator"
          f" in {time() - tick:.3f} seconds.")
    # - Setting starting point and optimizer -
    # Perform the first iteration of the original FP algorithm
    theta, history = first_iteration(
        solver,
        binary_idxs,
        rounding=rounding,
        integrality_loss=integrality_loss,
        feasibility_loss=feasibility_loss,
        init_cost_loss=init_cost_loss,
        reg_loss=reg_loss,
        w=(w1, w2, w3, w4),
    )
    # Setup torch optimizer
    theta = torch.DoubleTensor(theta)
    theta = torch.nn.Parameter(theta)
    optimizer = get_optimizer(theta, config.optm, config.lr, config.momentum)

    # Get mask for non binary vars
    binary_mask = get_binary_mask(len(theta), binary_idxs)

    # If true, will heuristically detect cycles and perturb theta
    use_restarts = not config.no_restarts

    tick = time()
    for num_iters in range(config.iter):
        # Normalize cost vector
        theta_aux = normalization(theta)

        # Solve linear relaxation of original problem for cost_vector
        x_lp = jac_estimator(theta_aux)

        # Round solution
        x_round = rounding.forward(x_lp, binary_idxs)

        # Measure losses
        initcostLoss = init_cost_loss(x_round)
        integralityLoss = integrality_loss(x_lp)
        feasibilityLoss = feasibility_loss(x_round)

        regularizationLoss = reg_loss(theta)
        # Measure integrality metric
        integralityMetricVal = integ_metric(
            x_lp.detach().clone().numpy(), binary_idxs=binary_idxs
        )

        # Compute the total loss as a linear combination
        totalLoss = (
            w1 * initcostLoss
            + w2 * integralityLoss
            + w3 * feasibilityLoss
            + w4 * regularizationLoss
        )

        # - Read values, print and store -
        totalLossVal = totalLoss.item()
        integralityLossVal = integralityLoss.item()
        feasibilityLossVal = feasibilityLoss.item()
        initcostLossVal = initcostLoss.item()
        regularizationLossVal = regularizationLoss.item()
        elapsed = time() - tick
        print(
            f"Iter {num_iters}, "
            f"IntMetric: {integralityMetricVal:.2e}, "
            f"Totloss: {totalLossVal:.4f}, "
            f"costLoss: {initcostLossVal:.2f}, "
            f"intLoss: {integralityLossVal:.4f}, "
            f"feasLoss: {feasibilityLoss:.2e}, "
            f"regLoss: {regularizationLossVal:.2f}, "
            f"elapsed: {elapsed:.1f}"
        )
        # Store computation in history
        history["totalLoss"].append(totalLossVal)
        history["integralityLoss"].append(integralityLossVal)
        history["feasibilityLoss"].append(feasibilityLossVal)
        history["theta"].append(theta.detach().clone().numpy())
        history["x_round"].append(x_round.detach().clone().numpy())
        history["x_lp"].append(x_lp.detach().clone().numpy())

        # Exit when x_lp is feasible
        found_solution = np.isclose(integralityMetricVal, 0) or np.isclose(
            feasibilityLossVal, 0
        )
        if found_solution:
            break
        # Gradient descent step on generalized loss
        optimizer.zero_grad()
        totalLoss.backward()

        # Update theta
        optimizer.step()
        if use_restarts:
            # Perturb if argmin solution is same as previous iterations
            if (num_iters + 1) % 100 == 0 or is_x_lp_cycling(history["x_lp"]):
                print("- RESTART: Perturb at iter ", num_iters, " -\n")
                flipped_idxs = perturb(theta, x_lp, x_round, binary_mask)
                flip_x_round(history, flipped_idxs)
                history["n_restarts"] += 1
            # Flip if rounded solution is same as last iteration
            elif is_x_round_same(
                history["x_round"][-1][binary_idxs],
                history["x_round"][-2][binary_idxs],
            ):
                print("- RESTART: Flip at iter ", num_iters, " -\n")
                flipped_idxs = flip(x_lp, theta, binary_idxs)
                flip_x_round(history, flipped_idxs)
                history["n_flips"] += 1

    print("Flipped ", history["n_flips"], " times")
    print("Restarted ", history["n_restarts"], " times")
    loss, metric, feas = totalLossVal, integralityMetricVal, feasibilityLossVal
    nb_iter, sol = num_iters + 1, x_lp
    return loss, metric, feas, nb_iter, sol
