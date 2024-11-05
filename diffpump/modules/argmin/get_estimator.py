from .identity import MinusIdEstimator
from .perturbed import PerturbedEstimator


def get_gradient_estimator(solver, method):
    """Initialize and return Jacobian estimator."""
    if method == "minusId":
        # Estimate Jacobian using minus-identity matrix
        estimator = MinusIdEstimator(solver)
    elif method == "perturbed":
        # Estimate Jacobian using additive perturbation
        estimator = PerturbedEstimator(solver)
    else:
        msg = "Estimator ", method, " not found."
        raise ValueError(msg)

    return estimator
