from copy import copy
from pathlib import Path

import numpy as np
import torch

from .solver import grb

ESTIM_ABR = {
    "perturbed": "pert",
    "minusId": "mID",
}

OUTPUT_PATH = Path("output") / "results"


def get_path_to_save_file(config):
    """
    Return path to file where results will be saved.
    """
    # Read experiment parameters and format filename
    method = ESTIM_ABR[config.estim]
    filename = f"INST_{config.instance}-ESTM_{method}"
    filename += f"-IT_{config.iter}-LR_{config.lr}"
    filename += f"-INTMET_{config.integ_metric}-P_{config.p}"
    filename += f"-OPT_{config.optm}"
    if config.integ_loss != 1.0:
        filename += f"-INTL_{config.integ_loss}"
    if config.feas_loss != 0:
        filename += f"-FEASL_{config.feas_loss}"
    if config.initcost_loss != 0:
        filename += f"-INCOSL_{config.initcost_loss}"
    if config.momentum != 0:
        filename += f"-MOM_{config.momentum}"

    is_original = "-ORIG_yes" if config.original else ""
    no_restarts = "-NOREST_no" if config.no_restarts else ""
    is_gd_projected = "-GDPROJ_yes" if config.proj_grad else ""
    is_norm = "-NORMTHETA_yes" if config.normtheta else ""
    filename += is_original + no_restarts + is_norm + is_gd_projected

    print(f"Experiment will be saved to file {filename}.csv")

    return OUTPUT_PATH / f"{filename}.csv"


def build_model(instance_path: Path, use_dense_linalg: bool):
    """
    Build optimization model.
    """
    if use_dense_linalg:
        model = grb.MIPModel(instance_path)
    else:
        model = grb.MIPSparseModel(instance_path)
    return model


def round_binary_vars(x_lp, binary_idxs):
    """
    Round only the binary variables of x_lp.

    Warning: numpy rounds 0.5 to 0 by default.
    """
    x_round = copy(x_lp)
    x_round[binary_idxs] = np.around(x_round[binary_idxs])
    return x_round


def get_binary_mask(num_variables, binary_idxs):
    """Get vector of size num_variables with ones at each binary variable."""
    # Check that binary idxs is not empty vector
    if len(binary_idxs) == 0:
        print("Error: no variables are binary!")
        raise ValueError
    binary_mask = np.zeros(num_variables)
    np.put(binary_mask, binary_idxs, 1)
    return binary_mask


def get_optimizer(theta, optimizer, lr, mom):
    def theta_params():
        yield theta

    if optimizer == "sgd":
        optimizer = torch.optim.SGD(theta_params(), lr=lr, momentum=mom)
    elif optimizer == "adam":
        optimizer = torch.optim.Adam(theta_params(), lr=lr)
    else:
        msg = "Unknown optimizer: use `sgd` or `adam`."
        raise ValueError(msg)

    return optimizer


def integ_metric(vec, *, binary_idxs):
    vec = vec[binary_idxs]
    # Compute min(x, 1-x) for each component,
    minvec1mvec = np.minimum(vec, 1 - vec)
    return np.max(minvec1mvec)
