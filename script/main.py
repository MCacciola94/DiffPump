"""
Run experiments with the differentiable feasibility pump
"""

import random
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from diffpump import diff_pump, feas_pump
from diffpump.parse import load_parser
from diffpump.solver import Solver
from diffpump.utils import build_model, get_path_to_save_file

DEFAULT_INSTANCE_FOLDER = Path("data") / "MIPLIB"


def run_experiment(parser: ArgumentParser,
                   instance_folder: Path = DEFAULT_INSTANCE_FOLDER,
                   save_results: bool = True) -> tuple[float, float, int]:
    """
    Run a single experiments with given configuration.
    The experiment may be repeated over multiple seeds.
    The results are stored in a csv file.

    Args:
        parser (ArgumentParser): experiment parameters
        instance_folder(Path): path to the folder containing the instances
        save_results (bool): whether to save the results to a csv file

    Returns:
        tuple[float, float, int]: loss, metric, nb_iter

    """
    instance_path = instance_folder / f"{parser.instance}.mps"
    save_path = get_path_to_save_file(parser)
    result_df = pd.DataFrame(columns=["Loss", "Metric", "Feas",
                                      "Elapsed", "Nb_epochs"])

    for i in range(parser.expnum):
        print("============================================================")
        print(f"Repetition {i+1} out of {parser.expnum}:")
        print("============================================================\n")
        # Set random seed
        random.seed(i)
        np.random.seed(i)
        torch.manual_seed(i)
        torch.cuda.manual_seed(i)

        # Build optimization model
        print(f"Building model for instance {parser.instance} with GurobiPy.")
        print(f"Instance path: {instance_path}")
        model = build_model(instance_path, parser.denselinalg)

        if parser.addnoise:
            # Add noise to the constraints: testing purposes only
            add_noise_to_lhs_constraints(model)

        # - Run feasibility pump -
        tick = time.time()
        if parser.original:
            loss, metric, feas, nb_iter, _ = feas_pump(model, parser.iter)
        else:
            loss, metric, feas, nb_iter, _ = diff_pump(model, parser)
        tock = time.time()
        elapsed = tock - tick

        # - Printing and saving -
        print(f"Experimented lasted {elapsed:.2f} sec. \n")
        if save_results:
            result_df = save_results_to_csv(loss, metric, feas, elapsed,
                                            nb_iter, result_df, save_path)
    return loss, metric, nb_iter


def save_results_to_csv(
        loss: float, metric: float, feas: float,
        elapsed: float, nb_iter: int,
        result_df: pd.DataFrame, save_path: Path) -> pd.DataFrame:
    """Save key performance indicators to .csv file."""
    # Create row for last experiment
    row = {"Loss": loss, "Metric": metric, "Feas": feas,
           "Elapsed": elapsed, "Nb. iter:": nb_iter}
    # Add row to current data frame if it is not empty
    result_df = pd.concat(
        [result_df if not result_df.empty else None, pd.DataFrame([row])],
        ignore_index=True)
    # Create folder to save_path if it does not exist
    save_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(save_path, index=False)
    return result_df


def add_noise_to_lhs_constraints(solver: Solver) -> None:
    """
    Adds a small perturbation to all coefficients of the LHS matrix.
    This breaks the symmetry in the solutions, so that normalizing
    the cost vectors does not change the LP solution found.

    This function should only be used within tests.
    """
    print("Adding small noise to LHS constraint matrix `A`.")
    constraints = solver.model.getConstrs()
    for const in constraints:
        row = solver.model.getRow(const)
        for i in range(row.size()):
            coeff = row.getCoeff(i)
            var = row.getVar(i)
            solver.model.chgCoeff(const, var, coeff + 0.01 * np.random.rand())


if __name__ == "__main__":
    parser = load_parser()
    config = parser.parse_args()
    run_experiment(config)
