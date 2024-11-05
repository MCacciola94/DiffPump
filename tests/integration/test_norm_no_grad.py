import warnings
from pathlib import Path

import pytest
from gurobipy import GurobiError
from numpy import isclose

from diffpump.parse import load_parser
from script.main import run_experiment

TEST_INSTANCE_FOLDER = Path("data") / "tests"


def _run_generalized_norm_no_grad(instance):
    parser = load_parser()
    config = parser.parse_args(
        [
            instance,
            "--lr", "0.9",
            "--reg_loss", "0.7",
            "--iter", "30",
            "--normtheta",
            "--addnoise",
        ]
    )
    return run_experiment(config, instance_folder=TEST_INSTANCE_FOLDER,
                          save_results=False)


def _run_generalized_no_norm(instance):
    parser = load_parser()
    config = parser.parse_args(
        [
            instance,
            "--lr", "0.9",
            "--reg_loss", "0.7",
            "--iter", "30",
            "--addnoise",
        ]
    )
    return run_experiment(config, instance_folder=TEST_INSTANCE_FOLDER,
                          save_results=False)


# Test on a few small-size instances
instances = ["cap6000", "harp2", "seymour"]


@pytest.mark.parametrize("instance", instances)
def test_generalized_norm_but_no_grad(instance):
    try:
        # Run feasibility pumps
        loss1, metric1, nb_epochs1 = _run_generalized_norm_no_grad(instance)
        loss2, metric2, nb_epochs2 = _run_generalized_no_norm(instance)
        gurobiLicenseAvailable = True
    except GurobiError:
        gurobiLicenseAvailable = False
        warnings.warn(
            "Warning: Gurobi license not found:"
            " cannot run integration test that solves MILP.",
            UserWarning,
        )

    # Check final results are identical
    if gurobiLicenseAvailable:
        assert isclose(loss1, loss2)
        assert isclose(metric1, metric2)
        assert isclose(nb_epochs1, nb_epochs2)
