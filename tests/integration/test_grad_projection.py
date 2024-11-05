import warnings
from pathlib import Path

import pytest
from gurobipy import GurobiError
from numpy import isclose

from diffpump.parse import load_parser
from script.main import run_experiment

TEST_INSTANCE_FOLDER = Path("data") / "tests"


def _run_generalized_normalized(instance):
    parser = load_parser()
    config = parser.parse_args(
        [
            instance,
            "--lr", "0.9",
            "--reg_loss", "1.0",
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
            "--reg_loss", "1.0",
            "--iter", "30",
            "--addnoise",
        ]
    )
    return run_experiment(config, instance_folder=TEST_INSTANCE_FOLDER,
                          save_results=False)


# Test on a few small-size instances
instances = ["cap6000", "harp2", "seymour"]


@pytest.mark.parametrize("instance", instances)
def test_projected_gradient_independent_normalization(instance):
    """
    Test projected gradient with and without normalization.

    Args:
        instance (string): path to instance .mps file.

    """
    try:
        # Run feasibility pumps
        _, metric1, nb_epochs1 = _run_generalized_normalized(instance)
        _, metric2, nb_epochs2 = _run_generalized_no_norm(instance)
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
        assert isclose(nb_epochs1, nb_epochs2)
        assert isclose(metric1, metric2)
