import warnings
from pathlib import Path

import numpy as np
import pytest
from gurobipy import GurobiError

from diffpump.parse import load_parser
from script.main import run_experiment

TEST_INSTANCE_FOLDER = Path("data") / "tests"


def _run_feasibility_pump(instance):
    parser = load_parser()
    config = parser.parse_args([instance, "--original"])
    return run_experiment(config, instance_folder=TEST_INSTANCE_FOLDER,
                          save_results=False)


def _run_differentiable_pump(instance):
    parser = load_parser()
    config = parser.parse_args([instance,
                                "--lr", "1.0",
                                "--reg_loss", "1.0"])
    return run_experiment(config, instance_folder=TEST_INSTANCE_FOLDER,
                          save_results=False)


# Test on a few small-size instances
instances = ["cap6000", "harp2", "markshare1", "seymour"]


@pytest.mark.parametrize("instance", instances)
def test_differentiable_reduces_to_original_algorithm(instance: str):
    try:
        # Run feasibility pumps
        _, metric1, nb_epochs1 = _run_feasibility_pump(instance)
        _, metric2, nb_epochs2 = _run_differentiable_pump(instance)
        has_license = True
    except GurobiError:
        has_license = False
        warnings.warn(
            "Warning: Gurobi license not found:"
            " cannot run integration test that solves MILP.",
            UserWarning,
        )

    if has_license:
        # Check final results are identical
        if nb_epochs1 < 98:
            assert np.isclose(metric1, 0)
        else:
            assert metric1 == metric2
        assert nb_epochs1 == nb_epochs2
