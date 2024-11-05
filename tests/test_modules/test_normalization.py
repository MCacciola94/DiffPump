import numpy as np
import pytest
import torch

from diffpump.modules import Normalization


def _get_gradient(theta, normalization):
    # Setup gradient descent optimizer
    theta = torch.nn.Parameter(theta)

    def theta_params():
        yield theta

    optimizer = torch.optim.SGD(theta_params(), lr=1.0, momentum=0.0)
    optimizer.zero_grad()
    theta_norm = normalization(theta)
    loss = theta_norm.sum()
    loss.backward()
    return theta.grad.detach().clone().numpy()


def normalized_theta(theta):
    return theta / torch.linalg.norm(theta)


@pytest.mark.parametrize("project_gradient", [True, False])
def test_forward_normalization(project_gradient: bool) -> None:
    """
    Normalize should normalize theta independently of gradient projection.
    """
    normalize_theta = True
    normalization = Normalization(normalize_theta, project_gradient)
    # Test 1
    theta = np.array([1, -1, -0.5, 0.0, 1.0])
    theta_tensor = torch.DoubleTensor(theta)
    assert np.isclose(
        normalization(theta_tensor), theta / np.linalg.norm(theta)
    ).all()
    # Test 2
    theta = np.array([1.654, -100, -0.5, 0.0, 1.0])
    theta_tensor = torch.DoubleTensor(theta)
    assert np.isclose(
        normalization(theta_tensor), theta / np.linalg.norm(theta)
    ).all()
    # Test 3
    theta = np.array([0.0001, 0.0006, -32])
    theta_tensor = torch.DoubleTensor(theta)
    assert np.isclose(
        normalization(theta_tensor), theta / np.linalg.norm(theta)
    ).all()


def test_backward_no_grad_normalization():
    normalize_theta = True
    project_gradient = False
    normalization = Normalization(normalize_theta, project_gradient)
    # test 1
    theta = np.array([1, -1, -0.5, 0.0, 1.0])
    theta = torch.DoubleTensor(theta)
    assert np.isclose(
        _get_gradient(theta, normalization), np.ones(len(theta))
    ).all()


def test_backward_normalization():
    normalize_theta = True
    project_gradient = True
    normalization = Normalization(normalize_theta, project_gradient)
    # test 1
    theta = np.array([1, -1, -0.5, 0.0, 1.0])
    theta = torch.DoubleTensor(theta)
    assert np.isclose(
        _get_gradient(theta, normalization),
        _get_gradient(theta, normalized_theta),
    ).all()
