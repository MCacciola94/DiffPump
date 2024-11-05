import numpy as np
import torch

from diffpump.modules import SoftRounding


def _get_gradient(x, binary_idxs, rounding):
    # Setup gradient descent optimizer
    x = torch.nn.Parameter(x)

    def x_params():
        yield x

    optimizer = torch.optim.SGD(x_params(), lr=1.0, momentum=0.0)
    optimizer.zero_grad()
    theta_norm = rounding(x, binary_idxs)
    loss = theta_norm.sum()
    loss.backward()
    return x.grad.detach().clone().numpy()


def test_forward_rounding():
    rounding = SoftRounding()
    # test 1
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9, 1.0])
    binary_idxs = np.array([0, 1, 2, 3, 4, 5, 7])
    x_tensor = torch.DoubleTensor(x)
    numpy_result = np.around(x_tensor)
    numpy_result[6] = x[6]
    expected_result = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.9, 1.0])
    assert np.isclose(rounding(x_tensor, binary_idxs), numpy_result).all()
    assert np.isclose(rounding(x_tensor, binary_idxs), expected_result).all()


def standard_gaussian_pdf(x):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x**2))


def test_backward_rounding():
    rounding = SoftRounding()
    # test 1
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9, 1.0])
    binary_idxs = np.array([0, 1, 2, 3, 4, 5, 7])
    x_tensor = torch.DoubleTensor(x)
    expected_gradient = (1 / 0.15) * standard_gaussian_pdf((0.5 - x) / 0.15)
    assert np.isclose(
        _get_gradient(x_tensor, binary_idxs, rounding), expected_gradient
    ).all()
