import numpy as np
import torch

from diffpump.losses import IntegralityLoss


def _get_gradient(x, integrality_loss):
    """Comput a single gradient using torch."""
    # Setup gradient descent optimizer
    x = torch.nn.Parameter(x)

    def x_params():
        yield x

    optimizer = torch.optim.SGD(x_params(), lr=1.0, momentum=0.0)
    optimizer.zero_grad()
    loss = integrality_loss(x)
    loss.backward()
    return x.grad.detach().clone().numpy()


def test_foward_minx1mx_loss():
    binary_idxs = [2, 3, 4, 5]
    p = 1
    integrality_loss = IntegralityLoss("minx1mx", p=p, binary_idxs=binary_idxs)
    # Test 1
    x = torch.DoubleTensor(np.array([1, 0.321, 0.5, 0.0, 0.8, 1.0]))
    assert integrality_loss(x) == (0.5 + 0.2)
    # Test 2
    x = torch.DoubleTensor(np.array([0.0, 0.321, 0.5, 0.0, 0.8, 0.0]))
    assert integrality_loss(x) == (0.5 + 0.2)


def test_backward_minx1mx_loss():
    binary_idxs = [0, 1, 2, 3, 4, 5]
    p = 1
    integrality_loss = IntegralityLoss("minx1mx", p=p, binary_idxs=binary_idxs)
    # Test 1
    x = torch.DoubleTensor(np.array([0.654, 0.321, 0.5, 0.0, 0.8, 1.0]))
    # NOTE: 0.5 is rounded down currently by default
    # Change third gradient to +1.0 if this is changed
    assert (
        _get_gradient(x, integrality_loss)
        == np.array([-1.0, 1.0, 1.0, 1.0, -1.0, -1.0])
    ).all()
    # Test 2
    x = torch.DoubleTensor(
        np.array([0.00001, 0.1, 0.3, 0.6, 0.7, 0.9999, 0.6543216])
    )
    assert (
        _get_gradient(x, integrality_loss)
        == np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 0.0])
    ).all()


def test_foward_x1mx_loss():
    binary_idxs = [2, 3, 4, 5]
    p = 1
    integrality_loss = IntegralityLoss("x1mx", p=p, binary_idxs=binary_idxs)
    # Test 1
    x = torch.DoubleTensor(np.array([1, 0.321, 0.5, 0.0, 0.8, 1.0]))
    assert np.isclose(integrality_loss(x), (0.5 * 0.5 + 0.2 * 0.8))
    # Test 2
    x = torch.DoubleTensor(np.array([0.0, 0.321, 0.5, 0.0, 0.8, 0.0]))
    assert np.isclose(integrality_loss(x), (0.5 * 0.5 + 0.2 * 0.8))


def test_backward_x1mx_loss():
    binary_idxs = [0, 1, 2, 3, 4, 5]
    p = 1
    integrality_loss = IntegralityLoss("x1mx", p=p, binary_idxs=binary_idxs)
    # Test 1
    x = torch.DoubleTensor(np.array([0.654, 0.321, 0.5, 0.0, 0.8, 1.0]))
    # NOTE: 0.5 is rounded down currently by default
    # Change third gradient to +1.0 if this is changed
    assert np.isclose(_get_gradient(x, integrality_loss), (1 - 2 * x)).all()
    # Test 2
    x = torch.DoubleTensor(
        np.array([0.00001, 0.1, 0.3, 0.6, 0.7, 0.9999, 0.6543216])
    )
    grad = 1 - 2 * x
    grad[-1] = 0.0
    assert np.isclose(_get_gradient(x, integrality_loss), grad).all()
