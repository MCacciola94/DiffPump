"""
Minus-Identity estimator
"""

from .base import AbstractEstimator, AbstractEstimatorFunction


class MinusIdEstimator(AbstractEstimator):
    """Uses minus the identity matrix as Jacobian estimator."""

    def __init__(self, solver):
        super().__init__(solver)
        self.func = MinusIdFunc


class MinusIdFunc(AbstractEstimatorFunction):
    @staticmethod
    def backward(_, grad_output):
        """Return downstream gradient multiplied by minus identity matrix."""
        return -grad_output, None
