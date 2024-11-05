#!/usr/bin/env python
"""
argmin
=======
Torch modules that solve argmin in forward and
estimate/approximate its Jacobian in backward.
"""

from .get_estimator import get_gradient_estimator
from .identity import MinusIdEstimator
from .perturbed import PerturbedEstimator

__all__ = [
    "MinusIdEstimator",
    "PerturbedEstimator",
    "get_gradient_estimator",
]
