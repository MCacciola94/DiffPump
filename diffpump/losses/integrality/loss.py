import torch

from ..base import AbstractLoss
from .minx1mx import minx1mx


class IntegralityLoss(AbstractLoss):
    def __init__(
        self,
        loss_name: str,
        *,
        p: float = 1,
        binary_idxs=None,
    ) -> None:
        super().__init__(loss_name)
        if binary_idxs is None or binary_idxs == []:
            msg = "No binary indices specified."
            raise ValueError(msg)
        self.p = p
        self.binary_idxs = binary_idxs

    def x1mx(self, vec: torch.DoubleTensor) -> torch.DoubleTensor:
        """Integrality loss: f(x) = x * (1 - x)"""
        if self.binary_idxs != []:
            vec = vec[self.binary_idxs]
        return (vec * (1 - vec)).sum()

    def minx1mx(self, vec: torch.DoubleTensor) -> torch.DoubleTensor:
        """Integrality loss: f(x) = min{x, 1-x}^p"""
        return minx1mx(vec, p=self.p, binary_idxs=self.binary_idxs)
