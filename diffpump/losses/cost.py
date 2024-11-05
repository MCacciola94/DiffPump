import torch

from .base import AbstractLoss


class CostLoss(AbstractLoss):
    def __init__(self, loss_name="linear_loss", *, init_cost=None):
        super().__init__(loss_name)
        self.init_cost = init_cost

    def linear_loss(self, vec: torch.DoubleTensor) -> torch.DoubleTensor:
        """Measure cost of input vector w.r.t. original cost vector."""
        return self.init_cost.dot(vec)
