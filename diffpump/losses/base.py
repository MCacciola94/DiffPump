from torch import nn


class AbstractLoss(nn.Module):
    """Abstract class for all loss functions."""

    def __init__(self, loss_name: str) -> None:
        super().__init__()
        self.name = loss_name

    def forward(self, x):
        return getattr(self, self.name)(x)
