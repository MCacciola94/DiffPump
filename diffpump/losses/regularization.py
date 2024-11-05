import torch

from .base import AbstractLoss


class RegularizationLoss(AbstractLoss):
    def __init__(self, loss_name="l2normsq"):
        super().__init__(loss_name)

    def l2normsq(self, vec: torch.DoubleTensor) -> torch.DoubleTensor:
        """
        Half the squared l_2 norm of the input vector.

        Args:
            vec (torch.DoubleTensor): input cost vector (theta)

        Returns:
            torch.Double: regularization loss

        """
        return 0.5 * torch.norm(vec, p=2) ** 2
