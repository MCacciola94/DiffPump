import numpy as np
import torch
from scipy import sparse as sp

from .base import AbstractLoss


class BaseFeasibility(AbstractLoss):
    def __init__(self, loss_name="relu_sum", *, A=None, b=None) -> None:
        super().__init__(loss_name)
        # Normalize constraint matrices
        A, b = self.normalize(A, b)

        # Check that constraints do not contain nan or inf
        if (
            A.isnan().any()
            or A.isinf().any()
            or b.isnan().any()
            or b.isinf().any()
        ):
            msg = "Nan or inf in normalized constraints."
            raise ValueError(msg)

        self.A = A
        self.b = b

    def relu_sum(self, vec):
        """
        Component-wise relu then sum.

        Args:
            vec (torch.DoubleTensor): vector of decision variables

        Returns:
            torch.Double: feasibility loss

        """
        relu = torch.nn.ReLU()
        slack = torch.matmul(self.A, vec) - self.b
        violation = relu(slack - 1e-8)
        return violation.sum() / len(self.b)


class FeasibilityLoss(BaseFeasibility):
    @staticmethod
    def normalize(A, b):
        """Normalize constraint matrices."""
        const_norm = np.sqrt(A.norm(dim=1) ** 2 + b**2)
        const_norm[const_norm < 1e-16] = 1
        A = (A.T / const_norm).T
        b = b / const_norm
        A = torch.DoubleTensor(A)
        b = torch.DoubleTensor(b)
        return A, b


class FeasibilitySparseLoss(BaseFeasibility):
    @staticmethod
    def normalize(A, b):
        """Normalize constraints matrices using sparse operations."""
        norm_const = sp.linalg.norm(sp.hstack([A, b.reshape(-1, 1)]), axis=1)
        norm_const[norm_const < 1e-16] = 1
        A = (A.T / norm_const).T
        b = b / norm_const
        A = torch.sparse_coo_tensor(
            indices=np.array(A.nonzero()), values=A.data, size=A.shape
        ).to(torch.float64)
        b = torch.DoubleTensor(b)
        return A, b
