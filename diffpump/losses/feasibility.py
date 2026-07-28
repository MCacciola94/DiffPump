import numpy as np
import torch
from scipy import optimize as sco
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
        # Use COO row/col/data together: A.nonzero() drops explicitly-stored
        # zeros while A.data keeps them, which would give mismatched index/value
        # counts (torch: "indices and values must have same nnz").
        A = A.tocoo()
        A = torch.sparse_coo_tensor(
            indices=np.vstack([A.row, A.col]), values=A.data, size=A.shape
        ).to(torch.float64)
        b = torch.DoubleTensor(b)
        return A, b


class _ArgminFeas(torch.autograd.Function):
    """
    eq.(21): argmin feasibility loss  g(x_int) = min_u G(x_int, u), where
        G(x,u) = (1/m) Σ_j ReLU( (A^int_j x + A^cont_j u - b_j) - eps )^q
    with A, b already row-normalized (Ax <= b convention, as in relu_sum). For a
    fixed rounded integer point x, the CONTINUOUS variables u are re-optimised to
    the best feasible completion (L-BFGS-B); g is that minimised violation.

    Backward uses the envelope theorem: dg/dx = dG/dx at u=u*(x); the du/dx term
    drops because ∇_u G = 0 at the optimum.

    A_int (m x n_int), A_cont (m x n_cont): scipy sparse, constant wrt the gradient,
    passed as non-tensor args.
    """

    @staticmethod
    def forward(ctx, x_int, A_int, A_cont, b, q, eps):
        x_np = x_int.detach().numpy()
        m = A_int.shape[0]
        n_cont = A_cont.shape[1]
        Aix = A_int @ x_np                              # (m,)

        def _g(u):
            lhs = Aix + (A_cont @ u if n_cont else 0.0)
            return np.maximum(lhs - b - eps, 0.0)       # violation, >=0

        def G(u):
            return float(np.sum(_g(u) ** q) / m)

        def dGdu(u):
            g = _g(u)
            coeff = q * g ** max(q - 1, 0) / m          # d/d(lhs)
            return A_cont.T @ coeff                     # (n_cont,)

        if n_cont > 0:
            res = sco.minimize(G, np.zeros(n_cont), jac=dGdu, method="L-BFGS-B",
                               options={"maxiter": 200, "ftol": 1e-12})
            u_opt = res.x
            g_val = float(res.fun)
        else:                                           # pure-integer: nothing to optimise
            u_opt = np.zeros(0)
            g_val = G(u_opt)

        ctx.save_for_backward(x_int)
        ctx.A_int, ctx.A_cont, ctx.b = A_int, A_cont, b
        ctx.u, ctx.q, ctx.eps, ctx.m, ctx.n_cont = u_opt, q, eps, m, n_cont
        return torch.tensor(g_val, dtype=torch.double)

    @staticmethod
    def backward(ctx, grad_output):
        (x_int,) = ctx.saved_tensors
        lhs = ctx.A_int @ x_int.detach().numpy()
        if ctx.n_cont:
            lhs = lhs + ctx.A_cont @ ctx.u
        g = np.maximum(lhs - ctx.b - ctx.eps, 0.0)
        coeff = ctx.q * g ** max(ctx.q - 1, 0) / ctx.m
        dGdx = ctx.A_int.T @ coeff                      # dG/dx at u* (n_int,)
        grad_x = torch.tensor(dGdx, dtype=torch.double) * grad_output
        return grad_x, None, None, None, None, None


class FeasibilityArgminLoss:
    """eq.(21) feasibility loss used by DP5. Splits the (row-normalised) Ax<=b
    system into integer/continuous columns once, then evaluates the argmin loss
    on the integer slice of the rounded solution. int_idx/cont_idx index the
    ORIGINAL variables (int_idx = the non-continuous ones, as get_binary_vars)."""

    def __init__(self, *, A, b, int_idx, cont_idx, q=2, eps=0.0):
        b = np.asarray(b, dtype=float).ravel()
        norm_const = sp.linalg.norm(sp.hstack([A, b.reshape(-1, 1)]), axis=1)
        norm_const[norm_const < 1e-16] = 1
        A = (A.T / norm_const).T
        b = b / norm_const
        if not np.all(np.isfinite(b)) or not np.all(np.isfinite(A.data)):
            msg = "Nan or inf in normalized constraints."
            raise ValueError(msg)
        A_csc = A.tocsc()
        self.A_int = A_csc[:, int_idx].tocsr()
        self.A_cont = (A_csc[:, cont_idx].tocsr() if cont_idx
                       else sp.csr_matrix((A.shape[0], 0)))
        self.b = b
        self.q = int(q)
        self.eps = float(eps)

    def __call__(self, x_int):
        return _ArgminFeas.apply(x_int, self.A_int, self.A_cont, self.b,
                                 self.q, self.eps)
