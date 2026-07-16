#!/usr/bin/env python
"""
Shortest path problem
"""

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from scipy import sparse as sp

from ...mps_loader import read_mps
from .grbmodel import optGrbModel


class MIPSparseModel(optGrbModel):
    def __init__(self, path):
        self.path = path
        # eq.(13) auxiliary structures (general-integer distance linearisation).
        # Initialised before super().__init__() because _get_model() fills them.
        self._d_vars = {}   # var index -> auxiliary d variable
        self._d_pos = {}    # var index -> constraint  x_i - d_i <= y_i
        self._d_neg = {}    # var index -> constraint -x_i - d_i <= -y_i
        self.general_int_idxs = []
        super().__init__()  # calls _get_model(), which sets A/b/sense + d-vars

    def _get_model(self):
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables

        """
        # The custom read_mps parser is only used by the dense model; the sparse
        # path takes constraints from Gurobi's model.getA(). Some MPS bound
        # formats trip read_mps (IndexError), so keep it non-fatal here.
        try:
            self.MPS = read_mps(self.path)
        except Exception:
            self.MPS = None

        # ceate a model
        m = gp.Model("GurobiMIPModel", env=self.env)
        m = gp.read(str(self.path))
        self.original_model = m
        # varibles (original, before relaxation)
        orig_vars = m.getVars()
        binary_vars_names = [v.VarName for v in orig_vars if v.vtype != "C"]
        # General-integer variables = non-continuous whose domain is NOT {0,1}.
        # For these the plain theta^T x pumping objective is not the distance and
        # can be unbounded; we linearise the eq.(13) distance with d-variables.
        gen_int_names = [
            v.VarName for v in orig_vars
            if v.vtype != "C" and not (v.LB == 0.0 and v.UB == 1.0)
        ]

        m = m.relax()
        # Method=-1 (auto) for the first cold solve; grbmodel.solve() switches to
        # primal simplex for the warm-started subsequent solves. Single-threaded.
        m.Params.Method = -1
        m.setParam(GRB.Param.Threads, 1)
        m.setParam(GRB.Param.OptimalityTol, 1e-9)
        m.setParam(GRB.Param.FeasibilityTol, 1e-9)
        m.setParam(GRB.Param.BarConvTol, 1e-16)

        x = m.getVars()
        x = dict(enumerate(x))
        self.var_to_idx = {var.VarName: i for i, var in x.items()}
        self.binary_vars = [
            self.var_to_idx[name] for name in binary_vars_names
        ]
        self.general_int_idxs = [self.var_to_idx[name] for name in gen_int_names]
        if len(binary_vars_names) != len(x.keys()):
            print("The model has both binary and continuous variables.")

        # Capture the ORIGINAL constraint matrix BEFORE adding the d-rows, so the
        # feasibility loss (get_constr) uses only the real constraints.
        m.update()
        self.A = m.getA()
        self.b = np.array(m.getAttr("RHS"))
        self.sense = np.array(m.getAttr("sense"))

        # eq.(13) linearisation for general integers:
        #   min ... + sum_i |theta_i| d_i   s.t.  d_i >= x_i - y_i,  d_i >= y_i - x_i
        # The RHS (y_i) is refreshed each iteration by set_pump_target().
        for idx in self.general_int_idxs:
            xi = x[idx]
            di = m.addVar(lb=0.0, ub=GRB.INFINITY, name=f"d_{idx}")
            self._d_vars[idx] = di
            self._d_pos[idx] = m.addConstr(xi - di <= 0.0, name=f"dpos_{idx}")
            self._d_neg[idx] = m.addConstr(-xi - di <= 0.0, name=f"dneg_{idx}")
        m.update()

        return m, x

    def set_pump_target(self, y):
        """Refresh the RHS of the eq.(13) linking constraints to the current
        rounded solution y (call before each pumping solve). No-op for pure
        binary/mixed-binary instances (no general integers)."""
        for idx, cpos in self._d_pos.items():
            yi = float(y[idx])
            cpos.RHS = yi
            self._d_neg[idx].RHS = -yi

    def setObj(self, c):
        """Build the pumping objective (eq. 13):
            sum_{binary/cont. i} theta_i x_i  +  sum_{gen-integer i} |theta_i| d_i.
        Binary variables keep the linear theta^T x form; general integers use the
        bounded weighted L1 distance via the auxiliary d-variables."""
        if len(c) != self.num_cost:
            msg = "Size of cost vector cannot match vars."
            raise ValueError(msg)
        gen = self._d_vars
        terms = [float(c[k]) * xi for k, xi in self.x.items()
                 if k not in gen and float(c[k]) != 0.0]
        terms += [abs(float(c[idx])) * di for idx, di in gen.items()
                  if float(c[idx]) != 0.0]
        self.model.setObjective(gp.quicksum(terms))

    def get_binary_vars(self):
        return self.binary_vars

    def check_feasibility(self, x):
        x = x.detach().numpy()
        A, b = self.get_constr()
        norm_const = sp.linalg.norm(sp.hstack([A, b.reshape(-1, 1)]), axis=1)
        A = (A.T / norm_const).T
        b = b / norm_const
        slack = A.dot(x) - b
        return np.all(slack < 1e-8)

    def get_ineq_constr(self):
        A = self.A.copy()
        b = self.b.copy()
        sense = self.sense

        A1 = A[(sense == "<")]
        b1 = b[(sense == "<")]
        A2 = A[(sense == ">")]
        b2 = b[(sense == ">")]

        # Drop vacuous constraints (a.x <= +inf / a.x >= -inf, e.g. degenerate
        # ranged rows): they are always satisfied, but their infinite row norm
        # turns the normalised feasibility system into nan.
        A1, b1 = A1[b1 < np.inf], b1[b1 < np.inf]
        A2, b2 = A2[b2 > -np.inf], b2[b2 > -np.inf]

        A = sp.vstack([A1, -A2])
        b = np.concatenate((b1, -b2))

        return A, b

    def get_eq_constr(self):
        A = self.A.copy()
        b = self.b.copy()
        sense = self.sense

        A = A[sense == "=", :]
        b = b[sense == "="]

        return A, b

    def get_constr(self):
        A = self.A.copy()
        b = self.b.copy()
        sense = self.sense

        A1, b1 = self.get_ineq_constr()

        A2 = A[sense == "=", :]
        b2 = b[sense == "="]

        A = sp.vstack([A1, A2, -A2])
        b = np.concatenate((b1, b2, -b2))

        return A, b

    def get_active_constr(self, x):
        x = x.detach().numpy()
        A, b = self.get_constr()
        norm_const = sp.linalg.norm(sp.hstack([A, b.reshape(-1, 1)]), axis=1)
        A = (A.T / norm_const).T
        b = b / norm_const
        slack = np.abs(A.dot(x) - b)
        A = A.tocsr()
        active_constr = (slack < 1e-2).nonzero()[0]
        return A[active_constr, :], b[active_constr]
