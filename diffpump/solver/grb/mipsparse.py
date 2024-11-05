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
        super().__init__()
        self.A = self.model.getA()
        self.b = self.model.getAttr("RHS")
        self.b = np.array(self.b)
        self.sense = self.model.getAttr("sense")
        self.sense = np.array(self.sense)

    def _get_model(self):
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables

        """
        self.MPS = read_mps(self.path)

        # ceate a model
        m = gp.Model("GurobiMIPModel", env=self.env)
        m = gp.read(str(self.path))
        self.original_model = m
        # varibles
        x = m.getVars()
        x = dict(enumerate(x))
        binary_vars_names = [
            var.VarName for i, var in x.items() if var.vtype != "C"
        ]

        m = m.relax()
        # Setting LP solver to deterministic method
        m.Params.Method = 4
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
        if len(binary_vars_names) != len(x.keys()):
            print("The model has both binary and continuous variables.")

        return m, x

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
