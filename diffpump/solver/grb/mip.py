import gurobipy as gp
import numpy as np
import torch
from gurobipy import GRB

from ...mps_loader import read_mps
from .grbmodel import optGrbModel


class MIPModel(optGrbModel):
    def __init__(self, path):
        self.path = path
        super().__init__()

    def _get_model(self):
        """
        Build Gurobi model.

        Returns:
            tuple: optimization model and variables

        """
        self.MPS = read_mps(self.path)

        # create a model
        m = gp.Model("GurobiMIPModel", env=self.env)
        m = gp.read(str(self.path))
        self.original_model = m
        # variables
        x = m.getVars()
        x = dict(enumerate(x))
        binary_vars_names = [
            var.VarName for _, var in x.items() if var.vtype != "C"
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
        return all(self.check_constr(name, x) for name in self.MPS.constraints)

    def check_constr(self, name, x):
        constr_val = 0
        constr = self.MPS.constraints[name]
        for var, coeff in constr["coefficients"].items():
            constr_val += x[self.var_to_idx[var]] * coeff

        if constr["type"] == "L":
            return (constr_val - self.MPS.rhs[self.rhs_name][name]) < +1e-4
        if constr["type"] == "G":
            return (constr_val - self.MPS.rhs[self.rhs_name][name]) > -1e-4
        if constr["type"] == "E":
            return abs(constr_val - self.MPS.rhs[self.rhs_name][name]) < 1e-4
        return None

    def get_ineq_constr(self):
        rhs_names = list(self.MPS.rhs.keys())
        if len(rhs_names) != 1:
            print("Multiple RHSs, Aborting")
            return 0

        self.rhs_name = rhs_names[0]

        A = []
        b = []
        n = self.num_cost
        for name in self.MPS.constraints:
            multplr = 1
            aux = np.zeros(n)
            constr = self.MPS.constraints[name]
            if constr["type"] == "E":
                continue
            if constr["type"] == "G":
                multplr = -1
            for var, coeff in constr["coefficients"].items():
                idx = self.var_to_idx[var]
                aux[idx] = coeff * multplr
            A.append(aux)
            b.append(multplr * self.MPS.rhs[self.rhs_name][name])
        return np.array(A), np.array(b)

    def get_eq_constr(self):
        A = []
        b = []
        n = self.num_cost
        for name in self.MPS.constraints:
            aux = np.zeros(n)
            constr = self.MPS.constraints[name]
            if constr["type"] != "E":
                continue
            for var, coeff in constr["coefficients"].items():
                idx = self.var_to_idx[var]
                aux[idx] = coeff
            A.append(aux)
            b.append(self.MPS.rhs[self.rhs_name][name])
        return np.array(A), np.array(b)

    def get_constr(self):
        A = []
        b = []
        n = self.num_cost
        for name in self.MPS.constraints:
            mltplr = 1
            aux = np.zeros(n)
            constr = self.MPS.constraints[name]
            if constr["type"] == "E":
                for var, coeff in constr["coefficients"].items():
                    idx = self.var_to_idx[var]
                    aux[idx] = -coeff
                A.append(aux)
                b.append(-self.MPS.rhs[self.rhs_name][name])
                aux = np.zeros(n)

            if constr["type"] == "G":
                mltplr = -1

            for var, coeff in constr["coefficients"].items():
                idx = self.var_to_idx[var]
                aux[idx] = coeff * mltplr
            A.append(aux)
            b.append(self.MPS.rhs[self.rhs_name][name] * mltplr)

        A, b = np.array(A), np.array(b)
        return A, b

    def get_active_constr(self, x):
        with torch.no_grad():
            A, bs = self.get_constr()
            norm_const = (torch.DoubleTensor(A).norm(dim=1) ** 2 + bs**2) ** (
                0.5
            )
            A = (A.T / norm_const).T
            A = A.detach().numpy()
            bs = bs / norm_const
            slack = torch.abs(torch.DoubleTensor(A).matmul(x) - bs)
            active_constr = (slack < 1e-2).nonzero().squeeze(1)
        return A[active_constr], bs[active_constr]
