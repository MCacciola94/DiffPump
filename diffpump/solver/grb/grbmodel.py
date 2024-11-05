from copy import copy

import gurobipy as gp
from gurobipy import GRB

from ..solver import Solver


class optGrbModel(Solver):
    """
    Abstract class for Gurobi-based optimization model.

    Adapted from PyEPO.

    Attributes:
        _model (GurobiPy model): Gurobi model

    """

    env = gp.Env()

    def __init__(self):
        super().__init__()
        # model sense
        self.model.update()
        if self.model.modelSense == GRB.MINIMIZE:
            self.modelSense = 1
        if self.model.modelSense == GRB.MAXIMIZE:
            self.modelSense = -1
        # Turn off gurobi verbose
        self.model.Params.outputFlag = 0

    def __repr__(self):
        return "optGRBModel " + self.__class__.__name__

    def setObj(self, c):
        """
        A method to set objective function

        Args:
            c (np.ndarray / list): cost of objective function

        """
        if len(c) != self.num_cost:
            msg = "Size of cost vector cannot match vars."
            raise ValueError(msg)
        obj = gp.quicksum(c[i] * self.x[k] for i, k in enumerate(self.x))
        self.model.setObjective(obj)

    def solve(self, time_limit=1e100):
        """
        A method to solve model

        Returns:
            tuple: optimal solution (list) and objective value (float)

        """
        self.model.setParam("TimeLimit", time_limit)
        self.model.update()
        self.model.optimize()
        self.model.setParam("TimeLimit", 1e100)
        return [self.x[k].x for k in self.x], self.model.objVal

    def copy(self):
        """
        A method to copy model

        Returns:
            Solver: new copied model

        """
        new_modelmodel = copy(self)
        # update model
        self.model.update()
        # new model
        new_modelmodel.model = self.model.copy()
        # variables for new model
        x = new_modelmodel.model.getVars()
        new_modelmodel.x = {key: x[i] for i, key in enumerate(self.x)}
        return new_modelmodel

    def add_constraint(self, coefs, rhs):
        """
        A method to add new constraint

        Args:
            coefs (np.ndarray / list): coeffcients of new constraint
            rhs (float): right-hand side of new constraint

        Returns:
            Solver: new model with the added constraint

        """
        if len(coefs) != self.num_cost:
            msg = "Size of coef vector cannot cost."
            raise ValueError(msg)
        # copy
        new_modelmodel = self.copy()
        # add constraint
        expr = (
            gp.quicksum(
                coefs[i] * new_modelmodel.x[k] for i, k in enumerate(new_modelmodel.x)
            )
            <= rhs
        )
        new_modelmodel.model.addConstr(expr)
        return new_modelmodel
