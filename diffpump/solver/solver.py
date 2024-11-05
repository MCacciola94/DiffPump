from copy import deepcopy


class Solver:
    """
    Abstract class for solving optimization models.

    Adapted from PyEPO.
    """

    def __init__(self):
        if not hasattr(self, "modelSense"):
            self.modelSense = 1
        self.model, self.x = self._get_model()

    def __repr__(self):
        return "Solver " + self.__class__.__name__

    @property
    def num_cost(self):
        """
        Number of cost to be predicted.
        """
        return len(self.x)

    def _get_model(self):
        """
        An abstract method to build a model from a optimization solver

        Returns:
            tuple: optimization model and variables

        """
        raise NotImplementedError

    def setObj(self, c):
        """
        An abstract method to set objective function

        Args:
            c (ndarray): cost of objective function

        """
        raise NotImplementedError

    def solve(self):
        """
        An abstract method to solve model

        Returns:
            tuple: optimal solution (list) and objective value (float)

        """
        raise NotImplementedError

    def copy(self):
        """
        An abstract method to copy model

        Returns:
            Solver: new copied model

        """
        return deepcopy(self)

    def add_constraint(self, coefs, rhs):
        """
        An abstract method to add new constraint

        Args:
            coefs (ndarray): coeffcients of new constraint
            rhs (float): right-hand side of new constraint

        Returns:
            Solver: new model with the added constraint

        """
        raise NotImplementedError

    def relax(self):
        """
        An abstrac method to relax all binary / integer variables.
        """
        raise NotImplementedError
