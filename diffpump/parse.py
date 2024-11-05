import argparse


def load_parser():
    """
    Setup configuration for experiments.
    """
    parser = argparse.ArgumentParser()
    # Add arguments to parser
    parser.add_argument("instance", type=str,
                        help="Name of instance to solve.")
    # Feasibility pump configuration
    parser.add_argument("--original", action="store_true",
                        help="Use original feasibility pump.")
    parser.add_argument("--no_restarts", action="store_true",
                        help="Disable restart operations.")
    parser.add_argument("--estim", type=str, default="minusId",
                        help="Method used for estimating Jacobian of argmin.")
    # Gradient projection and normalization
    parser.add_argument("--proj_grad", action="store_true",
                        help="Project gradient orthogonal to theta.")
    parser.add_argument("--normtheta", action="store_true",
                        help="Normalize theta before solving LP.")
    # Solver configuration
    parser.add_argument("--denselinalg", action="store_true",
                        help="Disable scipy sparse operations.")
    # Gradient descent configuration
    parser.add_argument("--optm", type=str,
                        default="sgd", choices=["sgd", "adam"],
                        help="Optimizer for gradient descent.")
    parser.add_argument("--momentum", type=float, default=0.0,
                        help="Momentum for optimizer.")
    parser.add_argument("--lr", type=float, default=1.0,
                        help="learning rate")
    # Loss functions
    parser.add_argument("--integ_metric", type=str,
                        default="minx1mx", choices=["minx1mx", "x1mx"],
                        help="Non-integrality loss function.")
    parser.add_argument("--integ_loss", type=float, default=1.0,
                        help="Weight of non-integrality loss.")
    parser.add_argument("--p", type=float, default=1,
                        help="Order of minx1mx integrality loss.")
    parser.add_argument("--initcost_loss", type=float, default=0.0,
                        help="Weight of cost loss.")
    parser.add_argument("--feas_loss", type=float, default=0.0,
                        help="Weight of feasibility loss.")
    parser.add_argument("--reg_loss", type=float, default=1.0,
                        help="Weight of regularization loss.")
    # Experiment configuration
    parser.add_argument("--iter", type=int, default=1000,
                        help="Maximum number of iterations.")
    parser.add_argument("--expnum", type=int, default=1,
                        help="Number of experiments.")
    # For tests
    parser.add_argument("--addnoise", action="store_true",
                        help="Add noise to contraints rhs. Use only in tests.")

    return parser
