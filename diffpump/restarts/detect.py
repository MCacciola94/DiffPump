import numpy as np


def is_x_lp_cycling(history_xstar, win_size=3):
    """
    Check whether last x_lp is observed in recent history.

    Args:
        history_xstar (list[np.array]): values of x_lp over iterations
        win_size (int, optional): length of window to detect cycles.
                                  Defaults to 3.

    Returns:
        bool: True if cycle detected, False otherwise

    """
    # Check if there are enough iterations
    if len(history_xstar) >= win_size:
        last_x = history_xstar[-1]
        # Check differences with previous solutions
        for old_x in history_xstar[-win_size:-1]:
            if np.all(np.isclose(old_x, last_x)):
                return True
    return False


def is_x_round_same(x_round_last, x_round_second_last):
    """
    Check whether last x_round solution is almost equal to previous one.

    Args:
        x_round_last (np.array): value of x_round in current iteration
        x_round_second_last (np.array): value of x_round in previous iteration

    Returns:
        bool: True if values are all approximately equal, False otherwise

    """
    return np.all(np.isclose(x_round_last, x_round_second_last))
