from globalsandimports import *

import pulp as pl

def generate_pulp_linear_solver(pl_solver = pl.PULP_CBC_CMD()) -> Callable[[np.ndarray[Fraction], np.ndarray[Fraction], Optional[np.ndarray[Fraction]]], np.ndarray[Real]]:
    """
    Returns a solver for the standard linear programming problem using a PuLP solver
    A@x=b, x>=0, minimize cx
    Defaults to CBC.

    Parameters
    ----------
    pl_solver:
        What PuLP solver to use.
    
    Returns
    -------
    Function that solves: A@x=b, x>=0, minimize cx given A, b, and c.
    """
    def solver(A: np.ndarray[Fraction], b: np.ndarray[Fraction], c: np.ndarray[Fraction] | None = None):
        problem = pl.LpProblem()
        variables = pl.LpVariable.dicts("x", range(A.shape[1]), 0)

        if not c is None:
            problem += sum([c[i] * variables[i] for i in range(A.shape[1])])

        for j in range(b.shape[0]):
            problem += sum([A[j, i] * variables[i] for i in range(A.shape[1])]) == b[j]

        status = problem.solve(pl_solver)
        
        if status==1:
            return np.array([pl.value(v) if pl.value(v) else 0 for _, v in variables.items()])
        
        logging.debug(pl.LpStatus[status])
        return None
    
    return solver
