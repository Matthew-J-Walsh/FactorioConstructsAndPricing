from globalsandimports import *

import pulp as pl

def generate_pulp_linear_solver(pl_solver = pl.PULP_CBC_CMD(presolve = False)) -> Callable[[sparse.coo_matrix, np.ndarray[np.longdouble], Optional[np.ndarray[np.longdouble]]], np.ndarray[Real]]:
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
    def solver(A: sparse.coo_matrix, b: np.ndarray[np.longdouble], c: np.ndarray[np.longdouble] | None = None):
        problem = pl.LpProblem()
        variables = pl.LpVariable.dicts("x", range(A.shape[1]), 0)

        if not c is None:
            problem += sum([c[i] * variables[i] for i in range(A.shape[1])])

        for j in range(b.shape[0]):
            summation = sum([A.data[k] * variables[A.col[k]] for k in range(A.nnz) if A.row[k]==j])
            if isinstance(summation, Real):
                assert np.isclose(summation, b[j]), "Invalid row "+str(j)
            else:
                problem += sum([A.data[k] * variables[A.col[k]] for k in range(A.nnz) if A.row[k]==j]) == b[j]

        status = problem.solve(pl_solver)
        
        if status==1:
            return np.array([pl.value(v) if pl.value(v) else 0 for _, v in variables.items()])
        
        logging.debug(pl.LpStatus[status])
        return None
    
    return solver
