from globalsandimports import *

import highspy

def generate_highs_linear_solver() -> Callable[[sparse.coo_matrix, np.ndarray[np.longdouble], Optional[np.ndarray[np.longdouble]]], np.ndarray[Real]]:
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
        problem = highspy.Highs()
        variables = [problem.addVar(0, highspy.kHighsInf) for _ in range(A.shape[1])]

        if not c is None:
            for i in range(A.shape[1]):
                problem.changeColCost(i, c[i])

        for j in range(b.shape[0]):
            summation = sum([A.data[k] * variables[A.col[k]] for k in range(A.nnz) if A.row[k]==j])
            if isinstance(summation, Real):
                assert np.isclose(summation, b[j]), "Invalid row "+str(j)
            else:
                problem.addConstr(summation == b[j])

        logging.debug("Beginning the highs solve.")
        problem.run()

        solution = problem.getSolution()
        status = problem.getModelStatus()
        print(solution)
        raise ValueError(status)
    
        return None
    
    return solver
