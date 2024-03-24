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
    def solver(A: sparse.coo_matrix, b: np.ndarray[np.longdouble], c: np.ndarray[np.longdouble] | None = None) -> np.ndarray[Real]:
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
        
        logging.error("PULP was unable to find a solution. Problem Status: "+pl.LpStatus[status])
        return None
    
    return solver

def generate_pulp_dual_solver(pl_solver = pl.PULP_CBC_CMD(presolve = False)) -> Callable[[sparse.coo_matrix, np.ndarray[np.longdouble], np.ndarray[np.longdouble]], tuple[np.ndarray[Real], np.ndarray[Real]]]:
    """
    Returns a solver for the standard linear programming problem using a PuLP solver
    A@x>=b, x>=0, minimize cx.
    A.T@y>=c, y>=0, minimize b.Ty.
    Defaults to CBC.

    Parameters
    ----------
    pl_solver:
        What PuLP solver to use.
    
    Returns
    -------
    A solver for the standard inequality linear programming problem using a PuLP solver returning both the primal and dual solutions.
    A@x>=b, x>=0, minimize cx.
    A.T@y>=c, y>=0, minimize b.Ty.
    """
    def solver(A: sparse.coo_matrix, b: np.ndarray[np.longdouble], c: np.ndarray[np.longdouble]) -> np.ndarray[Real]:
        problem = pl.LpProblem()
        variables = pl.LpVariable.dicts("x", range(A.shape[1]), 0)

        if not c is None:
            problem += sum([c[i] * variables[i] for i in range(A.shape[1])])

        constraint_mask = np.full(b.shape[0], True, dtype=bool)
        for j in range(b.shape[0]):
            summation = sum([A.data[k] * variables[A.col[k]] for k in range(A.nnz) if A.row[k]==j])
            if isinstance(summation, Real):
                constraint_mask[j] = False
                assert np.isclose(summation, b[j]), "Invalid row "+str(j)
            else:
                constraint_mask[j] = True
                problem += sum([A.data[k] * variables[A.col[k]] for k in range(A.nnz) if A.row[k]==j]) >= b[j]

        status = problem.solve(pl_solver)
        
        if status==1:
            primal = np.array([pl.value(v) if pl.value(v) else 0 for _, v in variables.items()])
            dual = np.zeros(b.shape[0])
            dual[np.where(constraint_mask)] = np.array([c.pi for name, c in problem.constraints.items()])
            return primal, dual
        
        logging.error("PULP was unable to find a solution. Problem Status: "+pl.LpStatus[status])
        return None
    
    return solver

def pulp_solver_via_mps(pl_solver = pl.PULP_CBC_CMD(presolve = False)):
    def solver(A: sparse.coo_matrix, b: np.ndarray[np.longdouble], c: np.ndarray[np.longdouble] | None = None) -> np.ndarray[Real]:
        filename = "tempproblem.mps"
        create_mps_file(filename, A, b, c)

        variables, problem = pl.LpProblem.fromMPS(filename)
        
        status = problem.solve(pl_solver)
        
        if status==1:
            return np.array([pl.value(variables["x_"+str(i)]) if "x_"+str(i) in variables.keys() and pl.value(variables["x_"+str(i)]) else 0 for i in range(A.shape[1])])
        
        logging.debug(pl.LpStatus[status])
        return None
    
    return solver

def create_mps_file(filename: str, A: sparse.coo_matrix, b: np.ndarray[np.longdouble], c: np.ndarray[np.longdouble] | None = None):
    """
    Creates a mps file for the standard linear programming problem
    A@x=b, x>=0, minimize cx
    """
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

    problem.writeMPS(filename)

def create_dual_mps_file(filename: str, A: sparse.coo_matrix, b: np.ndarray[np.longdouble], c: np.ndarray[np.longdouble] | None = None):
    """
    Creates a mps file for the standard linear programming problem
    A@x=b, x>=0, minimize cx
    """
    problem = pl.LpProblem()
    variables = pl.LpVariable.dicts("x", range(A.shape[1]), 0)

    if not c is None:
        problem += sum([c[i] * variables[i] for i in range(A.shape[1])])

    constraint_mask = np.full(b.shape[0], True, dtype=bool)
    for j in range(b.shape[0]):
        summation = sum([A.data[k] * variables[A.col[k]] for k in range(A.nnz) if A.row[k]==j])
        if isinstance(summation, Real):
            constraint_mask[j] = False
            assert np.isclose(summation, b[j]), "Invalid row "+str(j)
        else:
            constraint_mask[j] = True
            problem += sum([A.data[k] * variables[A.col[k]] for k in range(A.nnz) if A.row[k]==j]) >= b[j]

    problem.writeMPS(filename)

    return constraint_mask

