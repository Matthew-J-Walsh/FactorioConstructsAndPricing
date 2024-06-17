from globalsandimports import *
from utils import *

def pulp_dense_solver(A: np.ndarray, b: np.ndarray, c: np.ndarray, g: np.ndarray | None = None, ginv: np.ndarray | None = None) -> Tuple[np.ndarray | None, np.ndarray | None]:
    """Solver for standard linear programming minimization problem with geq. 
    A@x>=b, x>=0, minimize cx.
    A.T@y<=c, y>=0, maximize b.Ty.
    Uses the PuLP PULP_CBC_CMD solver.

    Parameters
    ----------
    A : np.ndarray
        constraint transform
    b : np.ndarray
        constraint vector
    c : np.ndarray
        cost vector
    g : np.ndarray | None, optional
        primal guess, by default None
    ginv : np.ndarray | None, optional
        dual guess, by default None

    Returns
    -------
    Tuple[np.ndarray | None, np.ndarray | None]
        Primal (None if no solution),
        Dual (None if no solution)
    """
    problem = pl.LpProblem()
    variables = pl.LpVariable.dicts("x", range(A.shape[1]), lowBound=0)
    var_arr = np.fromiter(variables.values(), dtype=object)

    if not c is None:
        problem += np.dot(var_arr, c)

    constraint_mask = np.full(b.shape[0], True, dtype=bool)

    summations = np.zeros(A.shape[0], dtype=object)
    for j, i in zip(*np.where(A!=0)):
        summations[j] += A[j, i] * var_arr[i]

    for j in range(b.shape[0]):
        if isinstance(summations[j], Real):
            constraint_mask[j] = False
            assert np.logical_or(summations[j] >= b[j], np.isclose(summations[j], b[j])), "Invalid row "+str(j)+": "+str(summations[j] - b[j])+", "+str(b[j])
        else:
            constraint_mask[j] = True
            problem += summations[j] >= b[j]

    if g is None:
        status = problem.solve(pl.PULP_CBC_CMD(presolve = False, gapRel=SOLVER_TOLERANCES['rtol']**2, gapAbs=SOLVER_TOLERANCES['atol']**2))
    else:
        for i, v in enumerate(variables.values()):
            assert v.setInitialValue(g[i])
        status = problem.solve(pl.PULP_CBC_CMD(presolve = False, warmStart = True, gapRel=SOLVER_TOLERANCES['rtol']**2, gapAbs=SOLVER_TOLERANCES['atol']**2))
    
    if status==1:
        primal = np.array([pl.value(v) if pl.value(v) else 0 for _, v in variables.items()])
        dual = np.zeros(b.shape[0])
        dual[np.where(constraint_mask)] = np.array([constraint.pi for name, constraint in problem.constraints.items()])
    
        #TODO: improve errors
        #assert linear_transform_is_gt(A, primal, b).all()
        #assert linear_transform_is_gt(-1 * A.T, dual, -1 * c).all()

        return primal, dual
    
    logging.error("PULP was unable to find a solution. Problem Status: "+pl.LpStatus[status])
    return None, None

def create_dual_mps_file(filename: str, A: sparse.csr_matrix, b: np.ndarray, c: np.ndarray | None = None) -> None:
    """Creates a mps file for standard linear programming minimization problem with geq.
    A@x>=b, x>=0, minimize cx.
    A.T@y<=c, y>=0, maximize b.Ty.

    Parameters
    ----------
    filename : str
        filename for the MPS file to go
    A : np.ndarray
        constraint transform
    b : np.ndarray
        constraint vector
    c : np.ndarray
        cost vector
    g : np.ndarray | None, optional
        primal guess, by default None
    ginv : np.ndarray | None, optional
        dual guess, by default None
    """
    raise NotImplementedError("Unfixed TODO")
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

def scipy_highs_dense_solver(A: np.ndarray, b: np.ndarray, c: np.ndarray, g: np.ndarray | None = None, ginv: np.ndarray | None = None) -> Tuple[np.ndarray | None, np.ndarray | None]:
    """Solver for standard linear programming minimization problem with geq. 
    A@x>=b, x>=0, minimize cx.
    A.T@y<=c, y>=0, maximize b.Ty.
    Uses the scipy 'highs' solver.

    Parameters
    ----------
    A : np.ndarray
        constraint transform
    b : np.ndarray
        constraint vector
    c : np.ndarray
        cost vector
    g : np.ndarray | None, optional
        primal guess, by default None
    ginv : np.ndarray | None, optional
        dual guess, by default None

    Returns
    -------
    Tuple[np.ndarray | None, np.ndarray | None]
        Primal (None if no solution),
        Dual (None if no solution)
    """
    sol = optimize.linprog(c, A_ub = -1 * A, b_ub = -1 * b, bounds=(0, None), x0=g)
    if sol.status!=0:
        logging.error("scipy highs solver failed.")
        return None, None
    dsol = optimize.linprog(c, A_ub = -1 * A, b_ub = -1 * b, bounds=(0, None), x0=g)
    if dsol.status!=0:
        logging.error("scipy highs solver failed.")
        return None, None
    
    primal = sol.x
    dual = dsol.x
    
    assert linear_transform_is_gt(A, primal, b).all()
    assert linear_transform_is_gt(-1 * A.T, dual, -1 * c).all()

    return primal, dual

#Holds the current best LP solver to use
BEST_LP_SOLVER = pulp_dense_solver