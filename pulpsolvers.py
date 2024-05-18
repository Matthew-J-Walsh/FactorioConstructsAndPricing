from globalsandimports import *

def generate_pulp_linear_solver(pl_solver = pl.PULP_CBC_CMD(presolve = False)) -> CallableSolver:
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
    raise NotImplementedError("Unfixed")
    def solver(A: sparse.csr_matrix, b: np.ndarray, c: np.ndarray | None = None, g: np.ndarray | None = None) -> np.ndarray | None:
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

def generate_pulp_dual_solver(pl_solver = pl.PULP_CBC_CMD(presolve = False), pl_warm_solver = pl.PULP_CBC_CMD(presolve = False, warmStart = True)) -> CallableDualSolver:
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
    def solver(A: sparse.csr_matrix, b: np.ndarray, c: np.ndarray, g: np.ndarray | None = None, ginv: np.ndarray | None = None) -> Tuple[np.ndarray | None, np.ndarray | None]:
        problem = pl.LpProblem()
        variables = pl.LpVariable.dicts("x", range(A.shape[1]), 0)
        Acoo = A.tocoo()

        if not c is None:
            problem += sum([c[i] * variables[i] for i in range(Acoo.shape[1])])

        constraint_mask = np.full(b.shape[0], True, dtype=bool)

        summations = np.full(b.shape[0], 0, dtype=object)
        for j, i, v in zip(Acoo.row, Acoo.col, Acoo.data):
            summations[j] += v * variables[i]

        for j in range(b.shape[0]):
            #summation = sum([A.data[k] * variables[A.col[k]] for k in range(A.nnz) if A.row[k]==j])
            #assert summations[j]==summation
            if isinstance(summations[j], Real):
                constraint_mask[j] = False
                #try:
                assert np.logical_or(summations[j] >= b[j], np.isclose(summations[j], b[j])), "Invalid row "+str(j)+": "+str(summations[j] - b[j])+", "+str(b[j]) # type: ignore
                #except:
                    #raise ValueError(summation - b[j])
            else:
                constraint_mask[j] = True
                problem += summations[j] >= b[j]

        if g is None:
            status = problem.solve(pl_solver)
        else:
            for i, v in enumerate(variables.values()):
                assert v.setInitialValue(g[i])
            status = problem.solve(pl_warm_solver)
        
        if status==1:
            primal = np.array([pl.value(v) if pl.value(v) else 0 for _, v in variables.items()])
            dual = np.zeros(b.shape[0])
            dual[np.where(constraint_mask)] = np.array([c.pi for name, c in problem.constraints.items()])
            return primal, dual
        
        logging.error("PULP was unable to find a solution. Problem Status: "+pl.LpStatus[status])
        return None, None
    
    return solver

def pulp_solver_via_mps(pl_solver = pl.PULP_CBC_CMD(presolve = False)) -> CallableSolver:
    def solver(A: sparse.csr_matrix, b: np.ndarray, c: np.ndarray | None = None, g: np.ndarray | None = None) -> np.ndarray | None:
        filename = "tempproblem.mps"
        create_mps_file(filename, A, b, c)

        variables, problem = pl.LpProblem.fromMPS(filename)
        
        status = problem.solve(pl_solver)
        
        if status==1:
            return np.array([pl.value(variables["x_"+str(i)]) if "x_"+str(i) in variables.keys() and pl.value(variables["x_"+str(i)]) else 0 for i in range(A.shape[1])])
        
        logging.debug(pl.LpStatus[status])
        return None
    
    return solver

def create_mps_file(filename: str, A: sparse.csr_matrix, b: np.ndarray, c: np.ndarray | None = None):
    """
    Creates a mps file for the standard linear programming problem
    A@x>=b, x>=0, minimize cx
    """
    raise NotImplementedError("Unfixed")
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

def create_dual_mps_file(filename: str, A: sparse.csr_matrix, b: np.ndarray, c: np.ndarray | None = None):
    """
    Creates a mps file for the standard linear programming problem
    A@x>=b, x>=0, minimize cx
    """
    raise NotImplementedError("Unfixed")
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

