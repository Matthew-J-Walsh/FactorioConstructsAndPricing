from globalsandimports import *

def generate_scip_linear_solver() -> Callable[[sparse.coo_matrix, np.ndarray[np.longdouble], Optional[np.ndarray[np.longdouble]]], np.ndarray[float]]:
    """
    Returns a solver for the standard linear programming problem pyscipopt 
    A@x=b, x>=0, minimize cx

    Returns
    -------
    Function that solves: A@x=b, x>=0, minimize cx given A, b, and c.
    """
    def solver(A: sparse.coo_matrix, b: np.ndarray[np.longdouble], c: np.ndarray[np.longdouble] | None = None):
        problem = scip.Model("Standard")
        variables = [problem.addVar("x"+str(i), lb=0) for i in range(A.shape[1])]
        
        if not c is None:
            problem.setObjective(sum([c[i] * variables[i] for i in range(A.shape[1])]))
            
        for j in range(b.shape[0]):
            problem.addCons(sum([A.data[k] * variables[A.col[k]] for k in range(A.nnz) if A.row[k]==j]) == b[j])

        problem.solveConcurrent()
        sol = problem.getBestSol()

        try: #TODO: check sol without try except
            return np.array([sol[variables[i]] for i in range(A.shape[1])])
        except:
            logging.error("SCIP solve unable to find solution. Problem Status: "+str(problem.getStatus()))
            return None
        
    return solver

def generate_scip_dual_solver() -> Callable[[sparse.coo_matrix, np.ndarray[np.longdouble], Optional[np.ndarray[np.longdouble]]], np.ndarray[float]]:
    """
    Returns a solver for the standard dual programming problem pyscipopt 
    A@x>=b, x>=0, minimize cx.
    A.T@y>=c, y>=0, minimize b.Ty.
    Defaults to CBC.
    
    Returns
    -------
    A solver for the standard inequality linear programming problem using pyscipopt returning both the primal and dual solutions.
    A@x>=b, x>=0, minimize cx.
    A.T@y>=c, y>=0, minimize b.Ty.
    """
    def solver(A: sparse.coo_matrix, b: np.ndarray[np.longdouble], c: np.ndarray[np.longdouble] | None = None):
        problem = scip.Model("Standard")
        variables = [problem.addVar("x"+str(i), lb=0) for i in range(A.shape[1])]
        
        if not c is None:
            problem.setObjective(sum([c[i] * variables[i] for i in range(A.shape[1])]))
            
        for j in range(b.shape[0]):
            problem.addCons(sum([A.data[k] * variables[A.col[k]] for k in range(A.nnz) if A.row[k]==j]) == b[j])

        problem.solveConcurrent()
        sol = problem.getBestSol()

        try: #TODO: check sol without try except
            primal = np.array([sol[variables[i]] for i in range(A.shape[1])])
            dual = np.array([problem.getDualsolLinear() for j in range(A.shape[0])])
        except:
            logging.error("SCIP solve unable to find solution. Problem Status: "+str(problem.getStatus()))
            return None
        
    return solver