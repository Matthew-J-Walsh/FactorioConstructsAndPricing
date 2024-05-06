from globalsandimports import *

from pulpsolvers import create_mps_file, create_dual_mps_file

def generate_highs_linear_solver() -> CallableSolver:
    """
    Returns a solver for the standard linear programming problem using the HiGHs solver
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
    def solver(A: sparse.coo_matrix, b: np.ndarray, c: np.ndarray | None = None, g: np.ndarray | None = None) -> np.ndarray | None:
        problem = highspy.Highs()
        filename = "tempproblem.mps"

        create_mps_file(filename, A, b, c)

        problem.readModel(filename)

        problem.run()
    
        if problem.modelStatusToString(problem.getModelStatus())=="Optimal":
            solution = problem.getSolution()
            x = np.zeros(A.shape[1])
            for i in range(len(solution.col_value)):
                if problem.getColName(i)[1][2:].isdigit():
                    x[int(problem.getColName(i)[1][2:])] = solution.col_value[i]
            return x
        
        logging.error("Highs couldn't find a solution. Problem status: "+problem.modelStatusToString(problem.getModelStatus()))
        return None

    return solver

def generate_highs_dual_solver() -> CallableDualSolver:
    """
    Returns a solver for the standard linear programming problem using the HiGHs solver
    A@x>=b, x>=0, minimize cx.
    A.T@y>=c, y>=0, minimize b.Ty.
    Defaults to CBC.

    Returns
    -------
    A solver for the standard inequality linear programming problem using the HiGHs solver returning both the primal and dual solutions.
    A@x>=b, x>=0, minimize cx.
    A.T@y>=c, y>=0, minimize b.Ty.
    """
    def solver(A: sparse.coo_matrix, b: np.ndarray, c: np.ndarray, g: np.ndarray | None = None, ginv: np.ndarray | None = None) -> Tuple[np.ndarray | None, np.ndarray | None]:
        problem = highspy.Highs()
        raise NotImplementedError("please implement g")
        filename = "tempproblem.mps"

        constraint_mask = create_dual_mps_file(filename, A, b, c)

        problem.readModel(filename)

        problem.run()
    
        if problem.modelStatusToString(problem.getModelStatus())=="Optimal":
            solution = problem.getSolution()
            x = np.zeros(A.shape[1])
            for i in range(len(solution.col_value)):
                if problem.getColName(i)[1][2:].isdigit():
                    x[int(problem.getColName(i)[1][2:])] = solution.col_value[i] #-1?
                else:
                    logging.error(problem.getColName(i))
            ytemp = np.zeros(len(solution.row_dual))
            for i in range(len(solution.row_dual)):
                if problem.getRowName(i)[1][2:].isdigit():
                    ytemp[int(problem.getRowName(i)[1][2:])-1] = solution.row_dual[i]
                else:
                    logging.error(problem.getRowName(i))
            y = np.zeros(A.shape[0])
            y[np.where(constraint_mask)[0]] = ytemp
            return x, y
        
        logging.error("Highs couldn't find a solution. Problem status: "+problem.modelStatusToString(problem.getModelStatus()))
        return None, None
    
    return solver

def generate_highs_dual_solver_pythonic() -> CallableDualSolver:
    """
    Returns a solver for the standard linear programming problem using the HiGHs solver
    A@x>=b, x>=0, minimize cx.
    A.T@y>=c, y>=0, minimize b.Ty.
    Defaults to CBC.

    Returns
    -------
    A solver for the standard inequality linear programming problem using the HiGHs solver returning both the primal and dual solutions.
    A@x>=b, x>=0, minimize cx.
    A.T@y>=c, y>=0, minimize b.Ty.
    """
    def solver(A: sparse.coo_matrix, b: np.ndarray, c: np.ndarray, g: np.ndarray | None = None, ginv: np.ndarray | None = None) -> Tuple[np.ndarray | None, np.ndarray | None]:
        problem = highspy.Highs()
        lp = highspy.HighsLp()

        inf = highspy.kHighsInf

        lp.num_col_ = c.shape[0]
        lp.num_row_ = b.shape[0]

        lp.col_cost_ = c#.copy()
        lp.col_lower_ = np.array([0]*c.shape[0], dtype=np.double)
        lp.col_upper_ = np.array([inf]*c.shape[0], dtype=np.double)

        lp.row_lower_ = b#.copy()
        lp.row_upper_ = np.array([inf]*b.shape[0], dtype=np.double)

        #a_matrix_.start_ needs to to have a size 1 larger than num_col_, it acts as aranges(i,i+1) for grouping the matrix by columns
        Acsc = A.tocsc()#copy?

        lp.a_matrix_.start_ = Acsc.indptr
        lp.a_matrix_.index_ = Acsc.indices
        lp.a_matrix_.value_ = Acsc.data

        logging.debug("Dumping")
        logging.debug(len(lp.col_cost_))
        logging.debug(lp.col_cost_)
        logging.debug(len(lp.col_lower_))
        logging.debug(lp.col_lower_)
        logging.debug(len(lp.col_upper_))
        logging.debug(lp.col_upper_)
        logging.debug(len(lp.row_lower_))
        logging.debug(lp.row_lower_)
        logging.debug(len(lp.row_upper_))
        logging.debug(lp.row_upper_)
        logging.debug(len(lp.a_matrix_.start_))
        logging.debug(lp.a_matrix_.start_)
        logging.debug(len(lp.a_matrix_.index_))
        logging.debug(lp.a_matrix_.index_)
        logging.debug(len(lp.a_matrix_.value_))
        logging.debug(lp.a_matrix_.value_)

        problem.passModel(lp)
        logging.debug("Passed")

        #if not g is None:
        #    initial_sol = highspy.HighsSolution()
        #    initial_sol.value_valid = True
        #    initial_sol.col_value = g
        #    problem.passSolution(initial_sol)
        #    logging.debug("Initial Solve Passed")

        problem.run()
        logging.debug("Ran")
    
        if problem.modelStatusToString(problem.getModelStatus())=="Optimal":
            solution = problem.getSolution()
            x = np.array(solution.col_value)
            assert x.shape[0]==A.shape[1]
            y = np.array(solution.row_dual)
            assert y.shape[0]==A.shape[0]
            return x, y
        
        logging.error("Highs couldn't find a solution. Problem status: "+problem.modelStatusToString(problem.getModelStatus()))
        return None, None
    
    return solver


