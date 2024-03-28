from globalsandimports import *

from pulpsolvers import create_mps_file, create_dual_mps_file

def generate_highs_linear_solver() -> Callable[[sparse.coo_matrix, np.ndarray[np.longdouble], Optional[np.ndarray[np.longdouble]]], np.ndarray[Real]]:
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
    def solver(A: sparse.coo_matrix, b: np.ndarray[np.longdouble], c: np.ndarray[np.longdouble] | None = None) -> np.ndarray[Real]:
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

def generate_highs_dual_solver() -> Callable[[sparse.coo_matrix, np.ndarray[np.longdouble], np.ndarray[np.longdouble]], tuple[np.ndarray[Real], np.ndarray[Real]]]:
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
    def solver(A: sparse.coo_matrix, b: np.ndarray[np.longdouble], c: np.ndarray[np.longdouble]) -> np.ndarray[Real]:
        problem = highspy.Highs()
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
        return None
    
    return solver


