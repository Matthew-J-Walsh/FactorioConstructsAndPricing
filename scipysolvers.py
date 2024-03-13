from globalsandimports import *

from scipy import optimize

def generate_scipy_linear_solver(method: str = "revised simplex", options: dict = {'pivot': 'bland', 'maxiter': 50000, 'presolve': True}) -> Callable[[sparse.coo_matrix, np.ndarray[np.longdouble], Optional[np.ndarray[np.longdouble]]], np.ndarray[Real]]:
    """
    Returns a solver for the standard linear programming problem using scipy.optimize.linprog
    A@x=b, x>=0, minimize cx
    Defaults to revised simplex with Bland pivoting.

    Parameters
    ----------
    method:
        What scipy linear optimization method to use.
    options:
        Additional option settings for the specific method chosen.
    
    Returns
    -------
    Function that solves: A@x=b, x>=0, minimize cx given A, b, and c.
    """
    def solver(A: sparse.coo_matrix, b: np.ndarray[np.longdouble], c: np.ndarray[np.longdouble] | None = None):
        if c is None:
            c = np.zeros(A.shape[1], dtype=np.longdouble)
        
        A = A.todense()
        result = optimize.linprog(c, A_eq=A, b_eq=b, bounds=(0, None), method=method, options=options)

        logging.debug(result.message+" Status: "+str(result.status))
        if result.status in [0,1,4]: #4 usually indicates possible issues with simplex reaching optimal, we leave it in there because most of the time its pretty close, if its some other error it will be caught on verification.
            return result.x
        
        return None
    return solver
