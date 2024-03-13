from globalsandimports import *

from utils import *
from feasibilityanalysis import *
from scipysolvers import generate_scipy_linear_solver
from pulpsolvers import generate_pulp_linear_solver
from scipsolvers import generate_scip_linear_solver
from highssolvers import generate_highs_linear_solver
from scipy import optimize


def verified_solver(solver: Callable[[sparse.coo_matrix, np.ndarray[np.longdouble], Optional[np.ndarray[np.longdouble]]], np.ndarray[Real]], 
                    name: str) -> Callable[[sparse.coo_matrix, np.ndarray[np.longdouble], Optional[np.ndarray[np.longdouble]]], np.ndarray[Real]]:
    """
    Returns a instance of the solver that verifies the result if given. Also eats errors unless debugging.

    Parameters
    ----------
    solver:
        Optimization function to use. Should solve problems of the form: 
        A@x=b, x>=0, minimize c*x. Return x. 
        If it cannot solve the problem for whatever reason it should return None.
    name:
        What to refer to solver as when giving warnings and throwing errors.

    Returns
    -------
    Function with output verification and error catching added.
    """
    def verified(A: sparse.coo_matrix, b: np.ndarray[np.longdouble], c: np.ndarray[np.longdouble] | None = None):
        try:
            logging.debug("Trying the "+name+" solver.")
            sol = solver(A, b, c)
            if not sol is None:
                if not linear_transform_is_close(A, sol, b).all():
                    if DEBUG_SOLVERS:
                        raise AssertionError(np.max(np.abs(A @ sol - b)))
                    else:
                        logging.warning(name+" gave a result but result wasn't feasible. As debugging is off this won't throw an error. Returning None.")
                        logging.warning("\tLargest recorded error is: "+str(np.max(np.abs(A @ sol - b))))
                        return None
            else:
                logging.debug("Solver returned None.")
            return sol
        except:
            if DEBUG_SOLVERS:
                raise RuntimeError(name)
            logging.warning(name+" threw an error. Returning None.")
            return None
    return verified

"""
PRIMARY_LP_SOLVERS and BACKUP_LP_SOLVERS are lists of solvers for problems of the form:
A@x=b, x>=0, minimize cx
Ordered list, when a LP problem is attempted to be solved these should be ran in order. This order is mostly due to personal experience in usefulness.
"""
PRIMARY_LP_SOLVERS = list(map(verified_solver,
                              [#generate_highs_linear_solver(),
                               generate_pulp_linear_solver(),
                               generate_scip_linear_solver(),
                               ],
                               [#"highspy",
                                "pulp CBC",
                                "scip",
                                ]))
#BACKUP_LP_SOLVERS = []
BACKUP_LP_SOLVERS = list(map(verified_solver,
                             [generate_scipy_linear_solver("highs-ipm", {}),
                              generate_scipy_linear_solver("highs", {}),
                              generate_scipy_linear_solver("highs-ds", {}),
                              generate_scipy_linear_solver("simplex", {}),],
                             ["highs-ipm",
                              "highs",
                              "highs-ds",
                              "simplex",]))


def solve_factory_optimization_problem(R_j_i: sparse.coo_matrix, u_j: np.ndarray[np.longdouble], c_i: np.ndarray[np.longdouble], reference: list[str] = []) -> np.ndarray[Real]:
    """
    Solve an optimization problem given a linear transformation on construct counts, a target output vector, and a cost vector.
    Attempts to use the various linear programming solvers until one works. 
    Runs PRIMARY_LP_SOLVERS without slack, then PRIMARY_LP_SOLVERS with slack, then BACKUP_LP_SOLVERS without slack, then BACKUP_LP_SOLVERS without slack.
    
    Parameters
    ----------
    R_j_i:
        Sparse matrix representing the linear transformation from a construct array to results.
    u_j:
        Vector of required outputs.
    c_i:
        Vector of costs of constructs.

    Returns
    -------
    Vector of rates that each construct is used in optimal factory.
    """
    A = R_j_i
    b = u_j
    A_slacked = sparse.hstack([R_j_i, -1 * sparse.eye(R_j_i.shape[0], dtype=np.longdouble, format="coo")], format="coo")
    assert A_slacked.shape[0]==R_j_i.shape[0] #TODO: remove check
    c = c_i
    c_slacked = np.concatenate([c_i, np.zeros(R_j_i.shape[0], dtype=np.longdouble)])

    #Ax>=b
    #Ax+k=b

    #A has a right-handed inverse.
    #A_slacked has a right-handed inverse.

    #for solver in PRIMARY_LP_SOLVERS:
    #    result = solver(A, b, c)
    #    if not result is None:
    #        return result
    for solver in PRIMARY_LP_SOLVERS:
        result = solver(A_slacked, b, c_slacked)
        if not result is None:
            return result[:c.shape[0]]
    #for solver in BACKUP_LP_SOLVERS:
    #    result = solver(A, b, c)
    #    if not result is None:
    #        return result
    #for solver in BACKUP_LP_SOLVERS:
    #    result = solver(A_slacked, b, c_slacked)
    #    if not result is None:
    #        return result[:c.shape[0]]
    
    """
    result = optimize.linprog(c.astype(np.longdouble), A_ub=-1 * A.astype(np.longdouble), b_ub=-1 * b.astype(np.longdouble), bounds=(0, None))
    if result.status==0:
        s = result.x
        assert np.logical_or(np.isclose(A.astype(np.longdouble) @ s, b.astype(np.longdouble)), A.astype(np.longdouble) @ s >= b.astype(np.longdouble)).all()
        np.savetxt("A.txt", A.astype(np.longdouble))
        np.savetxt("b.txt", b.astype(np.longdouble))
        np.savetxt("c.txt", c.astype(np.longdouble))
        raise AssertionError("WTF???")
    
    failures = []
    for j in range(len(u_j)):
        if u_j[j]!=0:
            b = np.full_like(u_j, np.longdouble(0))
            b[j] = u_j[j]
            works = False

            result = optimize.linprog(c.astype(np.longdouble), A_ub=-1 * A.astype(np.longdouble), b_ub=-1 * b.astype(np.longdouble))
            if result.status==0:
                works = True
            if not works:
                for solver in PRIMARY_LP_SOLVERS:
                    result = solver(A_slacked, b, c_slacked)
                    if not result is None:
                        works = True
                        break
            if not works:
                for solver in BACKUP_LP_SOLVERS:
                    result = solver(A_slacked, b, c_slacked)
                    if not result is None:
                        works = True
                        break
            if not works:
                logging.error(reference[j])
                failures.append(reference[j])

    logging.error(failures)
    """
    raise ValueError("Unable to form factory even with slack.")


def solve_pricing_model_calculation_problem(R_j_i: sparse.coo_matrix, s_i: np.ndarray[np.longdouble], u_j: np.ndarray[np.longdouble], c_i: np.ndarray[np.longdouble]) -> np.ndarray[Real]:
    """
    Calculates a pricing model given a list of constructs, their usages, the target output, and the inital pricing model.
    Attempts to use the various linear programming solvers until one works. 
    Runs PRIMARY_LP_SOLVERS without slack, then PRIMARY_LP_SOLVERS with slack, then BACKUP_LP_SOLVERS without slack, then BACKUP_LP_SOLVERS without slack.
    
    Parameters
    ----------
    R_j_i:
        Sparse matrix representing the linear transformation from a construct array to results.
    s_i:
        Vector of rates that each construct is used in optimal factory.
    u_j:
        Vector of required outputs.
    c_i:
        Vector of costs of constructs.
    
    Returns
    -------
    p_j:
        Vector representing the pricing model of an optimal setup.
    """
    m = len(s_i) #R_j_i.shape[1]
    n = len(u_j) #R_j_i.shape[0]

    #Rp<=c
    #SRp=c
    #Rp-Il=c
    #Sl=0
    #Pp=0
    #[R | I]       [c]
    #[0 | S] [p] = [0]
    #[P | 0] [l]   [0]
    S_k = sparse.diags(1-np.isclose(s_i.astype(np.longdouble), 0).astype(np.longdouble), format="coo")
    P_j = sparse.diags(1-linear_transform_is_close(R_j_i, s_i, u_j).astype(np.longdouble), format="coo")

    A = sparse.vstack([sparse.hstack([R_j_i.T, sparse.eye(m, dtype=np.longdouble, format="coo")], format="coo"),
                       sparse.hstack([sparse.coo_matrix((m, n), dtype=np.longdouble), S_k], format="coo"),
                       sparse.hstack([P_j, sparse.coo_matrix((n, m), dtype=np.longdouble)], format="coo")], format="coo")
    b = np.concatenate([c_i, np.zeros(m + n, dtype=np.longdouble)])
    A_slacked = sparse.hstack([A, sparse.eye(A.shape[0], dtype=np.longdouble, format="coo")], format="coo")
    slack_penalty = np.concatenate([np.zeros(A.shape[1]), np.ones(A.shape[0])])

    #A has no linear independence.
    #A_slacked has no linear independence.

    for solver in PRIMARY_LP_SOLVERS:
        result = solver(A, b)
        if not result is None:
            return result[:n]#, result[n:]
    #for solver in PRIMARY_LP_SOLVERS:
        result = solver(A_slacked, b, slack_penalty)
        if not result is None:
            return result[:n]#, result[n:m+n]
    #for solver in BACKUP_LP_SOLVERS:
    #    result = solver(A, b)
    #    if not result is None:
    #        return result[:n]#, result[n:]
    #for solver in BACKUP_LP_SOLVERS:
    #    result = solver(A_slacked, b, slack_penalty)
    #    if not result is None:
    #        return result[:n]#, result[n:m+n]

    raise ValueError("Unable to form pricing model even with slack.")

