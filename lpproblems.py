import numpy as np
import scipy as sp
import scipy.sparse
import scipy.optimize
import pulp as pl

from utils import *
from scipysolvers import generate_scipy_linear_solver
from pulpsolvers import generate_pulp_linear_solver
from scipsolvers import generate_scip_linear_solver
from feasibilityanalysis import *

from numbers import Real
from typing import TypeVar, Callable, Hashable, Iterable, Any, Optional


def verified_solver(solver: Callable[[np.ndarray[Fraction], np.ndarray[Fraction], Optional[np.ndarray[Fraction]]], np.ndarray[Real]], name: str) -> Callable[[np.ndarray[Fraction], np.ndarray[Fraction], Optional[np.ndarray[Fraction]]], np.ndarray[Real]]:
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
    def verified(A: np.ndarray[Fraction], b: np.ndarray[Fraction], c: np.ndarray[Fraction] | None = None):
        try:
            logging.info("Trying the "+name+" solver.")
            sol = solver(A, b, c)
            if not sol is None:
                if not np.isclose(A.astype(np.longdouble) @ sol.astype(np.longdouble), b.astype(np.longdouble)).all():
                    if DEBUG_SOLVERS:
                        raise AssertionError(np.max(np.abs(A.astype(np.longdouble) @ sol.astype(np.longdouble) - b.astype(np.longdouble))))
                    else:
                        logging.warning(name+" gave a result but result wasn't feasible. As debugging is off this won't throw an error. Returning None.")
                        return None
            else:
                logging.info("Solver returned None.")
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
                              [generate_scipy_linear_solver(),
                               generate_pulp_linear_solver(),
                               generate_scip_linear_solver(),
                               generate_scipy_linear_solver("highs-ipm", {}),],
                               ["revised simplex",
                                "pulp CBC",
                                "scip",
                                "highs-ipm",]))
BACKUP_LP_SOLVERS = list(map(verified_solver,
                             [generate_scipy_linear_solver("highs", {}),
                              generate_scipy_linear_solver("highs-ds", {}),
                              generate_scipy_linear_solver("interior-point", {}),
                              generate_scipy_linear_solver("simplex", {}),],
                             ["highs",
                              "highs-ds",
                              "interior-point",
                              "simplex",]))


def factory_optimization_problem(R_j_i: np.ndarray[Fraction], u_j: np.ndarray[Fraction], c_i: np.ndarray[Fraction], 
                                 solver: Callable[[np.ndarray[Fraction], np.ndarray[Fraction], Optional[np.ndarray[Fraction]]], np.ndarray[Real]],
                                 force: bool = True) -> np.ndarray:
    """
    Solve an optimization problem given a linear transformation on construct counts, a target output vector, and a cost vector using a particular solver.
    
    Parameters
    ----------
    R_j_i:
        Sparse matrix representing the linear transformation from a construct array to results.
    u_j:
        Vector of required outputs.
    c_i:
        Vector of costs of constructs.
    solver:
        Optimization function to use. Should solve problems of the form: 
        A@x=b, x>=0, minimize c*x. Return x. 
        If it cannot solve the problem for whatever reason it should return None.
    force:
        Should the problem be reformulated with slack (Ax>=b) if unable to solve equality problem.

    Returns
    -------
    Vector of rates that each construct is used in optimal factory.
    If such a factory cannot be made with this function return None.
    """
    raise DeprecationWarning
    assert inverse_analysis(np.concatenate([R_j_i, np.identity(R_j_i.shape[1])], axis=0), u_j)

    optimization_result = solver(R_j_i, u_j, c_i)
    if force and optimization_result is None:
        logging.info("Unable to solve exact problem, introducing slack.")
        #A@x>=b, A@x-b>=0,  A@x-b+s=0, [A | I][x / s] = b
        optimization_result = solver(np.concatenate([R_j_i, np.identity(R_j_i.shape[1])], axis=0), u_j, np.concatenate([c_i, np.zeros(R_j_i.shape[1])]))

    return optimization_result


def pricing_model_calculation_problem(R_j_i: np.ndarray[Fraction], s_i: np.ndarray, u_j: np.ndarray[Fraction], c_i: np.ndarray, 
                                      solver: Callable[[np.ndarray[Fraction], np.ndarray[Fraction], Optional[np.ndarray[Fraction]]], np.ndarray],
                                      force: bool = True) -> np.ndarray | None:
    """
    Calculates a pricing model given a list of constructs, their usages, the target output, and the inital pricing model using a particular solver.
    
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
    solver:
        Optimization function to use. Should solve problems of the form: 
        A@x=b, x>=0, minimize c*x. Return x. 
        If it cannot solve the problem for whatever reason it should return None.
    force:
        Should a result be forced rather than returning None.
        If it would have originally returned None will introduce slack and penalty
    
    Returns
    -------
    Vector representing the pricing model of an optimal setup.
    If such a pricing model cannot be made with this function return None.
    Should never return None if force is on.
    """
    raise DeprecationWarning
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
    S_k = np.diag(1-np.isclose(np.array(s_i), 0), dtype=Fraction)
    P_j = np.diag(1-np.isclose(R_j_i @ s_i, u_j), dtype=Fraction)

    A = np.concatenate([np.concatenate([R_j_i, np.identity(m)], axis=0),
                        np.concatenate([np.zeros(m, R_j_i.shape[0], dtype=Fraction), S_k], axis=0),
                        np.concatenate([P_j, np.zeros(n, m, dtype=Fraction)], axis=0)])
    b = np.concatenate([c_i, np.zeros(m + n, dtype=Fraction)])

    assert inverse_analysis(A, b)

    result = solver(A, b)

    if force and result is None:
        A_slacked = np.concatenate([A, np.identity(A.shape[1])], axis=0)
        slack_penalty = np.concatenate([np.zeros(A.shape[0]), np.ones(A.shape[1])])
        result = solver(A_slacked, b, slack_penalty)

    return result[:n]


def solve_factory_optimization_problem(R_j_i: np.ndarray, u_j: np.ndarray, c_i: np.ndarray) -> np.ndarray:
    """
    Solve an optimization problem given a linear transformation on construct counts, a target output vector, and a cost vector.
    Attempts to use the various linear programming solvers until one works. 
    Runs PRIMARY_LP_SOLVERS without force, then PRIMARY_LP_SOLVERS with force, then BACKUP_LP_SOLVERS without force, then BACKUP_LP_SOLVERS without force.
    
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
    If such a factory cannot be made with this function return None.
    """
    A = R_j_i
    b = u_j
    A_slacked = np.concatenate([R_j_i, -1 * np.identity(R_j_i.shape[0], dtype=Fraction)], axis=1)
    c = c_i
    c_slacked = np.concatenate([c_i, np.zeros(R_j_i.shape[0], dtype=Fraction)])

    #Ax>=b
    #Ax+k=b

    #assert inverse_analysis(A, b) #A has a right-handed inverse.
    #assert inverse_analysis(A_slacked, b) #A has a right-handed inverse.

    for solver in PRIMARY_LP_SOLVERS:
        result = solver(A, b, c)
        if not result is None:
            return result
    for solver in PRIMARY_LP_SOLVERS:
        result = solver(A_slacked, b, c_slacked)
        if not result is None:
            return result[:c.shape[0]]
    for solver in BACKUP_LP_SOLVERS:
        result = solver(A, b, c)
        if not result is None:
            return result
    for solver in BACKUP_LP_SOLVERS:
        result = solver(A_slacked, b, c_slacked)
        if not result is None:
            return result[:c.shape[0]]

    return None


def solve_pricing_model_calculation_problem(R_j_i: np.ndarray, s_i: np.ndarray, u_j: np.ndarray, c_i: np.ndarray) -> np.ndarray:
    """
    Calculates a pricing model given a list of constructs, their usages, the target output, and the inital pricing model.
    Attempts to use the various linear programming solvers until one works. 
    Runs PRIMARY_LP_SOLVERS without force, then PRIMARY_LP_SOLVERS with force, then BACKUP_LP_SOLVERS without force, then BACKUP_LP_SOLVERS without force.
    
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
    S_k = np.diag(1-np.isclose(np.array(s_i), 0)).astype(Fraction)
    P_j = np.diag(1-np.isclose(R_j_i @ s_i, u_j)).astype(Fraction)

    A = np.concatenate([np.concatenate([R_j_i.T, np.identity(m, dtype=Fraction)], axis=1),
                        np.concatenate([np.zeros((m, n), dtype=Fraction), S_k], axis=1),
                        np.concatenate([P_j, np.zeros((n, m), dtype=Fraction)], axis=1)], axis=0)
    b = np.concatenate([c_i, np.zeros(m + n, dtype=Fraction)])
    A_slacked = np.concatenate([A, np.identity(A.shape[0])], axis=1)
    slack_penalty = np.concatenate([np.zeros(A.shape[1]), np.ones(A.shape[0])])

    #assert inverse_analysis(A, b) #No linear independence found.

    for solver in PRIMARY_LP_SOLVERS:
        result = solver(A, b)
        if not result is None:
            return result[:n]
    for solver in PRIMARY_LP_SOLVERS:
        result = solver(A_slacked, b, slack_penalty)
        if not result is None:
            return result[:n]
    for solver in BACKUP_LP_SOLVERS:
        result = solver(A, b)
        if not result is None:
            return result[:n]
    for solver in BACKUP_LP_SOLVERS:
        result = solver(A_slacked, b, slack_penalty)
        if not result is None:
            return result[:n]

    return None

