import numpy as np
import scipy as sp
import scipy.sparse
from scipy import optimize

from utils import *
from generators import *

from numbers import Real
from typing import TypeVar, Callable, Hashable, Iterable, Any, Optional


def generate_scipy_linear_solver(method: str = "revised simplex", options: dict = {'pivot': 'bland', 'maxiter': 50000, 'presolve': True}) -> Callable[[np.ndarray[Fraction], np.ndarray[Fraction], Optional[np.ndarray[Fraction]]], np.ndarray[Real]]:
    """
    Returns a solver for the standard linear programming problem using scipy.optimize.linprog
    A@x=b, x>=0, minimize cx
    Defaults to revised simplex with Bland pivoting.
    """
    def solver(A: np.ndarray[Fraction], b: np.ndarray[Fraction], c: np.ndarray[Fraction] | None = None):
        if c is None:
            c = np.zeros(A.shape[1], dtype=Fraction)
        result = optimize.linprog(c.astype(np.longdouble), A_eq=A.astype(np.longdouble), b_eq=b.astype(np.longdouble), method=method, options=options)
        logging.info(result.message+" "+str(result.status))
        if result.status in [0,1,4]: #4 usually indicates possible issues with simplex reaching optimal, we leave it in there because most of the time its pretty close.
            return result.x
        return None
    return solver


def solve_optimization_problem_scipy(R_j_i, u_j: np.ndarray, c_i: np.ndarray, method: str = "revised simplex") -> sp.array:
    """
    Solve an optimization problem given a linear transformation on construct counts, a target output vector, and a cost vector.
    First tries to solve it with equality constraints, if that fails its relaxed to inequality constraints.
    
    Parameters
    ----------
    R_j_i:
        Sparse matrix representing the linear transformation from a construct array to results.
    u_j:
        Vector of required outputs.
    c_i:
        Vector of costs of constructs.
    method:
        Method to use in optimizer.

    Returns
    -------
    Vector of rates that each construct is used in optimal factory.
    """
    try:
        optimization_result = scipy.optimize.linprog(c_i, A_eq=R_j_i, b_eq=u_j, bounds=(0, None), method=method, options={'pivot': 'bland', 'maxiter': 50000})
        assert optimization_result.status in [0,1,4], optimization_result.message
    except:
        optimization_result = scipy.optimize.linprog(c_i, A_ub=-1*R_j_i, b_ub=-1*u_j, bounds=(0, None), method=method, options={'pivot': 'bland', 'maxiter': 50000})
        assert optimization_result.status in [0,1,4], optimization_result.message

    return optimization_result.x


def calculate_pricing_model_via_optimal_scipy(R_j_i: np.ndarray, s_i: np.ndarray, u_j: np.ndarray, c_i: np.ndarray, method: str = "revised simplex", R_j_i_refs: list[str] | None = None) -> sp.array:
    """
    Calculates a pricing model given a list of constructs, their usages, the target output, and the inital pricing model.
    
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
    method:
        What method the solver should use.
    
    Returns
    -------
    Vector representing the pricing model of an optimal setup.
    """
    m = len(s_i)
    n = len(u_j)

    o_j = R_j_i @ s_i #actual outputs
    
    zero_lagrangians = np.where(1-np.isclose(np.array(s_i), 0))[0].tolist()

    bounds = []
    for j in range(n):
        bounds.append((None if np.isclose(o_j, u_j).all() else 0, None if np.isclose(o_j[j], u_j[j]) else 0))
        #np.isclose(o_j, u_j).all() means equality being done. 
        #if equality is done there are no bounds, otherwise its strictly positive and 0 where decoupled due to dual feasiblity.
    for i in range(m):
        bounds.append((0, 0 if i in zero_lagrangians else None))

    #print(m)
    #print(n)
    #print(R_j_i.shape)
    #print(np.identity(m).shape)
    A_eq = np.concatenate([R_j_i, np.identity(m)], axis=0).T
    #print(A_eq.shape)
    b_eq = c_i
    #print(b_eq.shape)
    c = np.zeros(n + m)
    #print(c.shape)
    #print(bounds)

    optimization_result = scipy.optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method, options={'pivot': 'bland', 'maxiter': 50000})
    try:
        assert optimization_result.status in [0,1,4], optimization_result.message
    except:
        assert optimization_result.status==2 and not R_j_i_refs is None
        print(A_eq.shape) #514, 900
        print(len(R_j_i_refs)) #514
        i = 0
        while i < len(R_j_i_refs):
            print(i)
            A_eq_cp = np.delete(A_eq, i, axis=0)
            b_eq_cp = np.delete(b_eq, i, axis=0)
            print(A_eq_cp.shape)
            print(b_eq_cp.shape)
            if scipy.optimize.linprog(np.zeros(A_eq.shape[1]), A_eq=A_eq_cp, b_eq=b_eq_cp, bounds=bounds, method=method, options={'pivot': 'bland', 'maxiter': 50000}).status==2:
                A_eq = A_eq_cp
                b_eq = b_eq_cp
                R_j_i_refs.pop(i)
            else:
                i += 1
        np.savetxt("simp_rev_A_eq.txt", A_eq)
        np.savetxt("simp_rev_b_eq.txt", b_eq)
        print(R_j_i_refs)
        raise AssertionError

    #print(optimization_result.x[:n])
    #print(optimization_result.x[n:])

    return optimization_result.x[:n]


def calculate_pricing_model_via_prebuilt_scipy(R_j_i: sp.array, C_j_i: sp.array, s_i: sp.array, reference_index: int, method: str  = "revised simplex"):
    """
    Calculates a pricing model for an already built factory. p_j[reference_index] = 1
    
    Parameters
    ----------
    R_j_i:
        Sparse matrix representing the linear transformation from a construct array to results.
    C_i_j:
        Sparse matrix representing the bilinear form that takes a pricing model and construct usage rates to return a cost.
    s_i:
        Vector of rates that each construct is used in optimal factory.
    reference_index:
        What index to base all the rest of the pricing model values on. 
        (Arbitrarity multiplying a pricing model made with this function by a value yields another permissible pricing model,
         the reference_index is used to determine scaling)
    method:
        What method the solver should use.
         
    Returns
    -------
    Vector representing the pricing model of the given setup.
    """
    m = len(s_i)
    n = len(u_j)
    
    u_j = R_j_i @ s_i
    
    raise NotImplementedError #im tired, what is exactly going on here with the legrangian?
    lagrangian_multiplier = sp.sparse.identity((m, m))

    A_eq = sp.sparse.hstack([C_j_i - 1 * R_j_i, lagrangian_multiplier])
    b_eq = np.zeros(A_eq.shape[1])

    optimization_result = scipy.optimize.linprog(np.zeros(n + m), A_eq=A_eq, b_eq=b_eq, bounds=(None, None), method=method)
    assert optimization_result.success, optimization_result.message

    return optimization_result.x[:n]

