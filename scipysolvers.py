from utils import *
from generators import *
import operator
import numpy as np
import scipy as sp
import scipy.optimize


def solve_optimization_problem(R_i_j: sp.array, u_j: np.array, c_i: sp.array, method: str = "simplex") -> sp.array:
    """
    Solve an optimization problem given a linear transformation on construct counts, a target output vector, and a cost vector.
    First tries to solve it with equality constraints, if that fails its relaxed to inequality constraints.
    
    Parameters
    ----------
    R_i_j:
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
        optimization_result = scipy.optimize.linprog(c_i, A_eq=R_i_j, b_eq=u_j, bounds=(0, None), method=method)
        assert optimization_result.success, optimization_result.message
    except:
        optimization_result = scipy.optimize.linprog(c_i, A_ub=-1*R_i_j, b_ub=-1*u_j, bounds=(0, None), method=method)
        assert optimization_result.success, optimization_result.message

    return optimization_result.x


def calculate_pricing_model_via_optimal(R_i_j: sp.array, s_i: sp.array, u_j: np.array, c_i: sp.array, method: str = "simplex") -> sp.array:
    """
    Calculates a pricing model given a list of constructs, their usages, the target output, and the inital pricing model.
    
    Parameters
    ----------
    R_i_j:
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

    p_j = (R_i_j @ s_i)
    
    raise NotImplementedError #dual feasiblity detection.
    zero_lagrangians = np.nonzero(s_i)[0].tolist()

    bounds = []
    for j in range(n):
        bounds.append((0, None if np.isclose(p_j[j], u_j[j]) else 0))
    for i in range(m):
        bounds.append((0, 0 if i in zero_lagrangians else None))

    A_eq = sp.sparse.hstack([R_i_j, sp.sparse.identity(m)])
    b_eq = c_i

    optimization_result = scipy.optimize.linprog(np.ones(n + m) , A_eq=A_eq.to_dense(), b_eq=b_eq, bounds=bounds, method=method)
    assert optimization_result.success, optimization_result.message

    return optimization_result.x[:n]


def calculate_pricing_model_via_prebuilt(R_i_j: sp.array, C_i_j: sp.array, s_i: sp.array, u_j: np.array, reference_index: int, method: str  = "simplex"):
    """
    Calculates a pricing model for an already built factory. p_j[reference_index] = 1
    
    Parameters
    ----------
    R_i_j:
        Sparse matrix representing the linear transformation from a construct array to results.
    C_i_j:
        Sparse matrix representing the bilinear form that takes a pricing model and construct usage rates to return a cost.
    s_i:
        Vector of rates that each construct is used in optimal factory.
    u_j:
        Vector of required outputs.
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
    
    p_j = (R_i_j @ s_i)
    
    raise NotImplementedError #im tired, what is exactly going on here with the legrangian?
    lagrangian_multiplier = sp.sparse.identity((m, m))

    A_eq = sp.sparse.hstack([C_i_j - 1 * R_i_j, lagrangian_multiplier])
    b_eq = np.zeros(A_eq.shape[1])

    optimization_result = scipy.optimize.linprog(np.ones(n + m), A_eq=A_eq, b_eq=b_eq, bounds=(None, None), method=method)
    assert optimization_result.success, optimization_result.message

    return optimization_result.x[:n]

