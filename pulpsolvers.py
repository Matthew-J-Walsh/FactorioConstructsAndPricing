import numpy as np
import pulp as pl

from utils import *
from generators import *

from numbers import Real
from typing import TypeVar, Callable, Hashable, Iterable, Any, Optional


def generate_pulp_linear_solver(pl_solver = pl.PULP_CBC_CMD()) -> Callable[[np.ndarray[Fraction], np.ndarray[Fraction], Optional[np.ndarray[Fraction]]], np.ndarray[Real]]:
    """
    Returns a solver for the standard linear programming problem using a PuLP solver
    A@x=b, x>=0, minimize cx
    Defaults to CBC.
    """
    def solver(A: np.ndarray[Fraction], b: np.ndarray[Fraction], c: np.ndarray[Fraction] | None = None):
        problem = pl.LpProblem()
        variables = pl.LpVariable.dicts("x", range(A.shape[1]), 0)
        if not c is None:
            problem += sum([c[i] * variables[i] for i in range(A.shape[1])])
        for j in range(b.shape[0]):
            problem += sum([A[j, i] * variables[i] for i in range(A.shape[1])]) == b[j]
        status = problem.solve(pl_solver)
        if status==1:
            return np.array([pl.value(v) if pl.value(v) else 0 for _, v in variables.items()])
        logging.info(pl.LpStatus[status])
        return None
    return solver


def solve_optimization_problem(R_j_i: np.ndarray, u_j: np.ndarray, c_i: np.ndarray, method: str = "highs-ipm") -> sp.array:
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
    #print(R_j_i.shape)
    #print(u_j.shape)
    #print(c_i.shape)
    try:
        optimization_problem = pl.LpProblem("Equality", pl.LpMinimize)
        optimization_variables = pl.LpVariable.dicts("x", range(c_i.shape[0]), 0)
        optimization_problem += sum([c_i[i] * optimization_variables[i] for i in range(c_i.shape[0])])
        for j in range(u_j.shape[0]):
            optimization_problem += sum([R_j_i[j, i] * optimization_variables[i] for i in range(c_i.shape[0])]) == u_j[j]
        status = optimization_problem.solve(pl.PULP_CBC_CMD())
        assert status==1, status
        optimization_result = np.array([pl.value(v) for _, v in optimization_variables.items()])
        assert linear_transform_is_close(R_j_i, optimization_result, u_j).all()
    except:
        logging.warning("Had to drop back to inequality.")
        optimization_problem = pl.LpProblem("Inequality", pl.LpMinimize)
        optimization_variables = pl.LpVariable.dicts("x", range(c_i.shape[0]), 0)
        optimization_problem += sum([c_i[i] * optimization_variables[i] for i in range(c_i.shape[0])])
        for j in range(u_j.shape[0]):
            optimization_problem += sum([R_j_i[j, i] * optimization_variables[i] for i in range(c_i.shape[0])]) >= u_j[j]
        status = optimization_problem.solve(pl.PULP_CBC_CMD())
        assert status==1, status
        optimization_result = np.array([pl.value(v) for _, v in optimization_variables.items()])
        assert linear_transform_is_gt(R_j_i, optimization_result, u_j).all()
    
    return optimization_result


def calculate_pricing_model_via_optimal(R_j_i: np.ndarray, s_i: np.ndarray, u_j: np.ndarray, c_i: np.ndarray, method: str = "highs-ipm") -> sp.array:
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

    u_j = R_j_i @ s_i #actual outputs
    
    zero_lagrangians = np.nonzero(s_i)[0].tolist()
    prox = linear_transform_is_close(R_j_i, s_i, u_j)

    bounds = []
    for j in range(n):
        bounds.append((None if prox.all() else 0, None if prox[j] else 0))
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

    optimization_problem = pl.LpProblem("Pricing", pl.LpMinimize)
    optimization_variables = [pl.LpVariable("x"+str(i), bounds[i][0], bounds[i][1]) for i in range(c.shape[0])]
    for j in range(b_eq.shape[0]):
        optimization_problem += sum([A_eq[j, i] * optimization_variables[i] for i in range(c.shape[0])]) == b_eq[j]
    status = optimization_problem.solve(pl.PULP_CBC_CMD())
    assert status==1, status
    optimization_result = np.array([0 if pl.value(v) is None else pl.value(v) for v in optimization_variables])
    assert linear_transform_is_close(A_eq, optimization_result, b_eq).all()

    return optimization_result[:n]


def calculate_pricing_model_via_prebuilt(R_j_i: sp.array, C_j_i: sp.array, s_i: sp.array, reference_index: int, method: str  = "highs"):
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

