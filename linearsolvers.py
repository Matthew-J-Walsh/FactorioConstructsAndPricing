from utils import *
from generators import *
import operator
import scipy.optimize


def optimize_for_outputs_via_reference_model(families: list[LinearConstructFamily], universal_reference_list: list[str], output_target: CompressedVector, 
                                             known_technologies: TechnologicalLimitation, ref_pricing_model: CompressedVector, method: str = "simplex", 
                                             output_type: str = '>=') -> tuple[sp.sparray, np.ndarary]:
    """
    Runs optimization on a list of ConstructFamilies given a reference pricing model to find the optimal factory
    to build targeted outputs.
    
    Parameters
    ----------
    families:
        List of LinearConstructFamily-s.
    universal_reference_list:
        The reference list used in the compilation of the constructs.
    output_target:
        A CompressedVector representing the requested outputs of the factory to be built.
    known_technologies:
        A TechnologicalLimitation representing the current state of technology.
        Consider using 'tech_objection_via_spec' to make this.
    ref_pricing_model:
        A CompressedVector representing the pricing model that defines the costs of the factory's components.
    method: 
        What method to use in the solver.
    output_type: (optional, default: ">=")
        ">=" or "=", defining if output_target(s) should be strictly met or if excess is permissable.
    
    Returns
    -------
    s_i:
        Vector of rates that each construct is used in optimal factory.
    p_i:                                       
        Pricing model of items/fluids from this optimal factory.
    """
    output_type = [operator.ge, operator.eq][['>=','='].index(output_type)]
    
    n = len(universal_reference_list)

    constructs = sum([f.get_constructs(known_technologies) for f in families], [])
    R_i_j = sp.sparse.vstack([construct.vector for construct in constructs]).T

    u_j = np.zeros(n)
    for k, v in output_target.items():
        u_j[universal_reference_list.index(k)] = v

    p0_i = np.zeros(n)
    for k, v in ref_pricing_model.items():
        p0_i[universal_reference_list.index(k)] = v

    c_i = np.array([np.dot(p0_i, construct.cost) for construct in constructs])

    s_i = solve_optimization_problem(R_i_j, u_j, c_i, method, output_type=output_type)
    assert np.logical_or(np.isclose(R_i_j @ s_i, u_j), R_i_j @ s_i >= u_j).all(), "Somehow solution infeasible?"

    p_i = calculate_pricing_model_via_optimal(R_i_j, s_i, u_j, c_i, method, output_type=output_type)
    
    """
    logging.info("====================================")
    logging.info("Determined optimal setup to be: ")
    for i in range(len(s_i)):
        if s_i[i] != 0:
            logging.info(str(constructs[i].ident)+": "+str(s_i[i]))
    logging.info("====================================")
    
    logging.info("====================================")
    logging.info("Determined pricing model to be: ")
    for i in range(len(universal_reference_list)):
        if not np.isclose(0, p_i[i]):
            logging.info("%s = $%s", universal_reference_list[i], str(p_i[i]))
    logging.info("====================================")
    """
    
    return s_i, p_i


def solve_optimization_problem(R_i_j: sp.array, u_j: np.array, c_i: sp.array, method: str, output_type=operator.ge) -> sp.array:
    """
    Solve an optimization problem given a linear transformation on construct counts, a target output vector, and a cost vector.
    
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
    output_type:
        What the comparator being used is. operator.eq means that outputs must be exact, operator.ge means that outputs can be excessive.

    Returns
    -------
    Vector of rates that each construct is used in optimal factory.
    """
    if output_type==operator.ge:
        optimization_result = scipy.optimize.linprog(c_i, A_ub=-1*R_i_j, b_ub=-1*u_j, bounds=(0, None), method=method)
        assert optimization_result.success, optimization_result.message

    elif output_type==operator.eq:
        optimization_result = scipy.optimize.linprog(c_i, A_eq=R_i_j, b_eq=u_j, bounds=(0, None), method=method)
        assert optimization_result.success, optimization_result.message

    return optimization_result.x


def calculate_pricing_model_via_optimal(R_i_j, s_i, u_j, c_i, method, output_type=operator.ge):
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
    output_type:
        TODO
    
    Returns
    -------
    Vector representing the pricing model of an optimal setup.
    """
    m = len(s_i)
    n = len(u_j)

    p_j = (R_i_j @ s_i)
    
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


def calculate_pricing_model_via_prebuilt(R_i_j, C_i_j, s_i, u_j, reference_index, method, output_type=operator.ge):
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
    output_type:
        TODO
         
    Returns
    -------
    Vector representing the pricing model of the given setup.
    """
    m = len(s_i)
    n = len(u_j)
    
    p_j = (R_i_j @ s_i)
    
    zero_lagrangians = np.nonzero(s_i)[0].tolist()

    bounds = []
    for j in range(n):
        bounds.append((0 if output_type==operator.ge else None, None if np.isclose(p_j[j], u_j[j]) else 0))
    bounds[reference_index] = (1, 1)
    for i in range(m):
        bounds.append((0 if output_type==operator.ge else None, 0 if i in zero_lagrangians else None))
    

    lagrangian_multiplier = sp.sparse.identity((m, m))

    A_eq = sp.sparse.hstack([C_i_j - 1 * R_i_j, lagrangian_multiplier])
    b_eq = np.zeros(A_eq.shape[1])

    optimization_result = scipy.optimize.linprog(np.ones(n + m), A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method)
    assert optimization_result.success, optimization_result.message

    return optimization_result.x[:n]

