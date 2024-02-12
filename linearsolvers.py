from utils import *
from generators import *
from sparsetensors import *
import operator
import scipy.optimize
from cyipoptTest import *

#contains only exact solvers. Sadge


def optimize_for_outputs_via_reference_model(families, universal_reference_list, output_target, known_technologies, ref_pricing_model, method="simplex", output_type='>='):
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
        A dict representing the requested outputs of the factory to be built.
    known_technologies:
        A TechnologicalLimitation representing the current state of technology.
        Consider using 'tech_objection_via_spec' to make this.
    ref_pricing_model:
        The pricing model that defines the costs of the factory components
    method: (optional, default: "COBYLA")
        Non-linear optimizer to use. Currently only "COBYLA", "trust-constr", "ipopt", and "SLSQP"
        work. Don't mess with this unless some error is being thrown.
    output_type: (optional, default: ">=")
        ">=" or "=", defining if output_target(s) should be strictly met or if excess is allowed.
    
    Returns
    -------
    R_i_j:
        ????????????????????????????????????????????
    s_i:
        List of rates that each construct is used in optimal factory.
    p_i:
        Pricing model of items/fluids from this optimal factory.
    """
    assert output_type in ['>=','=']
    output_type = [operator.ge, operator.eq][['>=','='].index(output_type)]
    for f in families:
        assert isinstance(f, LinearConstructFamily), f
    assert isinstance(output_target, dict), "Output target should be compressed as a dict."
    for k, v in output_target.items():
        assert k in universal_reference_list, "Target outputs aren't found in reference list."
    assert isinstance(ref_pricing_model, dict), "We expect pricing models to come in compressed as a dict."
    for k, v in ref_pricing_model.items():
        assert k in universal_reference_list, "Target outputs aren't found in reference list."
    
    n = len(universal_reference_list)

    constructs = sum([f.get_constructs(known_technologies) for f in families], [])
    R_i_j = concatenate_sparse_tensors([construct.vector for construct in constructs], 0)

    u_j = np.zeros(n)
    for k, v in output_target.items():
        u_j[universal_reference_list.index(k)] = v
    p0_i = np.zeros(n)
    for k, v in ref_pricing_model.items():
        p0_i[universal_reference_list.index(k)] = v
    logging.info(p0_i)
    logging.info([construct.cost.to_dense() for construct in constructs])
    c_i = np.array([np.dot(p0_i, construct.cost.to_dense()) for construct in constructs])

    logging.info("Developed a total of %d constructs" % len(c_i))
    logging.info(c_i)

    s_i = solve_optimization_problem(R_i_j, u_j, c_i, method, universal_reference_list=universal_reference_list, output_type=output_type)
    assert np.logical_or(np.isclose(R_i_j.to_dense().T @ s_i, u_j), R_i_j.to_dense().T @ s_i >= u_j).all(), "What?"
    
    logging.info("====================================")
    logging.info("Determined optimal setup to be: ")
    for i in range(len(s_i)):
        if s_i[i] != 0:
            logging.info(str(constructs[i].ident)+": "+str(s_i[i]))
    logging.info("====================================")

    p_i = calculate_pricing_model_via_optimal(R_i_j, s_i, u_j, c_i, method, universal_reference_list=universal_reference_list, output_type=output_type)
    
    logging.info("====================================")
    logging.info("Determined pricing model to be: ")
    for i in range(len(universal_reference_list)):
        if not np.isclose(0, p_i[i]):
            logging.info("%s = $%s", universal_reference_list[i], str(p_i[i]))
    logging.info("====================================")
    
    return R_i_j, s_i, p_i


def solve_optimization_problem(R_i_j, u_j, c_i, method, universal_reference_list=None, output_type=operator.ge):
    """
    Solve an optimization problem given a list of constructs, a target output vector, and an inital pricing model.
    Returns s_i, usage amounts
    """
    assert R_i_j.shape[0]==c_i.shape[0], str(R_i_j.shape)+" "+str(c_i.shape)
    assert R_i_j.shape[1]==u_j.shape[0], str(R_i_j.shape)+" "+str(u_j.shape)

    if output_type==operator.ge:
        A_ub = -1*(R_i_j.to_dense()).T
        b_ub = -1*u_j
        optimization_result = scipy.optimize.linprog(c_i, A_ub=-1*(R_i_j.to_dense()).T, b_ub=-1*u_j, bounds=(0, None), method=method)
        try:
            assert optimization_result.success, optimization_result.message
        except:
            print(-1 * A_ub)
            print(-1 * b_ub)
            raise AssertionError

    elif output_type==operator.eq:
        A_eq = (R_i_j.to_dense()).T
        b_eq = u_j
        optimization_result = scipy.optimize.linprog(c_i, A_eq=(R_i_j.to_dense()).T, b_eq=u_j, bounds=(0, None), method=method)

    return optimization_result.x

def calculate_pricing_model_via_optimal(R_i_j, s_i, u_j, c_i, method, universal_reference_list=None, output_type=operator.ge):
    """
    Calculates a pricing model given a list of constructs, their usages, the target output, and the inital pricing model.
    """
    m = len(s_i)
    n = len(u_j)
    zero_costs = []
    p_j = (R_i_j.to_dense().T @ s_i)
    for j in range(p_j.shape[0]):
        if not np.isclose(p_j[j], u_j[j]):
            zero_costs.append(j)
    
    zero_lagrangians = np.nonzero(s_i)[0].tolist()

    R_i_j_stripped = R_i_j.column_stripping(zero_costs)

    lagrangian_multiplier = SparseTensor((m, m - len(zero_lagrangians)))
    j = 0
    for i in range(lagrangian_multiplier.shape[0]):
        if not i in zero_lagrangians:
            lagrangian_multiplier[i, j] = 1
            j += 1

    A_eq = extra_dimensional_projection([R_i_j_stripped, lagrangian_multiplier], [0])
    b_eq = c_i

    optimization_result = scipy.optimize.linprog(np.ones(n - len(zero_costs) + m - len(zero_lagrangians)), A_eq=A_eq.to_dense(), b_eq=b_eq, bounds=(0, None), method=method)

    try:
        assert optimization_result.success, optimization_result.message
    except:
        print(A_eq.to_dense())
        print(b_eq)
        raise AssertionError
    
    logging.info(optimization_result.x)
    logging.info(A_eq.to_dense() @ optimization_result.x)
    logging.info(b_eq)

    p_j = []
    j = 0
    for i in range(n):
        if i in zero_costs:
            p_j.append(0)
        else:
            p_j.append(optimization_result.x[j])
            j += 1
    p_j = np.array(p_j)

    return p_j

def calculate_pricing_model_via_prebuilt(R_i_j, C_i_j, s_i, u_j, reference_index, method, universal_reference_list=None, output_type=operator.ge):
    """
    Calculates a pricing model for an already built factory. p_j[reference_index] = 1
    """
    m = len(s_i)
    n = len(u_j)
    zero_costs = []
    p_j = (R_i_j.to_dense().T @ s_i)
    for j in range(n):
        if not np.isclose(p_j[j], u_j[j]):
            zero_costs.append(j)
    assert not reference_index in zero_costs, "You need a reference value that isn't forced to be zero, choose some better number"
    zero_costs.append(reference_index)
    zero_costs.sort()

    offset = np.zeros(m)
    C_minus_R_i_j = (-1 * R_i_j) + C_i_j
    for coord, value in zip(C_minus_R_i_j.coords, C_minus_R_i_j.values):
        if coord[1]==reference_index:
            offset[coord[0]] += value

    zero_lagrangians = np.nonzero(s_i)[0].tolist()

    C_minus_R_i_j_stripped = C_minus_R_i_j.column_stripping(zero_costs)
    lagrangian_multiplier = SparseTensor((m, m - len(zero_lagrangians)))
    j = 0
    for i in range(lagrangian_multiplier.shape[0]):
        if not i in zero_lagrangians:
            lagrangian_multiplier[i, j] = -1
            j += 1

    A_eq = extra_dimensional_projection([C_minus_R_i_j_stripped, lagrangian_multiplier], [0])
    b_eq = -1 * offset
    print(C_minus_R_i_j.shape) #(m, n)
    print(C_minus_R_i_j_stripped.shape) #(m, n)
    print(lagrangian_multiplier.shape)
    print(A_eq.shape)
    print(b_eq.shape)

    optimization_result = scipy.optimize.linprog(np.ones(n - len(zero_costs) + m - len(zero_lagrangians)), A_eq=A_eq.to_dense(), b_eq=b_eq, bounds=(0, None), method=method)

    try:
        assert optimization_result.success, optimization_result.message
    except:
        print(A_eq.to_dense())
        print(b_eq)
        raise AssertionError
    
    logging.info(optimization_result.x)
    logging.info(A_eq.to_dense() @ optimization_result.x)
    logging.info(b_eq)

    p_j = []
    j = 0
    for i in range(n):
        if i in zero_costs:
            p_j.append(0)
        else:
            p_j.append(optimization_result.x[j])
            j += 1
    p_j = np.array(p_j)

    return p_j

