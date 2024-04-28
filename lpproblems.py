from globalsandimports import *

from utils import *
from scipysolvers import generate_scipy_linear_solver
from pulpsolvers import generate_pulp_linear_solver, generate_pulp_dual_solver
from scipsolvers import generate_scip_linear_solver, generate_scip_dual_solver
from highssolvers import generate_highs_linear_solver, generate_highs_dual_solver


def verified_solver(solver: CallableSolver, name: str) -> CallableSolver:
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
    def verified(A: sparse.coo_matrix, b: np.ndarray, c: np.ndarray | None = None, g: np.ndarray | None = None) -> np.ndarray | None:
        try:
            if not g is None:
                assert linear_transform_is_close(A, g, b).all(), "Guess isn't a valid point"
            logging.debug("Trying the "+name+" solver.")
            sol: np.ndarray | None = solver(A, b, c=c, g=g)
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

def verified_dual_solver(solver: CallableDualSolver, name: str) -> CallableDualSolver:
    """
    Returns a instance of the dual solver that verifies the result if given. Also eats errors unless debugging.

    Parameters
    ----------
    solver:
        Optimization function to use. Should solve problems of the form: 
        A@x>=b, x>=0, minimize c*x.
        It should also provide the dual solution to:
        A.T@y<=c, y>=0, minimize b.T*y. 
        If it cannot solve the problem for whatever reason it should return None.
    name:
        What to refer to solver as when giving warnings and throwing errors.

    Returns
    -------
    Solver with output verification and error catching added.
    """
    def verified(A: sparse.coo_matrix, b: np.ndarray, c: np.ndarray, g: np.ndarray | None = None, ginv: np.ndarray | None = None) -> Tuple[np.ndarray | None, np.ndarray | None]:
        try:
            logging.debug("Trying the "+name+" solver.")
            primal, dual = solver(A, b, c=c, g=g)
            if not (primal is None or dual is None):
                if not linear_transform_is_gt(A, primal, b).all():
                    if DEBUG_SOLVERS:
                        raise AssertionError(np.max(np.abs(A @ primal - b)))
                    else:
                        logging.warning(name+" gave a result but result wasn't feasible. As debugging is off this won't throw an error. Returning None.")
                        logging.warning("\tLargest recorded error is: "+str(np.max(np.abs(A @ primal - b))))
                        return None, None
                if not linear_transform_is_gt(-1 * A.T, dual, -1 * c).all():
                    if DEBUG_SOLVERS:
                        raise AssertionError(np.max(np.abs(A.T @ dual - c)))
                    else:
                        logging.warning(name+" gave a result but dual wasn't feasible. As debugging is off this won't throw an error. Returning None.")
                        logging.warning("\tLargest recorded error is: "+str(np.max(np.abs(A.T @ dual - c))))
                        return None, None
            else:
                logging.debug("Solver returned None.")
            return primal, dual
        except:
            if DEBUG_SOLVERS:
                raise RuntimeError(name)
            logging.warning(name+" threw an error. Returning None.")
            return None, None
    return verified

def flip_dual_solver(solver: CallableDualSolver) -> CallableDualSolver:
    """
    Solves a problem by solving its dual instead of its main.
    For the moment actually tries both and times it.

    Parameters
    ----------
    solver:
        Optimization function to use. Should solve problems of the form: 
        A@x>=b, x>=0, minimize c*x.
        It should also provide the dual solution to:
        A.T@y<=c, y>=0, minimize b.T*y. 
        If it cannot solve the problem for whatever reason it should return None.
    name:
        What to refer to solver as when giving warnings and throwing errors.

    Returns
    -------
    Solver instead targeting the dual problem.
    """
    def dual_solver(A: sparse.coo_matrix, b: np.ndarray, c: np.ndarray, g: np.ndarray | None = None, ginv: np.ndarray | None = None) -> Tuple[np.ndarray | None, np.ndarray | None]:
        dual_D, primal_D = solver(-1 * A.T, -1 * c, -1 * b, g=ginv)
        return primal_D, dual_D
    return dual_solver

def iterative_dual_informed_unrelaxation(solver: CallableDualSolver) -> CallableDualSolver:
    """
    Uses a primal + dual solver and iteratively adds rows based on the dual weightings.

    Parameters
    ----------
    solver:
        Optimization function to use. Should solve problems of the form: 
        A@x>=b, x>=0, minimize c*x.
        It should also provide the dual solution to:
        A.T@y<=c, y>=0, minimize b.T*y. 
        If it cannot solve the problem for whatever reason it should return None.
    """
    def relaxation_solver(A: sparse.coo_matrix, b: np.ndarray, c: np.ndarray, g: np.ndarray | None = None, ginv: np.ndarray | None = None) -> Tuple[np.ndarray | None, np.ndarray | None]:
        if A.shape[1] < A.shape[0] * 10: #problems that are faster to do in 1 go
            return solver(A, b, c=c, g=g)

        logging.info("Setting up unrelaxation.")
        column_mask = np.full(A.shape[1], False, dtype=bool)
        orthants = {}
        Acsr = A.tocsr()
        for i in range(A.shape[1]):
            orth = vectors_orthant(Acsr[:, i].toarray().flatten())
            if not orth in orthants.keys():
                orthants[orth] = [i]
            else:
                orthants[orth].append(i)
        logging.info("Found a total of "+str(len(orthants.keys()))+" different orthants.")
        for k in orthants.keys():
            orthants[k] = np.array(orthants[k])

        def unrelax_rows(p):
            for s in orthants.values():
                i = np.argmax(p @ (Acsr / c[None, :]).tocsr()[:, s]) # type: ignore
                if column_mask[s[i]]:
                    assert np.max(p @ (Acsr / c[None, :]).tocsr()[:, s]) <= 1.01, str(s[i]) # type: ignore
                column_mask[s[i]] = True
        unrelax_rows(ginv)
        
        logging.info("Beginning unrelaxation.")
        while True:
            print()
            masked_primal, dual = solver(sparse.coo_matrix(Acsr[:, np.where(column_mask)[0]]), b, c[np.where(column_mask)[0]])
            if masked_primal is None or dual is None:
                logging.error("Please debug solver")
                return None, None
            if linear_transform_is_gt(-1 * A.T, dual, -1 * c).all():
                logging.info("Unrelaxation terminated; reforming primal.")
                primal = np.zeros(A.shape[1])
                primal[np.where(column_mask)[0]] = masked_primal
                return primal, dual
            last_count = np.count_nonzero(column_mask)
            unrelax_rows(dual)
            logging.info("Iterating. Unmasking a total of "+str(np.count_nonzero(column_mask)-last_count)+" new rows.")
            if np.count_nonzero(column_mask)-last_count == 0:
                assert linear_transform_is_gt(-1 * Acsr[:, np.where(column_mask)[0]].tocoo().T, dual, -1 * c[np.where(column_mask)[0]]).all(), "HUH?"
                raise RuntimeError("No new row added?")

    return relaxation_solver

def two_phase_assisted_solver(solver: CallableDualSolver) -> CallableDualSolver:
    """
    Runs an assisted solving pattern. First runs solver, then runs

    Parameters
    ----------
    solver:
        Optimization function to use. Should solve problems of the form: 
        A@x>=b, x>=0, minimize c*x.
        It should also provide the dual solution to:
        A.T@y<=c, y>=0, minimize b.T*y. 
        If it cannot solve the problem for whatever reason it should return None.
    """
    def assisted_solver(A: sparse.coo_matrix, b: np.ndarray, c: np.ndarray, g: np.ndarray | None = None, ginv: np.ndarray | None = None) -> Tuple[np.ndarray | None, np.ndarray | None]:
        if ginv is None:
            ginv = np.ones(A.shape[1])
        initial_mask = np.full(A.shape[1], False, dtype=bool)
        orthants = {}
        Acsr = A.tocsr()
        for i in range(A.shape[1]):
            orth = vectors_orthant(Acsr[:, i].toarray().flatten())
            if not orth in orthants.keys():
                orthants[orth] = [i]
            else:
                orthants[orth].append(i)
        logging.info("Found a total of "+str(len(orthants.keys()))+" different orthants.")
        for k in orthants.keys():
            orthants[k] = np.array(orthants[k])

        for s in orthants.values():
            i = np.argmax(ginv @ (Acsr / c[None, :]).tocsr()[:, s]) # type: ignore
            if initial_mask[s[i]]:
                assert np.max(ginv @ (Acsr / c[None, :]).tocsr()[:, s]) <= 1.01, str(s[i]) # type: ignore
            initial_mask[s[i]] = True
        
        masked_primal, masked_dual = solver(sparse.coo_matrix(Acsr[:, np.where(initial_mask)[0]]), b, c[np.where(initial_mask)[0]])
        unmasked_primal = np.zeros(A.shape[0])
        unmasked_primal[np.where(initial_mask)[0]] = masked_primal
        primal, dual = solver(A, b, c=c, g=unmasked_primal)
        return primal, dual
    
    return assisted_solver

def mass_dual_solver_timing_test(solver: CallableDualSolver) -> CallableDualSolver:
    """
    Makes a solver that times every dual solver method.

    Parameters
    ----------
    solver:
        Optimization function to use. Should solve problems of the form: 
        A@x>=b, x>=0, minimize c*x.
        It should also provide the dual solution to:
        A.T@y<=c, y>=0, minimize b.T*y. 
        If it cannot solve the problem for whatever reason it should return None.
    """
    def timing_solver(A: sparse.coo_matrix, b: np.ndarray, c: np.ndarray, g: np.ndarray | None = None, ginv: np.ndarray | None = None) -> Tuple[np.ndarray | None, np.ndarray | None]:
        logging.info("Starting a timing solver.")
        standard = solver
        flipped = flip_dual_solver(solver)
        iterative_di = iterative_dual_informed_unrelaxation(solver)
        assisted = two_phase_assisted_solver(solver)
        for slvr, name in zip([standard, flipped, iterative_di, assisted], ["standard", "flipped", "iterative_di", "assisted"]):
            start = time.time()
            slvr(A, b, c, g=g, ginv=ginv)
            end = time.time()
            logging.info(name+" ran in "+str(start-end)+"s")
        return standard(A, b, c, g=g, ginv=ginv)
    return timing_solver

"""
DUAL_LP_SOLVERS are lists of solvers for problems of the form:
A@x>=b, x>=0, minimize cx.
A.T@y>=c, y>=0, minimize by.
Ordered list, when a LP problem is attempted to be solved these should be ran in order. This order is mostly due to personal experience in usefulness.
"""
DUAL_LP_SOLVERS_PRIMAL = list(map(verified_dual_solver,
                           [generate_highs_dual_solver(),
                            generate_pulp_dual_solver(),
                            #generate_scip_dual_solver(),
                           ],
                           ["highspy",
                            "pulp CBC",
                            #"scip",
                           ]))
#DUAL VERSION
DUAL_LP_SOLVERS = list(map(flip_dual_solver, DUAL_LP_SOLVERS_PRIMAL))
ITERATIVE_DUAL_LP_SOLVERS = list(map(iterative_dual_informed_unrelaxation, DUAL_LP_SOLVERS))
#ITERATIVE_DUAL_LP_SOLVERS = list(map(iterative_dual_informed_unrelaxation, DUAL_LP_SOLVERS_PRIMAL))

"""
PRIMARY_LP_SOLVERS and BACKUP_LP_SOLVERS are lists of solvers for problems of the form:
A@x=b, x>=0, minimize cx
Ordered list, when a LP problem is attempted to be solved these should be ran in order. This order is mostly due to personal experience in usefulness.
"""
PRIMARY_LP_SOLVERS = list(map(verified_solver,
                              [#pulp_solver_via_mps(),
                               generate_highs_linear_solver(),
                               generate_pulp_linear_solver(),
                               #generate_scip_linear_solver(),
                               ],
                               [#"pulp CBC mps",
                                "highspy",
                                "pulp CBC",
                                #"scip",
                                ]))
BACKUP_LP_SOLVERS = []
#BACKUP_LP_SOLVERS = list(map(verified_solver,
#                             [generate_scipy_linear_solver("highs-ipm", {}),
#                              generate_scipy_linear_solver("highs", {}),
#                              generate_scipy_linear_solver("highs-ds", {}),
#                              generate_scipy_linear_solver("simplex", {}),],
#                             ["highs-ipm",
#                              "highs",
#                              "highs-ds",
#                              "simplex",]))

ALL_SOLVER_NAMES_ORDERED = ["highspy", "pulp CBC", "scip", "highs-ipm", "highs", "highs-ds", "simplex"]

if BENCHMARKING_MODE:
    for s_name in ALL_SOLVER_NAMES_ORDERED:
        BENCHMARKING_TIMES[s_name] = 0


def solve_factory_optimization_problem(R_j_i: sparse.coo_matrix, u_j: np.ndarray, c_i: np.ndarray) -> np.ndarray:
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
    A_slacked = sparse.coo_matrix(sparse.hstack([R_j_i, -1 * sparse.eye(R_j_i.shape[0], format="coo")], format="coo"))
    assert A_slacked.shape[0]==R_j_i.shape[0] #TODO: remove check
    c = c_i
    c_slacked = np.concatenate([c_i, np.zeros(R_j_i.shape[0])])

    #Ax>=b
    #Ax+k=b

    #A has a right-handed inverse.
    #A_slacked has a right-handed inverse.

    if BENCHMARKING_MODE:
        result = PRIMARY_LP_SOLVERS[0](A_slacked, b, c_slacked)
        if not result is None:
            for i, solver in enumerate(PRIMARY_LP_SOLVERS + BACKUP_LP_SOLVERS):
                start_time = time.time()
                res = solver(A_slacked, b, c_slacked)
                end_time = time.time()
                if res is None:
                    BENCHMARKING_TIMES[ALL_SOLVER_NAMES_ORDERED[i]] = np.nan
                BENCHMARKING_TIMES[ALL_SOLVER_NAMES_ORDERED[i]] += end_time - start_time
            return result[:c.shape[0]]
    else:
        for solver in PRIMARY_LP_SOLVERS:
            result = solver(A_slacked, b, c_slacked)
            if not result is None:
                return result[:c.shape[0]]
        for solver in BACKUP_LP_SOLVERS:
            result = solver(A_slacked, b, c_slacked)
            if not result is None:
                return result[:c.shape[0]]
        
    raise ValueError("Unable to form factory even with slack.")


def solve_factory_optimization_problem_dual(R_j_i: sparse.coo_matrix, u_j: np.ndarray, c_i: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve an optimization problem given a linear transformation on construct counts, a target output vector, and a cost vector.
    Returns both a rate usage and a pricing model.
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
    primal:
        Vector of rates that each construct is used in optimal factory.
    dual:
        Pricing module of the 
    """
    for dual_solver in DUAL_LP_SOLVERS:
        primal, dual = dual_solver(R_j_i, u_j, c_i)
        if not primal is None and not dual is None:
            return primal, dual
        
    raise ValueError("Unable to form factory even with slack.")


def solve_factory_optimization_problem_dual_iteratively(R_j_i: sparse.coo_matrix, u_j: np.ndarray, c_i: np.ndarray, p0_j: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve an optimization problem given a linear transformation on construct counts, a target output vector, and a cost vector.
    Returns both a rate usage and a pricing model.
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
    primal:
        Vector of rates that each construct is used in optimal factory.
    dual:
        Pricing module of the 
    """
    for dual_solver in ITERATIVE_DUAL_LP_SOLVERS:
        primal, dual = dual_solver(R_j_i, p0_j, u_j, c_i)
        if not primal is None and not dual is None:
            return primal, dual
        
    raise ValueError("Unable to form factory even with slack.")

