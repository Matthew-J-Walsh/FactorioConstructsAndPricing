from globalsandimports import *

from lookuptables import *
from utils import *
from scipysolvers import generate_scipy_linear_solver
from pulpsolvers import generate_pulp_linear_solver, generate_pulp_dual_solver, generate_pulp_dense_solver, pulp_dense_solver
from scipsolvers import generate_scip_linear_solver, generate_scip_dual_solver
from highssolvers import generate_highs_linear_solver, generate_highs_dual_solver, generate_highs_dual_solver_pythonic


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
    def verified(A: sparse.csr_matrix, b: np.ndarray, c: np.ndarray | None = None, g: np.ndarray | None = None) -> np.ndarray | None:
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
    def verified(A: sparse.csr_matrix, b: np.ndarray, c: np.ndarray, g: np.ndarray | None = None, ginv: np.ndarray | None = None) -> Tuple[np.ndarray | None, np.ndarray | None]:
        try:
            logging.debug("Trying the "+name+" solver.")
            primal, dual = solver(A, b, c=c, g=g)
            if not (primal is None or dual is None):
                if not linear_transform_is_gt(A, primal, b).all():
                    if DEBUG_SOLVERS:
                        raise AssertionError(np.max(np.abs(A @ primal - b)))
                    else:
                        logging.warning(name+" gave a result but result wasn't feasible. As debugging is off this won't throw an error. Returning None.")
                        logging.warning("\tLargest recorded error is: "+str(A @ primal - b))
                        return None, None
                if not linear_transform_is_gt(-1 * A.T, dual, -1 * c).all():
                    if DEBUG_SOLVERS:
                        raise AssertionError(np.max(np.abs(A.T @ dual - c)))
                    else:
                        logging.warning(name+" gave a result but dual wasn't feasible. As debugging is off this won't throw an error. Returning None.")
                        logging.warning("\tLargest recorded error is: "+str(A.T @ dual - c))
                        return None, None
                if not np.isclose(np.dot(primal, c), np.dot(dual, b), rtol=SOLVER_TOLERANCES['rtol'], atol=SOLVER_TOLERANCES['atol']):
                    if DEBUG_SOLVERS:
                        raise AssertionError(np.dot(primal, c) - np.dot(dual, b))
                    else:
                        logging.warning(name+" gave a result but didn't forfill strong duality. As debugging is off this won't throw an error. Returning None.")
                        logging.warning("\tDuality gap is: "+str(np.dot(primal, c))+", "+str(np.dot(dual, b)))
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

def verified_dense_solver(solver: CallableDenseSolver, name: str) -> CallableDenseSolver:
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
    def verified(A: np.ndarray, b: np.ndarray, c: np.ndarray, g: np.ndarray | None = None, ginv: np.ndarray | None = None) -> Tuple[np.ndarray | None, np.ndarray | None]:
        try:
            logging.debug("Trying the "+name+" solver.")
            primal, dual = solver(A, b, c=c, g=g)
            if not (primal is None or dual is None):
                if not linear_transform_is_gt(A, primal, b).all():
                    if DEBUG_SOLVERS:
                        raise AssertionError(np.max(np.abs(A @ primal - b)))
                    else:
                        logging.warning(name+" gave a result but result wasn't feasible. As debugging is off this won't throw an error. Returning None.")
                        logging.warning("\tLargest recorded error is: "+str(A @ primal - b))
                        return None, None
                if not linear_transform_is_gt(-1 * A.T, dual, -1 * c).all():
                    if DEBUG_SOLVERS:
                        raise AssertionError(np.max(np.abs(A.T @ dual - c)))
                    else:
                        logging.warning(name+" gave a result but dual wasn't feasible. As debugging is off this won't throw an error. Returning None.")
                        logging.warning("\tLargest recorded error is: "+str(A.T @ dual - c))
                        return None, None
                if not np.isclose(np.dot(primal, c), np.dot(dual, b), rtol=SOLVER_TOLERANCES['rtol'], atol=SOLVER_TOLERANCES['atol']):
                    if DEBUG_SOLVERS:
                        raise AssertionError(np.dot(primal, c) - np.dot(dual, b))
                    else:
                        logging.warning(name+" gave a result but didn't forfill strong duality. As debugging is off this won't throw an error. Returning None.")
                        logging.warning("\tDuality gap is: "+str(np.dot(primal, c))+", "+str(np.dot(dual, b)))
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
    def dual_solver(A: sparse.csr_matrix, b: np.ndarray, c: np.ndarray, g: np.ndarray | None = None, ginv: np.ndarray | None = None) -> Tuple[np.ndarray | None, np.ndarray | None]:
        dual_D, primal_D = solver(-1 * A.T, -1 * c, -1 * b, g=ginv)
        return primal_D, dual_D
    return dual_solver

def iterative_dual_informed_unrelaxation(solver: CallableDualSolver, dual_mode: bool = False) -> CallableDualSolver:
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
    def relaxation_solver(A: sparse.csr_matrix, b: np.ndarray, c: np.ndarray, g: np.ndarray | None = None, ginv: np.ndarray | None = None) -> Tuple[np.ndarray | None, np.ndarray | None]:
        if A.shape[1] < A.shape[0] * 10: #problems that are faster to do in 1 go
            return solver(A, b, c=c, g=g)
        
        if ginv is None:
            ginv = np.ones(A.shape[0])

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
            if dual_mode:
                dual, masked_primal = solver(-1 * sparse.csr_matrix(Acsr[:, np.where(column_mask)[0]]).T, -1 * c[np.where(column_mask)[0]], -1 * b)
            else:
                masked_primal, dual = solver(sparse.csr_matrix(Acsr[:, np.where(column_mask)[0]]), b, c[np.where(column_mask)[0]])
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

def spatial_mode_assisted_solver(solver: CallableDualSolver) -> CallableDualSolver:
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
    def assisted_solver(A: sparse.csr_matrix, b: np.ndarray, c: np.ndarray, g: np.ndarray | None = None, ginv: np.ndarray | None = None) -> Tuple[np.ndarray | None, np.ndarray | None]:
        if ginv is None:
            ginv = np.ones(A.shape[0])

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
        
        masked_primal, masked_dual = solver(sparse.csr_matrix(Acsr[:, np.where(initial_mask)[0]]), b, c[np.where(initial_mask)[0]])
        unmasked_primal = np.zeros(A.shape[1])
        assert unmasked_primal.shape[0]==initial_mask.shape[0]
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
    def timing_solver(A: sparse.csr_matrix, b: np.ndarray, c: np.ndarray, g: np.ndarray | None = None, ginv: np.ndarray | None = None) -> Tuple[np.ndarray | None, np.ndarray | None]:
        logging.info("Starting a timing solver.")
        #standard = solver
        #flipped = flip_dual_solver(solver)
        iterative_di = iterative_dual_informed_unrelaxation(solver)
        iterative_ddi = iterative_dual_informed_unrelaxation(solver, True)
        #assisted = spatial_mode_assisted_solver(solver)
        for slvr, name in zip([iterative_di, iterative_ddi], ["iterative_di", "iterative_ddi"]):
            start = time.time()
            slvr(A, b, c, g=g, ginv=ginv)
            end = time.time()
            logging.info(name+" ran in "+str(end-start)+"s")
        return iterative_di(A, b, c, g=g, ginv=ginv)
    return timing_solver

"""
DUAL_LP_SOLVERS are lists of solvers for problems of the form:
A@x>=b, x>=0, minimize cx.
A.T@y>=c, y>=0, minimize by.
Ordered list, when a LP problem is attempted to be solved these should be ran in order. This order is mostly due to personal experience in usefulness.
"""
DUAL_LP_SOLVERS_PRIMAL = list(map(verified_dual_solver,
                           [generate_pulp_dual_solver(),
                            generate_highs_dual_solver_pythonic(),
                            #generate_scip_dual_solver(),
                           ],
                           ["pulp CBC",
                            "highspy",
                            #"scip",
                           ]))
#DUAL VERSION
DUAL_LP_SOLVERS = list(map(flip_dual_solver, DUAL_LP_SOLVERS_PRIMAL))
ITERATIVE_DUAL_LP_SOLVERS = list(map(iterative_dual_informed_unrelaxation, DUAL_LP_SOLVERS))
TIMEING_TEST_DUAL_LP_SOLVERS = list(map(mass_dual_solver_timing_test, DUAL_LP_SOLVERS_PRIMAL))
#ITERATIVE_DUAL_LP_SOLVERS = list(map(iterative_dual_informed_unrelaxation, DUAL_LP_SOLVERS_PRIMAL))

#BEST_LP_SOLVER: CallableDualSolver = verified_dual_solver(generate_pulp_dual_solver(), "pulp CBC")
#BEST_LP_SOLVER: CallableDenseSolver = verified_dense_solver(generate_pulp_dense_solver(), "pulp CBC")
BEST_LP_SOLVER = pulp_dense_solver


def solve_factory_optimization_problem(construct: ComplexConstruct, u_j: np.ndarray, p0_j: np.ndarray, priced_indices: np.ndarray, known_technologies: TechnologicalLimitation,
                                       dual_guess: np.ndarray | None = None, spatial_mode: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve an optimization problem given a ComplexConstruct, a target output vector, and a cost vector.
    Attempts to use the various linear programming solvers until one works. 
    Runs PRIMARY_LP_SOLVERS without slack, then PRIMARY_LP_SOLVERS with slack, then BACKUP_LP_SOLVERS without slack, then BACKUP_LP_SOLVERS without slack.
    
    Parameters
    ----------
    construct:
        Construct being optimized, usually a ComplexConstruct of the entire factory.
    u_j:
        Target output vector.
    p0_j:
        Initial pricing model.
    known_technologies:
        Current tech level.
    dual_guess:
        Guess for the dual vector if applicable.
        
    Returns
    -------
    primal:
        The optimal factory vector.
    dual:
        The optimal pricing model for the factory.
    vectors:
        The sparse array of the effects of the factory elements.
    costs:
        The sparse array of the costs of the factory elements.
    idents:
        The array of identifiers of the constructs used.
    """
    primal, dual = None, dual_guess

    vectors, costs, idents = construct.reduce(p0_j, priced_indices, dual, known_technologies, spatial_mode=spatial_mode)
    i = 0
    while True:
        logging.info("Starting iteration "+str(i)+" of current two-phase optimization with "+str(len(idents))+" columns.")
        i += 1
        c_i = costs.T @ p0_j
        primal, dual = BEST_LP_SOLVER(vectors, u_j, c_i, ginv=dual)
        if primal is None or dual is None:
            raise RuntimeError("Unable to form factory even with slack.")
        
        new_vectors, new_costs, new_idents = construct.reduce(p0_j, priced_indices, dual, known_technologies, spatial_mode=spatial_mode)
        optimal_new_columns = np.where((new_vectors.T @ dual) > .99 * (new_costs.T @ p0_j))[0]
        new_vectors, new_costs, new_idents = new_vectors[:, optimal_new_columns], new_costs[:, optimal_new_columns], new_idents[optimal_new_columns]
        new_mask = true_new_column_mask(idents, new_idents)
        true_news = np.where(new_mask)[0]
        if len(true_news)==0:
            break

        logging.info("New columns found are: "+"\n\t".join([repr(ident) for ident in new_idents[true_news]]))

        optimal_columns = (vectors.T @ dual) > .99 * (c_i)

        logging.info("Suboptimal columns aremoved are: "+"\n\t".join([repr(ident) for ident in idents[np.where(optimal_columns)[0]]]))

        vectors = np.concatenate([vectors[:, np.where(optimal_columns)[0]], new_vectors[:, true_news]], axis=1)
        costs = np.concatenate([costs[:, np.where(optimal_columns)[0]], new_costs[:, true_news]], axis=1)
        idents = np.concatenate((idents[np.where(optimal_columns)[0]], new_idents[true_news]))

    return primal, dual, vectors, costs, idents


def solve_spatial_mode_factory_optimization_problem(construct: ComplexConstruct, u_j: np.ndarray, ps_j: np.ndarray, p0_j: np.ndarray, priced_indices: np.ndarray, known_technologies: TechnologicalLimitation,
                                       dual_guess: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve an optimization problem given a ComplexConstruct, a target output vector, and a cost vector.
    Attempts to use the various linear programming solvers until one works. 
    Runs PRIMARY_LP_SOLVERS without slack, then PRIMARY_LP_SOLVERS with slack, then BACKUP_LP_SOLVERS without slack, then BACKUP_LP_SOLVERS without slack.
    
    Parameters
    ----------
    construct:
        Construct being optimized, usually a ComplexConstruct of the entire factory.
    u_j:
        Target output vector.
    ps_j:
        Pricing costs for ore space.
    p0_j:
        Initial pricing model.
    known_technologies:
        Current tech level.
    dual_guess:
        Guess for the dual vector if applicable.
        
    Returns
    -------
    primal:
        The optimal factory vector.
    dual:
        The optimal pricing model for the factory.
    vectors:
        The sparse array of the effects of the factory elements.
    costs:
        The sparse array of the costs of the factory elements.
    idents:
        The array of identifiers of the constructs used.
    """
    raise RuntimeError()
    primal, dual, dualS = None, np.concatenate([np.array([0]), dual_guess]) if not dual_guess is None else None, dual_guess

    vectors, costs, idents = construct.reduce(ps_j, priced_indices, dualS, known_technologies, spatial_mode=True)
    i = 0
    while True:
        logging.info("Starting iteration "+str(i)+" of current two-phase optimization with "+str(len(idents))+" columns.")
        i += 1
        c_is = costs.T @ ps_j
        primal, dualS = BEST_LP_SOLVER(vectors, u_j, c_is)
        if primal is None or dualS is None:
            raise RuntimeError("Unable to form factory even with slack.")

        temp = np.dot(c_is, primal)
        c_i = costs.T @ p0_j
        primal, dual = BEST_LP_SOLVER(np.concatenate([c_is.reshape((1, -1)), vectors], axis=0), np.concatenate([np.array([np.dot(c_is, primal)]), u_j]), c_i)
        if primal is None or dual is None:
            raise RuntimeError("Unable to form factory even with slack.")
        assert np.dot(primal, c_is) >= temp, np.dot(primal, c_is) - temp

        new_vectorsS, new_costsS, new_identsS = construct.reduce(ps_j, priced_indices, dualS, known_technologies, spatial_mode=True)
        new_vectors, new_costs, new_idents = construct.reduce(p0_j, priced_indices, dual[1:], known_technologies)

        new_maskS = true_new_column_mask(idents, new_identsS)
        true_newsS = np.where(new_maskS)[0]
        new_mask = true_new_column_mask(idents, new_idents)
        true_news = np.where(new_mask)[0]
        if len(true_news)==0 and len(true_newsS)==0:
            break
        logging.info("Number of new columns added: "+str(len(true_news)+len(true_newsS)))

        suboptimal_columns = np.logical_and((vectors.T @ dualS) < .99 * (c_is), (vectors.T @ dual[1:]) < .99 * (c_i))
        logging.info("Number of old columns removed: "+str(len(np.where(suboptimal_columns)[0])))

        vectors = np.concatenate([vectors[:, np.where(np.logical_not(suboptimal_columns))[0]], new_vectorsS[:, true_newsS], new_vectors[:, true_news]], axis=1)
        costs = np.concatenate([costs[:, np.where(np.logical_not(suboptimal_columns))[0]], new_costsS[:, true_newsS], new_costs[:, true_news]], axis=1)
        idents = np.concatenate((idents[np.where(np.logical_not(suboptimal_columns))[0]], new_identsS[true_newsS], new_idents[true_news]))
        
    #return primal, dual[1:], vectors, costs, idents
    return primal, dualS, vectors, costs, idents

def true_new_column_mask(idents, new_idents):
    new_mask = np.full(new_idents.shape[0], True, dtype=bool)
    for i in range(new_mask.shape[0]):
        for j in range(idents.shape[0]):
            if new_idents[i]==idents[j]:
                new_mask[i] = False
                break
            if hash(new_idents[i]) < hash(idents[j]):
                break
    return new_mask


