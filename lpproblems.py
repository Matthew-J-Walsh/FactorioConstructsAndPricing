from globalsandimports import *

from lookuptables import *
from utils import *
from lpsolvers import *

def solve_factory_optimization_problem(construct: ComplexConstruct, u_j: np.ndarray, p0_j: np.ndarray, priced_indices: np.ndarray, known_technologies: TechnologicalLimitation,
                                       dual_guess: np.ndarray | None = None, spatial_mode: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    vectors, costs, true_costs, idents = construct.reduce(p0_j, priced_indices, dual, known_technologies, spatial_mode=spatial_mode)
    i = 0
    vc = vectors.copy()
    vc[vc < 0] = 0
    old_outputing_rows = vc.sum(axis=1) > 0
    while True:
        vc = vectors.copy()
        vc[vc < 0] = 0
        outputing_rows = vc.sum(axis=1) > 0
        assert (u_j[np.logical_not(outputing_rows)]<=0).all(), np.where(np.logical_and(u_j>0, np.logical_not(outputing_rows)))[0]#np.where(np.logical_and(old_outputing_rows, np.logical_not(outputing_rows)))[0]

        logging.info("Starting iteration "+str(i)+" of current optimization with "+str(len(idents))+" columns.")
        i += 1
        c_i = costs
        primal, dual = BEST_LP_SOLVER(vectors, u_j, c_i, ginv=dual)
        if primal is None or dual is None:
            logging.error("Row losses: "+str(np.where(np.logical_and(old_outputing_rows, np.logical_not(outputing_rows)))[0]))
            raise RuntimeError("Unable to form factory even with slack.")
        old_outputing_rows = outputing_rows

        assert not np.isclose(c_i, 0).all()
        assert not np.isclose(u_j, 0).all()
        assert not np.isclose(np.dot(u_j, dual), 0)
        assert not np.isclose(np.dot(c_i, primal), 0)
        assert not np.isclose(dual, 0).all()
        
        new_vectors, new_costs, new_true_costs, new_idents = construct.reduce(p0_j, priced_indices, dual, known_technologies, spatial_mode=spatial_mode)
        #optimal_new_columns = np.where(np.logical_or((new_vectors.T @ dual) >= .99 * new_costs, np.isclose(new_vectors.T @ dual, new_costs, rtol=SOLVER_TOLERANCES['rtol'], atol=SOLVER_TOLERANCES['atol'])))[0]
        optimal_new_columns = linear_transform_is_gt(new_vectors.T, dual, .99 * new_costs)
        new_vectors, new_costs, new_true_costs, new_idents = new_vectors[:, optimal_new_columns], new_costs[optimal_new_columns], new_true_costs[:, optimal_new_columns], new_idents[optimal_new_columns]
        new_mask = true_new_column_mask(idents, new_idents)
        true_news = np.where(new_mask)[0]
        if len(true_news)==0:
            break

        logging.info("New columns found are: "+"\n\t".join([repr((new_idents[true_news])[i])+" with column "+str(sparse.coo_matrix((new_vectors[:, true_news])[:, i]))+" \n\tand cost: "+str((new_costs[true_news])[i]) for i in range(new_idents[true_news].shape[0])]))

        #optimal_columns = np.logical_or((vectors.T @ dual) >= .99 * (c_i), np.isclose(vectors.T @ dual, c_i, rtol=SOLVER_TOLERANCES['rtol'], atol=SOLVER_TOLERANCES['atol']))
        optimal_columns = linear_transform_is_gt(vectors.T, dual, .99 * c_i)

        if spatial_mode:
            vectorsMinus = vectors.copy()
            vectorsMinus[vectorsMinus > 0] = 0
            vectorsPlus = vectors.copy()
            vectorsPlus[vectorsPlus < 0] = 0
            evalt = ((vectors.T @ dual) - c_i)
            evall = ((vectorsMinus.T @ dual))
            evalh = ((vectorsPlus.T @ dual))
            logging.info("Suboptimal columns removed are: "+"\n\t".join([repr(ident)+" with relative value: "+str(val)+" ; "+str(val_h)+" ; "+str(val_l)+" ; "+str(val_c) for ident, val, val_h, val_l, val_c in zip(idents[np.where(np.logical_not(optimal_columns))[0]], evalt[np.where(np.logical_not(optimal_columns))[0]], evall[np.where(np.logical_not(optimal_columns))[0]], evalh[np.where(np.logical_not(optimal_columns))[0]], c_i[np.where(np.logical_not(optimal_columns))[0]])]))
        else:
            logging.info("Suboptimal columns removed are: "+"\n\t".join([repr(ident)+" with relative value: "+str(val) for ident, val in zip(idents[np.where(np.logical_not(optimal_columns))[0]], ((vectors.T @ dual) / c_i)[np.where(np.logical_not(optimal_columns))[0]])]))

        vectors = np.concatenate([vectors[:, np.where(optimal_columns)[0]], new_vectors[:, true_news]], axis=1)
        costs = np.concatenate((costs[np.where(optimal_columns)[0]], new_costs[true_news]))
        true_costs = np.concatenate([true_costs[:, np.where(optimal_columns)[0]], new_true_costs[:, true_news]], axis=1)
        idents = np.concatenate((idents[np.where(optimal_columns)[0]], new_idents[true_news]))

        idents_hashes = np.array([hash(ide) for ide in idents])
        sort_list = idents_hashes.argsort()

        vectors, costs, true_costs, idents = vectors[:, sort_list], costs[sort_list], true_costs[:, sort_list], idents[sort_list]

    return primal, dual, vectors, costs, true_costs, idents


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


