from globalsandimports import *

from lookuptables import *
from utils import *
from lpsolvers import *

def solve_factory_optimization_problem(construct: ComplexConstruct, u_j: np.ndarray, cost_function: Callable[[CompiledConstruct, np.ndarray], np.ndarray], inverse_priced_indices: np.ndarray, known_technologies: TechnologicalLimitation,
                                       dual_guess: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve an optimization problem given a ComplexConstruct, a target output vector, and a cost vector.

    Parameters
    ----------
    construct : ComplexConstruct
        Construct being optimized, usually a ComplexConstruct of the entire factory
    u_j : np.ndarray
        Target output vector
    p0_j : np.ndarray
        Initial pricing model
    inverse_priced_indices : np.ndarray
        What indices of the pricing vector aren't priced
    known_technologies : TechnologicalLimitation
        Current tech level
    dual_guess : np.ndarray | None, optional
        Guess for the dual vector, by default None

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        The optimal factory vector,
        The optimal pricing model for the factory
        Matrix of the effect vectors of the used constructs
        Cost array of the used constructs
        Matrix of the cost vectors of the used construct
        Ident array of the used constructs

    Raises
    ------
    RuntimeError
        Unable to form a factory for whatever reason
    """
    primal, dual = None, dual_guess

    vectors, costs, true_costs, idents = construct.reduce(cost_function, inverse_priced_indices, dual, known_technologies)
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
        
        new_vectors, new_costs, new_true_costs, new_idents = construct.reduce(cost_function, inverse_priced_indices, dual, known_technologies)
        optimal_new_columns = linear_transform_is_gt(new_vectors.T, dual, .99 * new_costs)
        new_vectors, new_costs, new_true_costs, new_idents = new_vectors[:, optimal_new_columns], new_costs[optimal_new_columns], new_true_costs[:, optimal_new_columns], new_idents[optimal_new_columns]
        new_mask = true_new_column_mask(idents, new_idents)
        true_news = np.where(new_mask)[0]
        if len(true_news)==0:
            break

        logging.debug("New columns found are: "+"\n\t".join([repr((new_idents[true_news])[i])+" with column "+str(sparse.coo_matrix((new_vectors[:, true_news])[:, i]))+" \n\tand cost: "+str((new_costs[true_news])[i]) for i in range(new_idents[true_news].shape[0])]))

        optimal_columns = linear_transform_is_gt(vectors.T, dual, .99 * c_i)

        #if spatial_mode:
        #    logging.debug("Suboptimal columns removed are: "+"\n\t".join([repr(ident)+" with relative value: "+str(val)+" ; "+str(val_h)+" ; "+str(val_l)+" ; "+str(val_c) for ident, val, val_h, val_l, val_c in zip(idents[np.where(np.logical_not(optimal_columns))[0]], evalt[np.where(np.logical_not(optimal_columns))[0]], evall[np.where(np.logical_not(optimal_columns))[0]], evalh[np.where(np.logical_not(optimal_columns))[0]], c_i[np.where(np.logical_not(optimal_columns))[0]])]))
        #else:
        #    logging.debug("Suboptimal columns removed are: "+"\n\t".join([repr(ident)+" with relative value: "+str(val) for ident, val in zip(idents[np.where(np.logical_not(optimal_columns))[0]], ((vectors.T @ dual) / c_i)[np.where(np.logical_not(optimal_columns))[0]])]))

        vectors = np.concatenate([vectors[:, np.where(optimal_columns)[0]], new_vectors[:, true_news]], axis=1)
        costs = np.concatenate((costs[np.where(optimal_columns)[0]], new_costs[true_news]))
        true_costs = np.concatenate([true_costs[:, np.where(optimal_columns)[0]], new_true_costs[:, true_news]], axis=1)
        idents = np.concatenate((idents[np.where(optimal_columns)[0]], new_idents[true_news]))

        idents_hashes = np.array([hash(ide) for ide in idents])
        sort_list = idents_hashes.argsort()

        vectors, costs, true_costs, idents = vectors[:, sort_list], costs[sort_list], true_costs[:, sort_list], idents[sort_list]

    return primal, dual, vectors, costs, true_costs, idents

def true_new_column_mask(idents, new_idents) -> np.ndarray:
    """Calculates a mask for a new set of idents that removes already present elements

    Parameters
    ----------
    idents : _type_
        Old ident array
    new_idents : _type_
        New ident array

    Returns
    -------
    np.ndarray
        Mask of columns that are new
    """    
    new_mask = np.full(new_idents.shape[0], True, dtype=bool)
    for i in range(new_mask.shape[0]):
        for j in range(idents.shape[0]):
            if new_idents[i]==idents[j]:
                new_mask[i] = False
                break
            if hash(new_idents[i]) < hash(idents[j]):
                break
    return new_mask


