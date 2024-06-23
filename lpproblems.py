from globalsandimports import *

from lookuptables import *
from utils import *
from lpsolvers import *

def solve_factory_optimization_problem(construct: ComplexConstruct, u_j: np.ndarray, cost_function: Callable[[CompiledConstruct, np.ndarray], np.ndarray], inverse_priced_indices: np.ndarray, known_technologies: TechnologicalLimitation, 
                                       dual_guess: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, ColumnTable]:
    """Solve an optimization problem given a ComplexConstruct, a target output vector, and a cost vector

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

    vector_table = construct.vectors(cost_function, inverse_priced_indices, dual, known_technologies).reduced.sorted

    i = 0
    old_valid_rows = vector_table.valid_rows
    while True:
        valid_rows = vector_table.valid_rows
        assert (u_j[np.logical_not(valid_rows)]<=0).all(), np.where(np.logical_and(u_j>0, np.logical_not(valid_rows)))[0]#np.where(np.logical_and(old_outputing_rows, np.logical_not(outputing_rows)))[0]

        logging.info("Starting iteration "+str(i)+" of current optimization with "+str(len(vector_table.ident))+" columns.")
        i += 1
        primal, dual = BEST_LP_SOLVER(vector_table.vector, u_j, vector_table.cost, ginv=dual)
        if primal is None or dual is None:
            logging.error("Row losses: "+str(np.where(np.logical_and(old_valid_rows, np.logical_not(valid_rows)))[0]))
            raise RuntimeError("Unable to form factory even with slack.")
        old_valid_rows = valid_rows

        assert not np.isclose(vector_table.cost, 0).all()
        assert not np.isclose(u_j, 0).all()
        assert not np.isclose(np.dot(u_j, dual), 0)
        assert not np.isclose(np.dot(vector_table.cost, primal), 0)
        assert not np.isclose(dual, 0).all()
        
        new_vector_table = construct.vectors(cost_function, inverse_priced_indices, dual, known_technologies).reduced.sorted
        new_vector_table.mask(linear_transform_is_gt(new_vector_table.vector.T, dual, .99 * new_vector_table.cost))
        
        new_mask = true_new_column_mask(vector_table.ident, new_vector_table.ident)
        true_news = np.where(new_mask)[0]
        if len(true_news)==0:
            break

        new_vector_table.mask(true_news)
        logging.debug("New columns found are: "+"\n\t".join([repr((new_vector_table.ident)[i])+" with column "+str(sparse.coo_matrix((new_vector_table.vector)[:, i]))+" \n\tand cost: "+str((new_vector_table.cost)[i]) for i in range(new_vector_table.ident.shape[0])]))
        
        vector_table.mask(linear_transform_is_gt(vector_table.vector.T, dual, .99 * vector_table.cost))
        vector_table = (vector_table + new_vector_table).sorted

    return primal, dual, vector_table

def solve_manual_factory_optimization_problem(construct: ComplexConstruct, u_j: np.ndarray, cost_function: Callable[[CompiledConstruct, np.ndarray], np.ndarray], inverse_priced_indices: np.ndarray, known_technologies: TechnologicalLimitation, 
                                              extras: ColumnTable,
                                              dual_guess: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, ColumnTable]:
    """Solve an optimization problem given a ComplexConstruct, a target output vector, and a cost vector.
    With extra columns that shouldn't be filtered and have max priority cost

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
    extras : tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray[CompressedVector, Any]] | None
        Extra constructs to be added every time, never filtered. 
        Same format as construct.reduce outputs
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

    vector_table = construct.vectors(cost_function, inverse_priced_indices, dual, known_technologies).shadow_attachment(extras).reduced.sorted

    prio_primal, prio_dual = BEST_LP_SOLVER(vector_table.vector, u_j, vector_table.true_cost[-1, :])
    if prio_primal is None or prio_dual is None:
        raise RuntimeError("Unable to form factory even with slack.")
    prio_cost = np.dot(prio_primal, vector_table.true_cost[-1, :])

    i = 0
    old_valid_rows = vector_table.valid_rows
    while True:
        valid_rows = vector_table.valid_rows
        assert (u_j[np.logical_not(valid_rows)]<=0).all(), np.where(np.logical_and(u_j>0, np.logical_not(valid_rows)))[0]#np.where(np.logical_and(old_outputing_rows, np.logical_not(outputing_rows)))[0]

        logging.info("Starting iteration "+str(i)+" of current optimization with "+str(len(vector_table.ident))+" columns.")
        i += 1
        
        primal, dual = BEST_LP_SOLVER(np.concatenate((vector_table.vector, vector_table.true_cost[-1, :].reshape(1, -1)), axis=0), 
                                                                                  np.concatenate((u_j, np.array([prio_cost]))), 
                                                                                  vector_table.cost)
        if primal is None or dual is None:
            logging.error("Row losses: "+str(np.where(np.logical_and(old_valid_rows, np.logical_not(valid_rows)))[0]))
            raise RuntimeError("Unable to form factory even with slack.")
        old_valid_rows = valid_rows
        dual = dual[:-1]
        
        new_vector_table = construct.vectors(cost_function, inverse_priced_indices, dual, known_technologies).shadow_attachment(extras).reduced.sorted
        new_vector_table = new_vector_table.mask(linear_transform_is_gt(new_vector_table.vector.T, dual, .99 * new_vector_table.cost))

        new_mask = true_new_column_mask(vector_table.ident, new_vector_table.ident)
        true_news = np.where(new_mask)[0]
        if len(true_news)==0:
            break

        new_vector_table.mask(true_news)
        logging.debug("New columns found are: "+"\n\t".join([repr((new_vector_table.ident)[i])+" with column "+str(sparse.coo_matrix((new_vector_table.vector)[:, i]))+" \n\tand cost: "+str((new_vector_table.cost)[i]) for i in range(new_vector_table.ident.shape[0])]))

        vector_table.mask(linear_transform_is_gt(vector_table.vector.T, dual, .99 * vector_table.cost))
        vector_table = (vector_table + new_vector_table).sorted

    if np.isclose(dual, 0).all():
        dual = prio_dual

    return primal, dual, vector_table

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


