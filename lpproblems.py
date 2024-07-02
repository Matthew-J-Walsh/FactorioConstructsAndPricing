from globalsandimports import *

from lookuptables import *
from utils import *
from lpsolvers import *

def solve_factory_optimization_problem(construct: ComplexConstruct, u_j: np.ndarray, cost_function: CompiledCostFunction, inverse_priced_indices: np.ndarray, known_technologies: TechnologicalLimitation, 
                                       starting_columns: ColumnTable | None = None) -> tuple[np.ndarray, np.ndarray, ColumnTable]:
    """Solve an optimization problem given a ComplexConstruct, a target output vector, and a cost vector

    Parameters
    ----------
    construct : ComplexConstruct
        Construct being optimized, usually a ComplexConstruct of the entire factory
    u_j : np.ndarray
        Target output vector
    cost_function : CompiledCostFunction
        Cost function to use
    inverse_priced_indices : np.ndarray
        What indices of the pricing vector aren't priced
    known_technologies : TechnologicalLimitation
        Current tech level
    starting_columns : ColumnTable | None, optional
        What columns to start with

    Returns
    -------
    tuple[np.ndarray, np.ndarray, ColumnTable]
        The optimal factory vector,
        The optimal pricing model for the factory
        The columns of the optimal factory

    Raises
    ------
    RuntimeError
        Unable to form a factory for whatever reason
    """
    if starting_columns is None:
        vector_table = construct.vectors(cost_function, inverse_priced_indices, None, known_technologies).reduced.sorted
    else:
        vector_table = starting_columns

    i = 0
    old_valid_rows = vector_table.valid_rows
    while True:
        valid_rows = vector_table.valid_rows
        assert (u_j[np.logical_not(valid_rows)]<=0).all(), np.where(np.logical_and(u_j>0, np.logical_not(valid_rows)))[0]#np.where(np.logical_and(old_outputing_rows, np.logical_not(outputing_rows)))[0]

        logging.info("Starting iteration "+str(i)+" of current optimization with "+str(len(vector_table.ident))+" columns.")
        i += 1
        if i>100:
            raise RuntimeError()
        primal, dual = BEST_LP_SOLVER(vector_table.vector, u_j, vector_table.cost)
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
        
        true_news = true_new_column_mask(vector_table.vector, vector_table.cost, new_vector_table.vector, new_vector_table.cost)
        if len(true_news)==0:
            break

        new_vector_table.mask(true_news)
        logging.debug("New columns found are: "+"\n\t".join([repr((new_vector_table.ident)[i])+" with column "+str(sparse.coo_matrix((new_vector_table.vector)[:, i]))+" \n\tand cost: "+str((new_vector_table.cost)[i]) for i in range(new_vector_table.ident.shape[0])]))
        
        vector_table.mask(linear_transform_is_gt(vector_table.vector.T, dual, .99 * vector_table.cost))
        vector_table = (vector_table + new_vector_table).sorted

    return primal, dual, vector_table

def solve_manual_factory_optimization_problem(construct: ComplexConstruct, u_j: np.ndarray, cost_function: CompiledCostFunction, inverse_priced_indices: np.ndarray, known_technologies: TechnologicalLimitation, 
                                              manual_constructs: ColumnTable, starting_columns: ColumnTable | None = None) -> tuple[np.ndarray, np.ndarray, ColumnTable]:
    """Solve an optimization problem given a ComplexConstruct, a target output vector, and a cost vector.
    With extra columns that shouldn't be filtered and have max priority cost

    Parameters
    ----------
    construct : ComplexConstruct
        Construct being optimized, usually a ComplexConstruct of the entire factory
    u_j : np.ndarray
        Target output vector
    cost_function : CompiledCostFunction
        Cost function to use
    inverse_priced_indices : np.ndarray
        What indices of the pricing vector aren't priced
    known_technologies : TechnologicalLimitation
        Current tech level
    manual_constructs : ColumnTable
        Columns of the manual constructs
    starting_columns : ColumnTable | None, optional
        What columns to start with

    Returns
    -------
    tuple[np.ndarray, np.ndarray, ColumnTable]
        The optimal factory vector,
        The optimal pricing model for the factory
        The columns of the optimal factory

    Raises
    ------
    RuntimeError
        Unable to form a factory for whatever reason
    """
    if starting_columns is None:
        vector_table = construct.vectors(cost_function, inverse_priced_indices, None, known_technologies).shadow_attachment(manual_constructs).reduced.sorted
    else:
        vector_table = starting_columns

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
        
        new_vector_table = construct.vectors(cost_function, inverse_priced_indices, dual, known_technologies).shadow_attachment(manual_constructs).reduced.sorted
        new_vector_table = new_vector_table.mask(linear_transform_is_gt(new_vector_table.vector.T, dual, .99 * new_vector_table.cost))

        true_news = true_new_column_mask(vector_table.vector, vector_table.cost, new_vector_table.vector, new_vector_table.cost)
        if len(true_news)==0:
            break

        new_vector_table.mask(true_news)
        logging.debug("New columns found are: "+"\n\t".join([repr((new_vector_table.ident)[i])+" with column "+str(sparse.coo_matrix((new_vector_table.vector)[:, i]))+" \n\tand cost: "+str((new_vector_table.cost)[i]) for i in range(new_vector_table.ident.shape[0])]))

        vector_table.mask(linear_transform_is_gt(vector_table.vector.T, dual, .99 * vector_table.cost))
        vector_table = (vector_table + new_vector_table).sorted

    if np.isclose(dual, 0).all():
        dual = prio_dual

    return primal, dual, vector_table

def true_new_column_mask(columns: np.ndarray, costs: np.ndarray, new_columns: np.ndarray, new_costs: np.ndarray) -> np.ndarray:
    """Determines what columns of the new columns are not present in the reference columns, considers costs

    Parameters
    ----------
    columns : np.ndarray
        Reference columns
    costs : np.ndarray
        Reference costs
    new_columns : np.ndarray
        New columns
    new_costs : np.ndarray
        New costs

    Returns
    -------
    np.ndarray
        Column indicies of the new columns that are actually new
    """    
    c_hash = columns.sum(axis=0)
    n_hash = new_columns.sum(axis=0)

    mask = np.full(new_columns.shape[1], True, dtype=bool)

    for ci, ni in iterate_over_equals(c_hash, n_hash):
        if (columns[:, ci]==new_columns[:, ni]).all() and costs[ci]==new_costs[ni]:
            mask[ni] = False
    
    return np.where(mask)[0]

def iterate_over_equals(a1: np.ndarray, a2: np.ndarray) -> Iterator[tuple[int, int]]:
    """Creates an iterator for where elements of a1 equal elements of a2

    Parameters
    ----------
    a1 : np.ndarray
        First array
    a2 : np.ndarray
        Second array

    Yields
    ------
    Iterator[tuple[int, int]]
        Iterator of indicies of the first and second array.
    """    
    for i, v1 in enumerate(a1):
        for j in np.where(a2 == v1)[0]:
            yield (i, j)