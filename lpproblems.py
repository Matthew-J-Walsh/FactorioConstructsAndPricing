from globalsandimports import *

from lookuptables import *
from constructs import *
from utils import *
from lpsolvers import *
from transportation import *

def solve_factory_optimization_problem(construct: ComplexConstruct, u_j: np.ndarray, cost_function: PricedCostFunction, inverse_priced_indices: np.ndarray, transport_cost_table: dict[str, TransportationCompiler], 
                                       known_technologies: TechnologicalLimitation, starting_columns: ColumnTable | None = None) -> tuple[np.ndarray, np.ndarray, ColumnTable]:
    """Solve an optimization problem given a ComplexConstruct, a target output vector, and a cost vector

    Parameters
    ----------
    construct : ComplexConstruct
        Construct being optimized, usually a ComplexConstruct of the entire factory
    u_j : np.ndarray
        Target output vector
    cost_function : PricedCostFunction
        Cost function to use
    inverse_priced_indices : np.ndarray
        What indices of the pricing vector aren't priced
    transport_cost_table : dict[str, TransportCostPair]
        Table of transportation cost matricies applied based on complex construct transport types
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
        column_table = construct.columns(ColumnSpecifier(cost_function, inverse_priced_indices, None, TransportCost.empty(inverse_priced_indices.shape[0]),
                                                         transport_cost_table, known_technologies)).reduced.sorted
    else:
        column_table = starting_columns

    i = 0
    old_valid_rows = column_table.valid_rows
    while True:
        valid_rows = column_table.valid_rows
        assert (u_j[np.logical_not(valid_rows)]<=0).all(), np.where(np.logical_and(u_j>0, np.logical_not(valid_rows)))[0]#np.where(np.logical_and(old_outputing_rows, np.logical_not(outputing_rows)))[0]

        logging.info("Starting iteration "+str(i)+" of current optimization with "+str(len(column_table.idents))+" columns.")
        i += 1
        if i>100:
            raise RuntimeError()
        primal, dual = BEST_LP_SOLVER(column_table.columns, u_j, column_table.costs)
        if primal is None or dual is None:
            logging.error("Row losses: "+str(np.where(np.logical_and(old_valid_rows, np.logical_not(valid_rows)))[0]))
            raise RuntimeError("Unable to form factory even with slack.")
        old_valid_rows = valid_rows

        assert not np.isclose(column_table.costs, 0).all()
        assert not np.isclose(u_j, 0).all()
        assert not np.isclose(np.dot(u_j, dual), 0)
        assert not np.isclose(np.dot(column_table.costs, primal), 0)
        assert not np.isclose(dual, 0).all()
        
        new_column_table = construct.columns(ColumnSpecifier(cost_function, inverse_priced_indices, dual, TransportCost.empty(inverse_priced_indices.shape[0]),
                                                             transport_cost_table, known_technologies)).reduced.sorted
        new_column_table.mask(linear_transform_is_gt(new_column_table.columns.T, dual, .99 * new_column_table.costs))
        
        true_news = true_new_column_mask(column_table.columns, column_table.costs, new_column_table.columns, new_column_table.costs)
        if len(true_news)==0:
            break

        new_column_table.mask(true_news)
        logging.debug("New columns found are: "+"\n\t".join([repr((new_column_table.idents)[i])+" with column "+str(sparse.coo_matrix((new_column_table.columns)[:, i]))+" \n\tand cost: "+str((new_column_table.costs)[i]) for i in range(new_column_table.idents.shape[0])]))
        
        column_table.mask(linear_transform_is_gt(column_table.columns.T, dual, .99 * column_table.costs))
        column_table = (column_table + new_column_table).sorted

    return primal, dual, column_table

def solve_manual_factory_optimization_problem(construct: ComplexConstruct, u_j: np.ndarray, cost_function: PricedCostFunction, inverse_priced_indices: np.ndarray, transport_cost_table: dict[str, TransportationCompiler],
                                              known_technologies: TechnologicalLimitation, manual_constructs: ColumnTable, starting_columns: ColumnTable | None = None) -> tuple[np.ndarray, np.ndarray, ColumnTable]:
    """Solve an optimization problem given a ComplexConstruct, a target output vector, and a cost vector.
    With extra columns that shouldn't be filtered and have max priority cost

    Parameters
    ----------
    construct : ComplexConstruct
        Construct being optimized, usually a ComplexConstruct of the entire factory
    u_j : np.ndarray
        Target output vector
    cost_function : PricedCostFunction
        Cost function to use
    inverse_priced_indices : np.ndarray
        What indices of the pricing vector aren't priced
    transport_cost_table : dict[str, TransportCostPair]
        Table of transportation cost matricies applied based on complex construct transport types
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
        column_table = construct.columns(ColumnSpecifier(cost_function, inverse_priced_indices, None, TransportCost.empty(inverse_priced_indices.shape[0]),
                                                         transport_cost_table, known_technologies)).shadow_attachment(manual_constructs).reduced.sorted
    else:
        column_table = starting_columns

    prio_primal, prio_dual = BEST_LP_SOLVER(column_table.columns, u_j, column_table.true_costs[-1, :])
    if prio_primal is None or prio_dual is None:
        raise RuntimeError("Unable to form factory even with slack.")
    prio_cost = np.dot(prio_primal, column_table.true_costs[-1, :])

    i = 0
    old_valid_rows = column_table.valid_rows
    while True:
        valid_rows = column_table.valid_rows
        assert (u_j[np.logical_not(valid_rows)]<=0).all(), np.where(np.logical_and(u_j>0, np.logical_not(valid_rows)))[0]#np.where(np.logical_and(old_outputing_rows, np.logical_not(outputing_rows)))[0]

        logging.info("Starting iteration "+str(i)+" of current optimization with "+str(len(column_table.idents))+" columns.")
        i += 1
        
        primal, dual = BEST_LP_SOLVER(np.concatenate((column_table.columns, column_table.true_costs[-1, :].reshape(1, -1)), axis=0), 
                                                                                  np.concatenate((u_j, np.array([prio_cost]))), 
                                                                                  column_table.costs)
        if primal is None or dual is None:
            logging.error("Row losses: "+str(np.where(np.logical_and(old_valid_rows, np.logical_not(valid_rows)))[0]))
            raise RuntimeError("Unable to form factory even with slack.")
        old_valid_rows = valid_rows
        dual = dual[:-1]
        
        new_column_table = construct.columns(ColumnSpecifier(cost_function, inverse_priced_indices, dual, TransportCost.empty(inverse_priced_indices.shape[0]),
                                                             transport_cost_table, known_technologies)).shadow_attachment(manual_constructs).reduced.sorted
        new_column_table = new_column_table.mask(linear_transform_is_gt(new_column_table.columns.T, dual, .99 * new_column_table.costs))

        true_news = true_new_column_mask(column_table.columns, column_table.costs, new_column_table.columns, new_column_table.costs)
        if len(true_news)==0:
            break

        new_column_table.mask(true_news)
        logging.debug("New columns found are: "+"\n\t".join([repr((new_column_table.idents)[i])+" with column "+str(sparse.coo_matrix((new_column_table.columns)[:, i]))+" \n\tand cost: "+str((new_column_table.costs)[i]) for i in range(new_column_table.idents.shape[0])]))

        column_table.mask(linear_transform_is_gt(column_table.columns.T, dual, .99 * column_table.costs))
        column_table = (column_table + new_column_table).sorted

    if np.isclose(dual, 0).all():
        dual = prio_dual

    return primal, dual, column_table

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