from __future__ import annotations

from globalsandimports import *
from utils import *
from transportation import *

if TYPE_CHECKING:
    from lookuptables import CompiledConstruct
    from constructs import PointEvaluations
    from tools import FactorioInstance

# have classes the compile into other classes to do this more efficiently

@runtime_checkable
class CompiledCostFunction(Protocol):
    """Class for typing of all cost functions after getting a pricing vector, construct, and transport costs.
    """    
    def __call__(self, point_evaluations: PointEvaluations) -> np.ndarray:
        raise NotImplementedError
    
@runtime_checkable
class PricedCostFunction(Protocol):
    """Class for typing of all cost functions after getting a pricing vector.
    """    
    def __call__(self, construct: CompiledConstruct, transport_cost: TransportCost) -> CompiledCostFunction:
        raise NotImplementedError

@runtime_checkable
class CostFunction(Protocol):
    """Class for typing of all cost functions before getting anything.
    """    
    def __call__(self, pricing_vector: np.ndarray) -> PricedCostFunction:
        raise NotImplementedError
    

def empty_cost_function(pricing_vector: np.ndarray) -> PricedCostFunction:
    """Cost function that always returns a correctly shaped zero array

    Parameters
    ----------
    pricing_vector : np.ndarray
        Pricing vector to use for this cost function (ignored here)

    Returns
    -------
    PricedCostFunction
        Zero priced cost function
    """    
    def empty_priced_function(construct: CompiledConstruct, transport_cost: TransportCost) -> CompiledCostFunction:
        """Priced cost function that always returns a correctly shaped zero array

        Parameters
        ----------
        construct : CompiledConstruct
            Construct being priced (ignored here)
        transport_cost : TransportCost
            Transport costs to use (ignored here)

        Returns
        -------
        CompiledCostFunction
            Zero compiled cost function
        """        
        def empty_compiled_function(point_evaluations: PointEvaluations) -> np.ndarray:
            """Compiled cost function that always returns a correctly shaped zero array

            Parameters
            ----------
            point_evaluations : PointEvaluations
                PointEvaluations for the points to price (used to determine proper zero array shape)

            Returns
            -------
            np.ndarray
                Properly shaped array of zeros
            """            
            return np.zeros(point_evaluations.evaulated_cost.shape[0])
        return empty_compiled_function
    return empty_priced_function

def true_cost_function(construct: CompiledConstruct, transport_cost: TransportCost, point_evaluations: PointEvaluations) -> np.ndarray:
    """Calcualtes the true item cost of a setup. Eventually used to decide what previous factories must build.
    Only accepts 1 point at the moment. TODO

    Parameters
    ----------
    construct : CompiledConstruct
        Construct being priced
    transport_cost : TransportCost
        Transport costs to use
    point_evaluations : PointEvaluations
        PointEvaluations for the points to price

    Returns
    -------
    np.ndarray
        True cost vector for the construct evaluated at the point
    """    
    return (point_evaluations.evaulated_cost + construct.base_cost + transport_cost.static_cost @ construct.flow_characterization + \
            point_evaluations.multilinear_effect @ construct.flow_transform @ transport_cost.scaling_cost).reshape(-1, 1)

def standard_cost_function(pricing_vector: np.ndarray) -> PricedCostFunction:
    """Cost function using the previous factories pricing model

    Parameters
    ----------
    pricing_vector : np.ndarray
        Previous factories pricing model

    Returns
    -------
    PricedCostFunction
        The half compiled cost function
    """    
    def standard_priced_function(construct: CompiledConstruct, transport_cost: TransportCost) -> CompiledCostFunction:
        """Priced cost function using the previous factories pricing model

        Parameters
        ----------
        construct : CompiledConstruct
            Construct being priced
        transport_cost : TransportCost
            Transport costs to use

        Returns
        -------
        CompiledCostFunction
            The fully compiled cost function
        """        
        static_cost = np.dot(transport_cost.static_cost @ construct.flow_characterization, pricing_vector) + np.dot(construct.base_cost, pricing_vector)
        scaling_cost = construct.flow_transform @ transport_cost.scaling_cost @ pricing_vector
        def standard_compiled_function(point_evaluations: PointEvaluations) -> np.ndarray:
            """Compiled cost function using the previous factories pricing model

            Parameters
            ----------
            point_evaluations : PointEvaluations
                PointEvaluations for the points to price

            Returns
            -------
            np.ndarray
                Point costs
            """            
            out = point_evaluations.evaulated_cost @ pricing_vector + static_cost + point_evaluations.multilinear_effect @ scaling_cost
            if not isinstance(out, np.ndarray):
                return np.array([out])
            return out.reshape(-1)
        return standard_compiled_function
    return standard_priced_function

def space_cost_function(pricing_vector: np.ndarray) -> PricedCostFunction:
    """Cost function using the space the factory takes

    Parameters
    ----------
    pricing_vector : np.ndarray
        Previous factories pricing model (ignored here)

    Returns
    -------
    PricedCostFunction
        The half compiled cost function
    """    
    def space_priced_function(construct: CompiledConstruct, transport_cost: TransportCost) -> CompiledCostFunction:
        """Priced cost function using the space the factory takes

        Parameters
        ----------
        construct : CompiledConstruct
            Construct being priced
        transport_cost : TransportCost
            Transport costs to use (not used yet TODO)

        Returns
        -------
        CompiledCostFunction
            The fully compiled cost function
        """        
        def space_compiled_function(point_evaluations: PointEvaluations) -> np.ndarray:
            """Compiled cost function using the space the factory takes

            Parameters
            ----------
            point_evaluations : PointEvaluations
                PointEvaluations for the points to price

            Returns
            -------
            np.ndarray
                Point costs
            """            
            out = point_evaluations.effective_area + construct.effective_area
            if not isinstance(out, np.ndarray):
                return np.array([out])
            return out.reshape(-1)
        return space_compiled_function
    return space_priced_function

def spatial_cost_function(pricing_vector: np.ndarray) -> PricedCostFunction:
    """Cost function using the space the factory takes on ore patches

    Parameters
    ----------
    pricing_vector : np.ndarray
        Previous factories pricing model (ignored here)

    Returns
    -------
    PricedCostFunction
        The half compiled cost function
    """    
    space_func: PricedCostFunction = space_cost_function(pricing_vector)
    def spatial_priced_function(construct: CompiledConstruct, transport_cost: TransportCost) -> CompiledCostFunction:
        """Priced cost function using the space the factory takes on ore patches

        Parameters
        ----------
        construct : CompiledConstruct
            Construct being priced
        transport_cost : TransportCost
            Transport costs to use (not used yet TODO)

        Returns
        -------
        CompiledCostFunction
            The fully compiled cost function
        """        
        if construct._isa_mining_drill:
            return space_func(construct, transport_cost)
        else:
            def spatial_empty_compiled_function(point_evaluations: PointEvaluations) -> np.ndarray:
                """Compiled cost function using the space the factory takes on ore patches for constructs that don't need a ore patch

                Parameters
                ----------
                point_evaluations : PointEvaluations
                    PointEvaluations for the points to price

                Returns
                -------
                np.ndarray
                    Properly shaped zero array
                """            
                return np.zeros(point_evaluations.evaulated_cost.shape[0])
            return spatial_empty_compiled_function
    return spatial_priced_function

def deproductivity_effect_transform(effect_transform: sparse.csr_matrix) -> sparse.csr_matrix:
    """Removes the productivity multiplier from a effect transform.
    Used for calculating ore taken from the ground

    Parameters
    ----------
    effect_transform : sparse.csr_matrix
        Effect transform to remove productivity from

    Returns
    -------
    sparse.csr_matrix
        De-productivitied effect transform
    """    
    prod_index = ACTIVE_MODULE_EFFECTS.index('productivity')
    new_effect_transform = sparse.lil_matrix((len(MODULE_EFFECT_ORDERING), effect_transform.shape[1]))
    for i, j in zip(*new_effect_transform.nonzero()):
        neffects = {e for e in MODULE_EFFECT_ORDERING[i] if e != prod_index}
        for ni in range(len(MODULE_EFFECT_ORDERING)):
            if MODULE_EFFECT_ORDERING[ni]==neffects:
                new_effect_transform[ni, j] = effect_transform[i, j]
                break
    return sparse.csr_matrix(new_effect_transform)

def ore_cost_function(pricing_vector: np.ndarray) -> PricedCostFunction:
    """Cost function using the ore mined from the ground

    Parameters
    ----------
    pricing_vector : np.ndarray
        Previous factories pricing model (ignored here)

    Returns
    -------
    PricedCostFunction
        The half compiled cost function
    """    
    def ore_priced_function(construct: CompiledConstruct, transport_cost: TransportCost) -> CompiledCostFunction:
        """Priced cost function using the ore mined from the ground

        Parameters
        ----------
        construct : CompiledConstruct
            Construct being priced
        transport_cost : TransportCost
            Transport costs to use (not used yet TODO)

        Returns
        -------
        CompiledCostFunction
            The fully compiled cost function
        """        
        if construct._isa_mining_drill:
            effect_transform_positive = construct.effect_transform.copy()
            effect_transform_positive[effect_transform_positive < 0] = 0
            effect_transform_positive = deproductivity_effect_transform(effect_transform_positive)
            effect_vector = effect_transform_positive @ construct._instance.raw_ore_pricing
            def ore_compiled_function(point_evaluations: PointEvaluations) -> np.ndarray:
                """Compiled cost function using the ore mined from the ground

                Parameters
                ----------
                point_evaluations : PointEvaluations
                    PointEvaluations for the points to price

                Returns
                -------
                np.ndarray
                    Point costs
                """            
                out = point_evaluations.multilinear_effect @ effect_vector
                if not isinstance(out, np.ndarray):
                    return np.array([out])
                return out
            return ore_compiled_function
        else:
            def ore_empty_compiled_function(point_evaluations: PointEvaluations) -> np.ndarray:
                """Compiled cost function using the ore mined from the ground for constructs that don't need a ore patch

                Parameters
                ----------
                point_evaluations : PointEvaluations
                    PointEvaluations for the points to price

                Returns
                -------
                np.ndarray
                    Properly shaped zero array
                """            
                return np.zeros(point_evaluations.evaulated_cost.shape[0])
            return ore_empty_compiled_function
    return ore_priced_function

def multiply_cost_function(func: CostFunction, multiplier: Real) -> CostFunction:
    """Multiplies cost function by a scalar

    Parameters
    ----------
    func : CostFunction
        Base cost function to scale
    multiplier : Real
        Scalar

    Returns
    -------
    CostFunction
        Resulting cost function
    """    
    def multiplied_cost_function(pricing_vector: np.ndarray) -> PricedCostFunction:
        func_priced: PricedCostFunction = func(pricing_vector)
        def multiplied_priced_function(construct: CompiledConstruct, transport_cost: TransportCost) -> CompiledCostFunction:
            func_compiled: CompiledCostFunction = func_priced(construct, transport_cost)
            def multiplied_compiled_function(point_evaluations: PointEvaluations) -> np.ndarray:
                return func_compiled(point_evaluations) * multiplier
            return multiplied_compiled_function
        return multiplied_priced_function
    return multiplied_cost_function

def add_cost_functions(funcA: CostFunction, funcB: CostFunction) -> CostFunction:
    """Adds two cost functions together

    Parameters
    ----------
    funcA : CostFunction
        A cost function
    funcB : CostFunction
        Another cost function

    Returns
    -------
    CostFunction
        Resulting cost function that adds the two cost function's outputs
    """    
    def added_cost_function(pricing_vector: np.ndarray) -> PricedCostFunction:
        pricedA: PricedCostFunction = funcA(pricing_vector)
        pricedB: PricedCostFunction = funcB(pricing_vector)
        def added_priced_function(construct: CompiledConstruct, transport_cost: TransportCost) -> CompiledCostFunction:
            compiledA: CompiledCostFunction = pricedA(construct, transport_cost)
            compiledB: CompiledCostFunction = pricedB(construct, transport_cost)
            def added_compiled_function(point_evaluations: PointEvaluations) -> np.ndarray:
                return compiledA(point_evaluations) + compiledB(point_evaluations)
            return added_compiled_function
        return added_priced_function
    return added_cost_function

def hybrid_cost_function(input: dict[str, Real], instance: FactorioInstance) -> CostFunction:
    """Creates a combination cost function based on input weightings.
    'standard', 'basic', 'simple', 'baseline', 'dual' specify the standard cost method (last factory)
    'spatial', 'ore space', 'tiles', 'mining', 'mining tiles', 'resource space' specify the spatial cost method
    'ore', 'ore count', 'raw', 'raw resource', 'resources', 'resource count' specify the ore used cost method

    Parameters
    ----------
    input : dict[str, Real]
        Hybrid model definition
    instance : FactorioInstance
        Factorio instance to use

    Returns
    -------
    Callable[[np.ndarray, CompiledConstruct, np.ndarray, TechnologicalLimitation], np.ndarray]
        Uncompiled cost function
    """    
    func: CostFunction = empty_cost_function

    for phrase in ['standard', 'basic', 'simple', 'baseline', 'dual']:
        if phrase in input.keys():
            func = add_cost_functions(func, multiply_cost_function(standard_cost_function, input[phrase]))
            break

    for phrase in ['spatial', 'ore space', 'tiles', 'mining', 'mining tiles', 'resource space']:
        if phrase in input.keys():
            func = add_cost_functions(func, multiply_cost_function(spatial_cost_function, input[phrase]))
            break
    
    for phrase in ['ore', 'ore count', 'raw', 'raw resource', 'resources', 'resource count']:
        if phrase in input.keys():
            func = add_cost_functions(func, multiply_cost_function(ore_cost_function, input[phrase]))
            break
    
    return func


