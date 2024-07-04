from __future__ import annotations

from globalsandimports import *
from utils import *

if TYPE_CHECKING:
    import lookuptables as luts
    import constructs as cstcts
    import tools as tools

class CostFunction(Protocol):
    """Class for typing of all cost functions before getting a pricing vector.
    """    
    def __call__(self, pricing_vector: np.ndarray, construct: luts.CompiledConstruct, point_evaluations: cstcts.PointEvaluations) -> np.ndarray:
        raise NotImplementedError
    
class CompiledCostFunction(Protocol):
    """Class for typing of all cost functions after getting a pricing vector.
    """    
    def __call__(self, construct: luts.CompiledConstruct, point_evaluations: cstcts.PointEvaluations) -> np.ndarray:
        raise NotImplementedError

def standard_cost_function(pricing_vector: np.ndarray, construct: luts.CompiledConstruct, point_evaluations: cstcts.PointEvaluations) -> np.ndarray:
    """Cost function based on previous dual vector

    Parameters
    ----------
    pricing_vector : np.ndarray
        Previous dual vector
    construct : luts.CompiledConstruct
        Construct being priced
    lookup_indicies : np.ndarray
        Indicies in lookup table to calculate for
    known_technologies : TechnologicalLimitation
        Current tech level to calculate for 

    Returns
    -------
    np.ndarray
        Cost array
    """    
    out = point_evaluations.beacon_cost @ pricing_vector + np.dot(construct.base_cost, pricing_vector)
    if not isinstance(out, np.ndarray):
        return np.array([out])
    return out

def spatial_cost_function(pricing_vector: np.ndarray, construct: luts.CompiledConstruct, point_evaluations: cstcts.PointEvaluations) -> np.ndarray:
    """Cost function based on ore tiles used

    Parameters
    ----------
    pricing_vector : np.ndarray
        Cost imposed for space usage
    construct : luts.CompiledConstruct
        Construct being priced
    lookup_indicies : np.ndarray
        Indicies in lookup table to calculate for
    known_technologies : TechnologicalLimitation
        Current tech level to calculate for 

    Returns
    -------
    np.ndarray
        Cost array
    """    
    if construct.isa_mining_drill:
        return standard_cost_function(pricing_vector, construct, point_evaluations)
    else:
        #print(point_evaluations.beacon_cost.shape)
        return np.zeros(point_evaluations.beacon_cost.shape[0])

def ore_cost_function(pricing_vector: np.ndarray, construct: luts.CompiledConstruct, point_evaluations: cstcts.PointEvaluations) -> np.ndarray:
    """Cost function based on ore produced (not mined).
    Doesn't account for productivity

    Parameters
    ----------
    pricing_vector : np.ndarray
        Cost imposed for positive throughput
    construct : luts.CompiledConstruct
        Construct being priced
    lookup_indicies : np.ndarray
        Indicies in lookup table to calculate for
    known_technologies : TechnologicalLimitation
        Current tech level to calculate for 

    Returns
    -------
    np.ndarray
        Cost array
    """
    effect_transform_positive = construct.effect_transform.copy()
    effect_transform_positive[effect_transform_positive < 0] = 0
    effect_vector = effect_transform_positive @ pricing_vector
    out = point_evaluations.multilinear_effect @ effect_vector
    if not isinstance(out, np.ndarray):
        return np.array([out])
    return out

def space_cost_function(pricing_vector: np.ndarray, construct: luts.CompiledConstruct, point_evaluations: cstcts.PointEvaluations) -> np.ndarray:
    """Cost function based on tiles taken up. Useful for space platforms

    Parameters
    ----------
    pricing_vector : np.ndarray
        Cost imposed for space usage
    construct : luts.CompiledConstruct
        Construct being priced
    lookup_indicies : np.ndarray
        Indicies in lookup table to calculate for
    known_technologies : TechnologicalLimitation
        Current tech level to calculate for 

    Returns
    -------
    np.ndarray
        Cost array
    """    
    out = point_evaluations.effective_area + construct.effective_area
    if not isinstance(out, np.ndarray):
        return np.array([out])
    return out

def throughput_cost_function(pricing_vector: np.ndarray, construct: luts.CompiledConstruct, point_evaluations: cstcts.PointEvaluations) -> np.ndarray:
    """TODO: Based on item throughput rate in trains?"""
    raise NotImplementedError()

def hybrid_cost_function(input: dict[str, Real], instance: tools.FactorioInstance) -> CostFunction:
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
    Callable[[np.ndarray, luts.CompiledConstruct, np.ndarray, TechnologicalLimitation], np.ndarray]
        Uncompiled cost function
    """    
    func = lambda pricing_vector, construct, point_evaluations: np.zeros(point_evaluations.beacon_cost.shape[0])

    for phrase in ['standard', 'basic', 'simple', 'baseline', 'dual']:
        if phrase in input.keys():
            func = lambda pricing_vector, construct, point_evaluations: \
                func(pricing_vector, construct, point_evaluations) + \
                    input[phrase] * standard_cost_function(pricing_vector, construct, point_evaluations)
            break

    for phrase in ['spatial', 'ore space', 'tiles', 'mining', 'mining tiles', 'resource space']:
        if phrase in input.keys():
            func = lambda pricing_vector, construct, point_evaluations: \
                func(pricing_vector, construct, point_evaluations) + input[phrase] * spatial_cost_function(instance.spatial_pricing, construct, point_evaluations)
            break
    
    for phrase in ['ore', 'ore count', 'raw', 'raw resource', 'resources', 'resource count']:
        if phrase in input.keys():
            func = lambda pricing_vector, construct, point_evaluations: \
                func(pricing_vector, construct, point_evaluations) + \
                    input[phrase] * ore_cost_function(instance.raw_ore_pricing, construct, point_evaluations)
            break
    
    return func


