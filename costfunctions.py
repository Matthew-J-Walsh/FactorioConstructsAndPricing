from globalsandimports import *
from utils import *
import lookuptables as luts


def standard_cost_function(pricing_vector: np.ndarray, construct: luts.CompiledConstruct, lookup_indicies: np.ndarray, known_technologies: TechnologicalLimitation) -> np.ndarray:
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
    out = construct.lookup_table(known_technologies).cost_transform[lookup_indicies, :] @ pricing_vector + np.dot(construct.base_cost_vector, pricing_vector)
    if not isinstance(out, np.ndarray):
        return np.array([out])
    return out

def spatial_cost_function(pricing_vector: np.ndarray, construct: luts.CompiledConstruct, lookup_indicies: np.ndarray, known_technologies: TechnologicalLimitation) -> np.ndarray:
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
        return standard_cost_function(pricing_vector, construct, lookup_indicies, known_technologies)
    else:
        return np.zeros(lookup_indicies.shape[0])

def ore_cost_function(pricing_vector: np.ndarray, construct: luts.CompiledConstruct, lookup_indicies: np.ndarray, known_technologies: TechnologicalLimitation) -> np.ndarray:
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
    out = construct.lookup_table(known_technologies).multilinear_effect_transform[lookup_indicies, :] @ effect_vector
    if not isinstance(out, np.ndarray):
        return np.array([out])
    return out

def space_cost_function(pricing_vector: np.ndarray, construct: luts.CompiledConstruct, lookup_indicies: np.ndarray, known_technologies: TechnologicalLimitation) -> np.ndarray:
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
    out = construct.lookup_table(known_technologies).effective_area_table[lookup_indicies] + construct.effective_area
    if not isinstance(out, np.ndarray):
        return np.array([out])
    return out

def throughput_cost_function(pricing_vector: np.ndarray, construct: luts.CompiledConstruct, lookup_indicies: np.ndarray) -> np.ndarray:
    """TODO: Based on item throughput rate in trains?"""
    raise NotImplementedError()

def hybrid_cost_function(input: dict[str, Real], instance) -> Callable[[np.ndarray, luts.CompiledConstruct, np.ndarray, TechnologicalLimitation], np.ndarray]:
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
    func = lambda pricing_vector, construct, lookup_indicies, known_technologies: np.zeros(lookup_indicies.shape[0])

    for phrase in ['standard', 'basic', 'simple', 'baseline', 'dual']:
        if phrase in input.keys():
            func = lambda pricing_vector, construct, lookup_indicies, known_technologies: \
                func(pricing_vector, construct, lookup_indicies, known_technologies) + \
                    input[phrase] * standard_cost_function(pricing_vector, construct, lookup_indicies, known_technologies)
            break

    for phrase in ['spatial', 'ore space', 'tiles', 'mining', 'mining tiles', 'resource space']:
        if phrase in input.keys():
            func = lambda pricing_vector, construct, lookup_indicies, known_technologies: \
                func(pricing_vector, construct, lookup_indicies, known_technologies) + input[phrase] * spatial_cost_function(instance.spatial_pricing, construct, lookup_indicies, known_technologies)
            break
    
    for phrase in ['ore', 'ore count', 'raw', 'raw resource', 'resources', 'resource count']:
        if phrase in input.keys():
            func = lambda pricing_vector, construct, lookup_indicies, known_technologies: \
                func(pricing_vector, construct, lookup_indicies, known_technologies) + \
                    input[phrase] * ore_cost_function(instance.raw_ore_pricing, construct, lookup_indicies, known_technologies)
            break
    
    return func


