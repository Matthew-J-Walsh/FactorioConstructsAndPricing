from __future__ import annotations

from globalsandimports import *

SOLID_REFERENCE_VALUE: int = 0
FLUID_REFERENCE_VALUE: int = 1
ELECTRIC_REFERENCE_VALUE: int = 2
HEAT_REFERENCE_VALUE: int = 2
RESEARCH_REFERENCE_VALUE: int = 3
UNKNOWN_REFERENCE_VALUE: int = -1

class TransportPair(NamedTuple):
    densities: np.ndarray
    options: np.ndarray

class TransportTable(NamedTuple):
    solid: TransportPair
    liquid: TransportPair
    electric: TransportPair
    heat: TransportPair

class TransportCostPair(NamedTuple):
    static: np.ndarray
    scaling: np.ndarray

class TransportCostTable(NamedTuple):
    solid: TransportCostPair
    liquid: TransportCostPair
    electric: TransportCostPair
    heat: TransportCostPair


class TransportationMethod(Protocol):
    """Transporation methods cost function typing

    Parameters
    ----------
    pricing_vector : np.ndarray
        The pricing vector in use

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ref_list vector that adds into the cost vector to calculate an additional cost
        ref_list by ref_list matrix that multiplies into the effect vector to calculate an additional cost
    """    
    def __call__(self, pricing_vector: np.ndarray, inverse_priced_indices: np.ndarray, transportation_table: TransportTable) -> TransportCostTable:     
        raise NotImplementedError

def classify_reference_list(reference_list: Sequence[str], data: dict) -> np.ndarray:
    """Makes a classification array for the reference_list values

    Parameters
    ----------
    reference_list : Sequence[str]
        Reference list
    data : dict
        data.raw

    Returns
    -------
    np.ndarray
        Classification array
    """    
    classification_array = np.full(len(reference_list), UNKNOWN_REFERENCE_VALUE, dtype=int)
    for i in range(len(reference_list)):
        for cata in ITEM_SUB_PROTOTYPES:
            if reference_list[i] in data[cata].keys():
                classification_array[i] = SOLID_REFERENCE_VALUE
        if reference_list[i] in data["Fluid"].keys():
            classification_array[i] = FLUID_REFERENCE_VALUE
        if reference_list[i] == 'electric':
            classification_array[i] = ELECTRIC_REFERENCE_VALUE
        if reference_list[i] == 'heat':
            classification_array[i] = HEAT_REFERENCE_VALUE
        if RESEARCH_SPECIAL_STRING in reference_list[i]:
            classification_array[i] = RESEARCH_REFERENCE_VALUE

    assert not (classification_array==UNKNOWN_REFERENCE_VALUE).any(), np.array(reference_list)[classification_array==UNKNOWN_REFERENCE_VALUE]
    return classification_array

def get_belt_transport_table(classification_array: np.ndarray, reference_list: Sequence[str], data: dict) -> TransportTable:
    """numpy arrays for belt-teir density
    """
    solid_densities = np.zeros_like(classification_array)
    liquid_densities = np.zeros_like(classification_array)
    electric_densities = np.zeros_like(classification_array)
    heat_densities = np.zeros_like(classification_array)

    solid_densities[classification_array==SOLID_REFERENCE_VALUE] = 1
    liquid_densities[classification_array==FLUID_REFERENCE_VALUE] = 1
    electric_densities[classification_array==ELECTRIC_REFERENCE_VALUE] = 1
    heat_densities[classification_array==HEAT_REFERENCE_VALUE] = 1

    solid_options = np.zeros((classification_array.shape[0], 2))
    liquid_options = np.zeros((classification_array.shape[0], 2))
    electric_options = np.zeros((classification_array.shape[0], 1))
    heat_options = np.zeros((classification_array.shape[0], 1))

    for i in range(classification_array.shape[0]):
        if reference_list[i] in data['inserter'].keys():
            solid_options[i, 0] = data['inserter'][reference_list[i]]['throughput'] 
        if reference_list[i] in data['transport-belt'].keys():
            solid_options[i, 1] = data['transport-belt'][reference_list[i]]['speed'] * 480 #https://lua-api.factorio.com/latest/prototypes/TransportBeltConnectablePrototype.html#speed
        
        if reference_list[i] in data['pipe'].keys():
            liquid_options[i, 0] = 1200
        if reference_list[i] in data['pipe-to-ground'].keys():
            liquid_options[i, 1] = 1200

        if reference_list[i] in data['electric-pole'].keys():
            electric_options[i] = (2 * data['electric-pole'][reference_list[i]]['supply_area_distance']) ** 2 #https://lua-api.factorio.com/latest/prototypes/ElectricPolePrototype.html#supply_area_distance
        
        if reference_list[i] in data['heat-pipe'].keys():
            heat_options[i] = 1

    return TransportTable(TransportPair(solid_densities, solid_options), 
                          TransportPair(liquid_densities, liquid_options), 
                          TransportPair(electric_densities, electric_options), 
                          TransportPair(heat_densities, heat_options))

def get_rail_transport_table(classification_array: np.ndarray, stack_sizes: np.ndarray) -> TransportTable:
    """numpy arrays for rail-teir density
    """
    raise NotImplementedError

def get_logistics_transport_table(classification_array: np.ndarray, barrel_sizes: np.ndarray) -> TransportTable:
    """numpy arrays for rail-teir density
    """
    raise NotImplementedError

def get_space_platform_transport_table(classification_array: np.ndarray, barrel_sizes: np.ndarray) -> TransportTable:
    """numpy arrays for rail-teir density
    """
    raise NotImplementedError

def belt_transportation_cost(pricing_vector: np.ndarray, inverse_priced_indices: np.ndarray, belt_transport_table: TransportTable) -> TransportCostTable:
    """One side: 1 inserter, and prolly like 1 belt/item/sec
    Fluids: 1+1 normal pipe, 4+1 underground pipe
    """    
    (solid_densities, solid_options), (liquid_densities, liquid_options), (electric_densities, electric_options), (heat_densities, heat_options) = belt_transport_table

    valid_solid, valid_liquid, valid_electric, valid_heat = map(lambda options: options * (1 - inverse_priced_indices),
                                                                (solid_options, liquid_options, electric_options, heat_options))
    
    best_solid, best_liquid, best_electric, best_heat = map(lambda density, valid: (density * valid / pricing_vector).argmax(axis=0),
                                                            (solid_densities, liquid_densities, electric_densities, heat_densities),
                                                            (valid_solid, valid_liquid, valid_electric, valid_heat))
    
    solid_scaling = np.zeros((pricing_vector.shape[0], pricing_vector.shape[0]))
    solid_scaling[:, best_solid[0]] = 1.0 / solid_densities
    solid_scaling[:, best_solid[1]] = 1.0 / solid_densities

    liquid_static = np.zeros((pricing_vector.shape[0], pricing_vector.shape[0]))
    liquid_static[:, best_liquid[0]] = 2
    liquid_static[:, best_liquid[1]] = 5

    electric_static = np.zeros((pricing_vector.shape[0], pricing_vector.shape[0]))
    electric_static[:, best_electric] = 10 / electric_densities #is 10 appropriate?

    heat_static = np.zeros((pricing_vector.shape[0], pricing_vector.shape[0]))
    heat_static[:, best_heat] = 3

    return TransportCostTable(TransportCostPair(np.zeros_like(pricing_vector), solid_scaling),
                              TransportCostPair(liquid_static, np.zeros((pricing_vector.shape[0], pricing_vector.shape[0]))),
                              TransportCostPair(electric_static, np.zeros((pricing_vector.shape[0], pricing_vector.shape[0]))),
                              TransportCostPair(heat_static, np.zeros((pricing_vector.shape[0], pricing_vector.shape[0]))))
    

def rail_transportation_cost(pricing_vector: np.ndarray, inverse_priced_indices: np.ndarray, rail_transport_table: TransportTable) -> TransportCostTable:
    """One side: 2 inserters, a the highest level storage container, and prolly like 1 belt/item/sec
    Fluids: a couple of pipes and a pump.
    """
    raise NotImplementedError

def logistics_transportation_cost(pricing_vector: np.ndarray, inverse_priced_indices: np.ndarray, logistics_transport_table: TransportTable) -> TransportCostTable:
    """One side: prolly like .25 bots/item/sec
    Fluids: Barreling?
    """
    raise NotImplementedError

def space_platform_transporation_cost(pricing_vector: np.ndarray, inverse_priced_indices: np.ndarray, space_platform_transport_table: TransportTable) -> TransportCostTable:
    """Not even guessable right now.
    """
    raise NotImplementedError

BASE_TRANSPORT_COST_FUNCTION: TransportationMethod = belt_transportation_cost # type: ignore