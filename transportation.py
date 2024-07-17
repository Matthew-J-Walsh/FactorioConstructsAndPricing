from __future__ import annotations

from globalsandimports import *


SOLID_REFERENCE_VALUE: int = 0
FLUID_REFERENCE_VALUE: int = 1
ELECTRIC_REFERENCE_VALUE: int = 2
HEAT_REFERENCE_VALUE: int = 3
RESEARCH_REFERENCE_VALUE: int = 4
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

    @staticmethod
    def empty(size: int) -> TransportCostPair:
        return TransportCostPair(np.zeros((size, size)), np.zeros((size, size)))

class TransportationTableMethod(Protocol):
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
    def __call__(self, classification_array: np.ndarray, reference_list: Sequence[str], data: dict) -> TransportTable:     
        raise NotImplementedError

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
    def __call__(self, pricing_vector: np.ndarray, inverse_priced_indices: np.ndarray, classification_array: np.ndarray, transportation_table: TransportTable) -> TransportCostPair:     
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
        if reference_list[i] in data['fluid'].keys():
            classification_array[i] = FLUID_REFERENCE_VALUE
        if '@' in reference_list[i]:
            assert reference_list[i].split('@')[0] in data['fluid'].keys()
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

def get_rail_transport_table(classification_array: np.ndarray, reference_list: Sequence[str], data: dict) -> TransportTable:
    """numpy arrays for rail-teir density
    """
    return TransportTable(TransportPair(np.zeros(0), np.zeros(0)),
                          TransportPair(np.zeros(0), np.zeros(0)),
                          TransportPair(np.zeros(0), np.zeros(0)),
                          TransportPair(np.zeros(0), np.zeros(0)))

def get_logistic_transport_table(classification_array: np.ndarray, reference_list: Sequence[str], data: dict) -> TransportTable:
    """numpy arrays for rail-teir density
    """
    return TransportTable(TransportPair(np.zeros(0), np.zeros(0)),
                          TransportPair(np.zeros(0), np.zeros(0)),
                          TransportPair(np.zeros(0), np.zeros(0)),
                          TransportPair(np.zeros(0), np.zeros(0)))

def get_space_platform_transport_table(classification_array: np.ndarray, reference_list: Sequence[str], data: dict) -> TransportTable:
    """numpy arrays for rail-teir density
    """
    return TransportTable(TransportPair(np.zeros(0), np.zeros(0)),
                          TransportPair(np.zeros(0), np.zeros(0)),
                          TransportPair(np.zeros(0), np.zeros(0)),
                          TransportPair(np.zeros(0), np.zeros(0)))

BELT_TRANSPORT_STRING = "belt"
def belt_transportation_cost(pricing_vector: np.ndarray, inverse_priced_indices: np.ndarray, classification_array: np.ndarray, belt_transport_table: TransportTable) -> TransportCostPair:
    """One side: 1 inserter, and prolly like 1 belt/item/sec
    Fluids: 1+1 normal pipe, 4+1 underground pipe
    """    
    pricing_vector = pricing_vector.reshape(-1, 1)
    inverse_priced_indices = inverse_priced_indices.reshape(-1, 1)
    (solid_densities, solid_options), (liquid_densities, liquid_options), (electric_densities, electric_options), (heat_densities, heat_options) = belt_transport_table

    valid_solid, valid_liquid, valid_electric, valid_heat = map(lambda options: options * (1 - inverse_priced_indices),
                                                                (solid_options, liquid_options, electric_options, heat_options))
    
    #Wait why aren't density and valid the same array?
    best_solid, best_liquid, best_electric, best_heat = map(lambda valid: np.nanargmax(valid / pricing_vector, axis=0),
                                                            #(solid_densities, liquid_densities, electric_densities, heat_densities),
                                                            (valid_solid, valid_liquid, valid_electric, valid_heat))
    
    static = np.zeros((pricing_vector.shape[0], pricing_vector.shape[0]))
    scaling = np.zeros((pricing_vector.shape[0], pricing_vector.shape[0]))

    scaling[classification_array==SOLID_REFERENCE_VALUE, best_solid[0]] = solid_densities[classification_array==SOLID_REFERENCE_VALUE] / solid_options[best_solid[0], 0]
    scaling[classification_array==SOLID_REFERENCE_VALUE, best_solid[1]] = solid_densities[classification_array==SOLID_REFERENCE_VALUE] / solid_options[best_solid[1], 1]

    static[classification_array==FLUID_REFERENCE_VALUE, best_liquid[0]] = 2
    static[classification_array==FLUID_REFERENCE_VALUE, best_liquid[1]] = 5

    static[classification_array==ELECTRIC_REFERENCE_VALUE, best_electric] = 1

    static[classification_array==HEAT_REFERENCE_VALUE, best_heat] = 3

    return TransportCostPair(static.T, scaling)

RAIL_TRANSPORT_STRING = "rail"
def rail_transportation_cost(pricing_vector: np.ndarray, inverse_priced_indices: np.ndarray, classification_array: np.ndarray, rail_transport_table: TransportTable) -> TransportCostPair:
    """One side: 2 inserters, a the highest level storage container, and prolly like 1 belt/item/sec
    Fluids: a couple of pipes and a pump.
    """
    return TransportCostPair.empty(pricing_vector.shape[0])

LOGISTIC_TRANSPORT_STRING = "logistic"
def logistic_transportation_cost(pricing_vector: np.ndarray, inverse_priced_indices: np.ndarray, classification_array: np.ndarray, logistics_transport_table: TransportTable) -> TransportCostPair:
    """One side: prolly like .25 bots/item/sec
    Fluids: Barreling?
    """
    return TransportCostPair.empty(pricing_vector.shape[0])

SPACE_TRANSPORT_STRING = "space"
def space_platform_transporation_cost(pricing_vector: np.ndarray, inverse_priced_indices: np.ndarray, classification_array: np.ndarray, space_platform_transport_table: TransportTable) -> TransportCostPair:
    """Not even guessable right now.
    """
    return TransportCostPair.empty(pricing_vector.shape[0])


TRANSPORT_TABLE_FUNCTIONS: dict[str, TransportationTableMethod] = {BELT_TRANSPORT_STRING: get_belt_transport_table,
                                                                   RAIL_TRANSPORT_STRING: get_rail_transport_table,
                                                                   LOGISTIC_TRANSPORT_STRING: get_logistic_transport_table,
                                                                   SPACE_TRANSPORT_STRING: get_space_platform_transport_table} # type: ignore

TRANSPORT_COST_FUNCTIONS: dict[str, TransportationMethod] = {BELT_TRANSPORT_STRING: belt_transportation_cost,
                                                             RAIL_TRANSPORT_STRING: rail_transportation_cost,
                                                             LOGISTIC_TRANSPORT_STRING: logistic_transportation_cost,
                                                             SPACE_TRANSPORT_STRING: space_platform_transporation_cost} # type: ignore

