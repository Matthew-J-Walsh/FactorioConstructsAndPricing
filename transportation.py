from __future__ import annotations

from globalsandimports import *


SOLID_REFERENCE_VALUE: int = 0
FLUID_REFERENCE_VALUE: int = 1
ELECTRIC_REFERENCE_VALUE: int = 2
HEAT_REFERENCE_VALUE: int = 3
RESEARCH_REFERENCE_VALUE: int = 4
UNKNOWN_REFERENCE_VALUE: int = -1


BELT_TRANSPORT_STRING = "belt"
RAIL_TRANSPORT_STRING = "rail"
LOGISTIC_TRANSPORT_STRING = "logistic"
SPACE_TRANSPORT_STRING = "space"


class TransportCost():
    static_cost: np.ndarray
    static_effect: np.ndarray
    scaling_cost: np.ndarray
    scaling_effect: np.ndarray

    def __init__(self, static_cost: np.ndarray, static_effect: np.ndarray, scaling_cost: np.ndarray, scaling_effect: np.ndarray) -> None:
        self.static_cost = static_cost
        self.static_effect = static_effect
        self.scaling_cost = scaling_cost
        self.scaling_effect = scaling_effect

    @staticmethod
    def empty(size: int) -> TransportCost:
        return TransportCost(np.zeros((size, size)), np.zeros(size), np.zeros((size, size)), np.zeros(size))
    
    def __add__(self, other: TransportCost) -> TransportCost:
        return TransportCost(self.static_cost + other.static_cost, self.static_effect + other.static_effect, 
                             self.scaling_cost + other.scaling_cost, self.scaling_effect + other.scaling_effect)

@runtime_checkable
class TransportationCompiler(Protocol):
    """Completes a transport cost analysis with proper energy analysis
    """    
    def __call__(self, dual_vector: np.ndarray | None) -> TransportCost:
        raise NotImplementedError

@runtime_checkable
class TransportationPrecompiler(Protocol):
    """Compiles a transportation method for a factory
    """    
    def __call__(self, pricing_vector: np.ndarray, inverse_priced_indices: np.ndarray) -> TransportationCompiler:     
        raise NotImplementedError

@runtime_checkable
class TransportationMethod(Protocol):
    """Defines a transporation method
    """    
    def __call__(self, classification_array: np.ndarray, reference_list: Sequence[str], data: dict) -> TransportationPrecompiler:     
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


def belt_transportation_cost(classification_array: np.ndarray, reference_list: Sequence[str], data: dict) -> TransportationPrecompiler:
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

    def belt_transport_precompiled_cost(pricing_vector: np.ndarray, inverse_priced_indices: np.ndarray) -> TransportationCompiler:
        pricing_vector = pricing_vector.reshape(-1, 1)
        inverse_priced_indices = inverse_priced_indices.reshape(-1, 1)

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

        static = static.T

        def belt_transport_compiled_cost(dual_vector: np.ndarray | None) -> TransportCost:

            return TransportCost(static, np.zeros(static.shape[0]), scaling, np.zeros(scaling.shape[0]))
        
        return belt_transport_compiled_cost
    
    return belt_transport_precompiled_cost

def get_rail_transport_table(classification_array: np.ndarray, reference_list: Sequence[str], data: dict) -> TransportationPrecompiler:
    """numpy arrays for rail-teir density
    """
    raise NotImplementedError
    return TransportTable(TransportPair(np.zeros(0), np.zeros(0)),
                          TransportPair(np.zeros(0), np.zeros(0)),
                          TransportPair(np.zeros(0), np.zeros(0)),
                          TransportPair(np.zeros(0), np.zeros(0)))

def get_logistic_transport_table(classification_array: np.ndarray, reference_list: Sequence[str], data: dict) -> TransportationPrecompiler:
    """numpy arrays for rail-teir density
    """
    raise NotImplementedError
    return TransportTable(TransportPair(np.zeros(0), np.zeros(0)),
                          TransportPair(np.zeros(0), np.zeros(0)),
                          TransportPair(np.zeros(0), np.zeros(0)),
                          TransportPair(np.zeros(0), np.zeros(0)))

def get_space_platform_transport_table(classification_array: np.ndarray, reference_list: Sequence[str], data: dict) -> TransportationPrecompiler:
    """numpy arrays for rail-teir density
    """
    raise NotImplementedError
    return TransportTable(TransportPair(np.zeros(0), np.zeros(0)),
                          TransportPair(np.zeros(0), np.zeros(0)),
                          TransportPair(np.zeros(0), np.zeros(0)),
                          TransportPair(np.zeros(0), np.zeros(0)))


TRANSPORT_COST_FUNCTIONS: dict[str, TransportationMethod] = {BELT_TRANSPORT_STRING: belt_transportation_cost,}
                                                             #RAIL_TRANSPORT_STRING: rail_transportation_cost,
                                                             #LOGISTIC_TRANSPORT_STRING: logistic_transportation_cost,
                                                             #SPACE_TRANSPORT_STRING: space_platform_transporation_cost}

