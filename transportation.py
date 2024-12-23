from __future__ import annotations

from globalsandimports import *
from datarawparse import *


_SOLID_REFERENCE_VALUE: int = 0
_FLUID_REFERENCE_VALUE: int = 1
_ELECTRIC_REFERENCE_VALUE: int = 2
_HEAT_REFERENCE_VALUE: int = 3
_RESEARCH_REFERENCE_VALUE: int = 4
_UNKNOWN_REFERENCE_VALUE: int = -1

_TRANSPORT_TYPE_REFERENCES: dict[str, int] = {
    "solid": _SOLID_REFERENCE_VALUE, 
    "fluid": _FLUID_REFERENCE_VALUE, 
    "electric": _ELECTRIC_REFERENCE_VALUE, 
    "heat": _HEAT_REFERENCE_VALUE
}

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
    def __call__(self, pricing_vector: np.ndarray, inverse_priced_indices: np.ndarray, known_technologies: TechnologicalLimitation) -> TransportationCompiler:     
        raise NotImplementedError

@runtime_checkable
class TransportationMethod(Protocol):
    """Defines a transporation method
    """    
    def __call__(self, instance: FactorioInstance) -> TransportationPrecompiler:     
        raise NotImplementedError


def classify_reference_list(reference_list: tuple[str, ...], data: dict) -> np.ndarray:
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
    classification_array = np.full(len(reference_list), _UNKNOWN_REFERENCE_VALUE, dtype=int)
    for i in range(len(reference_list)):
        for cata in ITEM_SUB_PROTOTYPES:
            if reference_list[i] in data[cata].keys():
                classification_array[i] = _SOLID_REFERENCE_VALUE
        if reference_list[i] in data['fluid'].keys():
            classification_array[i] = _FLUID_REFERENCE_VALUE
        if '@' in reference_list[i]:
            assert reference_list[i].split('@')[0] in data['fluid'].keys()
            classification_array[i] = _FLUID_REFERENCE_VALUE
        if reference_list[i] == 'electric':
            classification_array[i] = _ELECTRIC_REFERENCE_VALUE
        if reference_list[i] == 'heat':
            classification_array[i] = _HEAT_REFERENCE_VALUE
        if RESEARCH_SPECIAL_STRING in reference_list[i]:
            classification_array[i] = _RESEARCH_REFERENCE_VALUE

    assert not (classification_array==_UNKNOWN_REFERENCE_VALUE).any(), np.array(reference_list)[classification_array==_UNKNOWN_REFERENCE_VALUE]
    return classification_array


def transport_fuel_options(energy_usage: Fraction, energy_source: dict, instance: FactorioInstance) -> np.ndarray:
    """Calculate the fuel options for a transportation energy source

    Parameters
    ----------
    energy_usage : Fraction
        Amount of energy that is used
    energy_source : dict
        Energy source https://lua-api.factorio.com/stable/types/EnergySource.html
    instance : FactorioInstance
        Associated FactorioInstance
    
    Returns
    -------
    np.ndarray
        A matrix where the first dimension is the fuel option and the second is the effect of that fuel
    """
    fuel_options: list[np.ndarray] = []
    for fuel_name, energy_density, burnt_result in fuels_from_energy_source(energy_source, instance):
        fuel_option = np.zeros((1, len(instance.reference_list)))
        fuel_option[0, instance.reference_list.index(fuel_name)] = float(-1 * energy_usage / energy_density)
        if burnt_result:
            fuel_option[0, instance.reference_list.index(burnt_result)] = float(energy_usage / energy_density)
        fuel_options.append(fuel_option)

    return np.concatenate(fuel_options, axis=0)

def standard_transportation_cost(instance: FactorioInstance, options: dict[str, tuple[list[str], ...]], option_scalings: dict[str, list[str]], 
                                 option_type: dict[str, list[str]], default_densities: dict[str, np.ndarray]) -> TransportationPrecompiler:
    """Standard transport cost method generator

    Parameters
    ----------
    instance : FactorioInstance
        Associated FactorioInstance
    options : dict[str, tuple[list[str], ...]]
        Transportation options avaiable
    option_scalings : dict[str, list[str]]
        Scaling of the transporation options
    option_type : dict[str, list[str]]
        What value of the transporation option to calculate with
    default_densities : dict[str, np.ndarray]
        Default required transporation density

    Returns
    -------
    TransportationPrecompiler
        A transporation precompiling function

    Raises
    ------
    ValueError
        Energy source errors
    """
    
    inverted_options: dict[str, tuple[str, int]] = {}
    for transport_type, l in options.items():
        for i in range(len(l)):
            sl = l[i]
            for e in sl:
                inverted_options[e] = (transport_type, i)

    densities = {transport_type: np.zeros((len(instance.reference_list), len(option_scalings[transport_type]))) for transport_type in _TRANSPORT_TYPE_REFERENCES.keys()}
    for transport_type, transport_type_reference in _TRANSPORT_TYPE_REFERENCES.items():
        densities[transport_type][instance.reference_classifications==transport_type_reference] = default_densities[transport_type]

    option_tables = {transport_type: np.zeros((len(instance.reference_list), len(option_scalings[transport_type]))) for transport_type in _TRANSPORT_TYPE_REFERENCES.keys()}
    fuel_tables: dict[str, tuple[dict[str, np.ndarray], ...]] = {transport_type: tuple([{} for _ in range(len(option_scalings[transport_type]))]) for transport_type in _TRANSPORT_TYPE_REFERENCES.keys()}

    for element, (transport_type, i) in inverted_options.items():
        for element_option in instance.data_raw[element].values():
            #if element_option['name'] in instance.reference_list:
            option_tables[transport_type][instance.reference_list.index(element_option['name']), i] = (element_option[option_type[transport_type][i]] if option_type[transport_type][i] in element_option.keys() else 1)
            if 'energy_source' in element_option:
                try:
                    fuel_tables[transport_type][i][element_option['name']] = transport_fuel_options(element_option['energy_usage_raw'], element_option['energy_source'], instance)
                except:
                    raise ValueError(element_option['type'])

    def transport_precompiled_cost(pricing_vector: np.ndarray, inverse_priced_indices: np.ndarray, known_technologies: TechnologicalLimitation) -> TransportationCompiler:
        pricing_vector = pricing_vector.reshape(-1, 1)
        inverse_priced_indices = inverse_priced_indices.reshape(-1, 1)

        valid_tables = {transport_type: tab * (1 - inverse_priced_indices) for transport_type, tab in option_tables.items()}
        
        best_tables = {transport_type: np.nanargmax(valid_table / pricing_vector, axis=0) for transport_type, valid_table in valid_tables.items()}
        
        static = np.zeros((pricing_vector.shape[0], pricing_vector.shape[0]))
        scaling = np.zeros((pricing_vector.shape[0], pricing_vector.shape[0]))

        for transport_type, transport_type_reference in _TRANSPORT_TYPE_REFERENCES.items():
            for i, best_sub_element_index in enumerate(best_tables[transport_type]):
                if option_scalings[transport_type][i]=="scaling":
                    scaling[instance.reference_classifications==transport_type_reference, best_sub_element_index] = densities[transport_type][instance.reference_classifications==transport_type_reference, i] / option_tables[transport_type][best_sub_element_index, i]
                else: #elif ordinalities_placeholder_name[transport_type][i]=="static":
                    scaling[instance.reference_classifications==transport_type_reference, best_sub_element_index] = densities[transport_type][instance.reference_classifications==transport_type_reference, i]

        static = static.T

        def transport_compiled_cost(dual_vector: np.ndarray | None) -> TransportCost:

            static_fuel = np.zeros((len(instance.reference_list), len(instance.reference_list)))
            scaling_fuel = np.zeros((len(instance.reference_list), len(instance.reference_list)))

            for transport_type, transport_type_reference in _TRANSPORT_TYPE_REFERENCES.items():
                for i, best_sub_element_index in enumerate(best_tables[transport_type]):
                    if instance.reference_list[best_sub_element_index] in fuel_tables[transport_type][i]:
                        if dual_vector is None:
                            best_fuel = 0
                        else:
                            best_fuel = int((fuel_tables[transport_type][i][instance.reference_list[best_sub_element_index]] @ dual_vector).argmax()) #argmax might have weird results TODO
                        if option_scalings[transport_type][i]=="scaling":
                            scaling_fuel[instance.reference_classifications==transport_type_reference] += fuel_tables[transport_type][i][instance.reference_list[best_sub_element_index]][best_fuel]
                        else: #elif ordinalities_placeholder_name[transport_type][i]=="static":
                            static_fuel[instance.reference_classifications==transport_type_reference] += fuel_tables[transport_type][i][instance.reference_list[best_sub_element_index]][best_fuel]

            if dual_vector is None:
                return TransportCost(static, np.zeros(static_fuel.shape[0]), scaling, np.zeros(scaling_fuel.shape[0]))
            else:
                return TransportCost(static, static_fuel @ dual_vector, scaling, scaling_fuel @ dual_vector)
        
        return transport_compiled_cost
    
    return transport_precompiled_cost

def belt_transportation_cost(instance: FactorioInstance) -> TransportationPrecompiler:
    """Transporation cost method for belts

    Parameters
    ----------
    instance : FactorioInstance
        Associated FactorioInstance

    Returns
    -------
    TransportationPrecompiler
        Transporation cost precompiler for belts
    """
    options: dict[str, tuple[list[str], ...]] = {
        "solid": (["inserter"], ["transport-belt"]),
        "fluid": (["pipe"], ["pipe-to-ground"]),
        "electric": (["electric-pole"],),
        "heat": (["heat-pipe"],),
    }

    option_scalings: dict[str, list[str]] = {
        "solid": ['scaling', 'scaling'],
        "fluid": ['static', 'static'],
        "electric": ['static'],
        "heat": ['static'],
    }

    option_type: dict[str, list[str]] = {
        "solid": ['throughput', 'throughput'],
        "fluid": ['throughput', 'throughput'],
        "electric": ['area'],
        "heat": ['throughput'],
    }

    default_densities: dict[str, np.ndarray] = {
        "solid": np.array([1, 1]),
        "fluid": np.array([2, 5]),
        "electric": np.array([1]),
        "heat": np.array([3]),
    }

    return standard_transportation_cost(instance, options, option_scalings, option_type, default_densities)

def rail_transportation_cost(instance: FactorioInstance) -> TransportationPrecompiler:
    """Transporation cost method for rails

    Parameters
    ----------
    instance : FactorioInstance
        Associated FactorioInstance

    Returns
    -------
    TransportationPrecompiler
        Transporation cost precompiler for rails
    """
    options: dict[str, tuple[list[str], ...]] = {
        "solid": (["rail-planner"], ["train-stop"], ["rail-signal"], ["rail-chain-signal"], ["locomotive"], ["cargo-wagon"]),
        "fluid": (["rail-planner"], ["train-stop"], ["rail-signal"], ["rail-chain-signal"], ["locomotive"], ["fluid-wagon"]),
        "electric": (["electric-pole"],),
        "heat": tuple([]),
    }

    option_scalings: dict[str, list[str]] = {
        "solid": ['static', 'static', 'static', 'static', 'static', 'static'],
        "fluid": ['static', 'static', 'static', 'static', 'static', 'static'],
        "electric": ['static'],
        "heat": [],
    }

    option_type: dict[str, list[str]] = {
        "solid": ['throughput', 'throughput', 'throughput', 'throughput', 'throughput', 'throughput'],
        "fluid": ['throughput', 'throughput', 'throughput', 'throughput', 'throughput', 'throughput'],
        "electric": ['length'],
        "heat": [],
    }

    default_densities: dict[str, np.ndarray] = {
        "solid": np.array([200, 2, 20, 20, 1, 4]),
        "fluid": np.array([200, 2, 20, 20, 1, 4]),
        "electric": np.array([1000]),
        "heat": np.zeros(0),
    }

    return standard_transportation_cost(instance, options, option_scalings, option_type, default_densities)

def logistic_transportation_cost(instance: FactorioInstance) -> TransportationPrecompiler:
    """Transporation cost method for logistics bots

    Parameters
    ----------
    instance : FactorioInstance
        Associated FactorioInstance

    Returns
    -------
    TransportationPrecompiler
        Transporation cost precompiler for logistics bots
    """
    options: dict[str, tuple[list[str], ...]] = {
        "solid": (["roboport"], ["logistic-robot"], ['logistic-container']),
        "fluid": (),
        "electric": (["electric-pole"],),
        "heat": tuple([]),
    }

    option_scalings: dict[str, list[str]] = {
        "solid": ['static', 'scaling', 'static'],
        "fluid": [],
        "electric": ['static'],
        "heat": [],
    }

    option_type: dict[str, list[str]] = {
        "solid": ['throughput', 'throughput', 'throughput'],
        "fluid": [],
        "electric": ['length'],
        "heat": [],
    }

    default_densities: dict[str, np.ndarray] = {
        "solid": np.array([1, 1, 2]),
        "fluid": np.zeros(0),
        "electric": np.array([1000]),
        "heat": np.zeros(0),
    }

    return standard_transportation_cost(instance, options, option_scalings, option_type, default_densities)

def get_space_platform_transport_table(instance: FactorioInstance) -> TransportationPrecompiler:
    """numpy arrays for space-teir density
    """
    raise NotImplementedError
    return TransportTable(TransportPair(np.zeros(0), np.zeros(0)),
                          TransportPair(np.zeros(0), np.zeros(0)),
                          TransportPair(np.zeros(0), np.zeros(0)),
                          TransportPair(np.zeros(0), np.zeros(0)))


TRANSPORT_COST_FUNCTIONS: dict[str, TransportationMethod] = {BELT_TRANSPORT_STRING: belt_transportation_cost,
                                                             RAIL_TRANSPORT_STRING: rail_transportation_cost,
                                                             LOGISTIC_TRANSPORT_STRING: logistic_transportation_cost,}
                                                             #SPACE_TRANSPORT_STRING: space_platform_transporation_cost}

