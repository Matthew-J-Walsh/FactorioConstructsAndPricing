from __future__ import annotations

from globalsandimports import *
from utils import *

if TYPE_CHECKING:
    from tools import FactorioInstance

class UncompiledConstruct:
    """An uncompiled construct, contains all the information needed to compile a single construct.

    Members
    -------
    ident : str
        Unique identifier
    drain : CompressedVector
        Passive drain to the product space
    deltas : CompressedVector
        Changes to product space from running the construct
    effect_effects : dict[str, list[str]]
        Specifies how this construct is affected by module effects
    allowed_modules : list[tuple[str, bool, bool]]
        Each tuple represents a module, if it can be used inside the building, and if it can be used in beacons for the building
    internal_module_limit : int
        Number of module slots inside the building
    base_inputs : CompressedVector
        The inputs required to start the machine, used for future catalyst calculations
    cost : CompressedVector
        The cost of a single instance (without any modules)
    limit : TechnologicalLimitation
        Required technological level to make this construct (without any modules)
    building : dict
        Link the the building entity for tile size values
        https://lua-api.factorio.com/latest/prototypes/EntityPrototype.html#tile_width
        https://lua-api.factorio.com/latest/prototypes/EntityPrototype.html#tile_height
    base_productivity : Fraction
        Baseline productivity effect of the building
        https://lua-api.factorio.com/latest/prototypes/CraftingMachinePrototype.html#base_productivity
        https://lua-api.factorio.com/latest/prototypes/MiningDrillPrototype.html#base_productivity
    research_effected : list[str]
        What research modifiers effect this construct
    """
    ident: str
    drain: CompressedVector
    deltas: CompressedVector
    effect_effects: dict[str, list[str]]
    allowed_modules: list[tuple[str, bool, bool]]
    internal_module_limit: int
    base_productivity: Fraction
    base_inputs: CompressedVector
    cost: CompressedVector
    limit: TechnologicalLimitation
    building: dict
    research_effected: list[str]

    def __init__(self, ident: str, drain: CompressedVector, deltas: CompressedVector, effect_effects: dict[str, list[str]], 
                 allowed_modules: list[tuple[str, bool, bool]], internal_module_limit: int, base_inputs: CompressedVector, cost: CompressedVector, 
                 limit: TechnologicalLimitation, building: dict, base_productivity: Fraction = Fraction(0), research_effected: list[str] | None = None) -> None:
        """
        Parameters
        ----------
        ident : str
            Unique identifier
        drain : CompressedVector
            Passive drain to the product space
        deltas : CompressedVector
            Changes to product space from running the construct
        effect_effects : dict[str, list[str]]
            Specifies how this construct is affected by module effects
        allowed_modules : list[tuple[str, bool, bool]]
            Each tuple represents a module, if it can be used inside the building, and if it can be used in beacons for the building
        internal_module_limit : int
            Number of module slots inside the building
        base_inputs : CompressedVector
            The inputs required to start the machine, used for future catalyst calculations
        cost : CompressedVector
            The cost of a single instance (without any modules)
        limit : TechnologicalLimitation
            Required technological level to make this construct (without any modules)
        building : dict
            Link the the building entity for tile size values
            https://lua-api.factorio.com/latest/prototypes/EntityPrototype.html#tile_width
            https://lua-api.factorio.com/latest/prototypes/EntityPrototype.html#tile_height
        base_productivity : Fraction, optional
            Baseline productivity effect of the building
            https://lua-api.factorio.com/latest/prototypes/CraftingMachinePrototype.html#base_productivity
            https://lua-api.factorio.com/latest/prototypes/MiningDrillPrototype.html#base_productivity
            , by default Fraction(0)
        research_effected : list[str], optional
            What research modifiers effect this construct
        """
        self.ident = ident
        self.drain = drain
        self.deltas = deltas
        self.effect_effects = effect_effects
        self.allowed_modules = allowed_modules
        self.internal_module_limit = internal_module_limit
        self.base_inputs = base_inputs
        self.cost = cost
        self.limit = limit
        self.building = building
        self.base_productivity = base_productivity
        if not research_effected is None:
            self.research_effected = research_effected
        else:
            self.research_effected = []
        
    def __repr__(self) -> str:
        return str(self.ident)+\
                "\n\tAn added drain of: "+str(self.drain)+\
                "\n\tWith a vector of: "+str(self.deltas)+\
                "\n\tAn effect table of: "+str(self.effect_effects)+\
                "\n\tAllowed modules: "+str(self.allowed_modules)+\
                "\n\tInternal module count: "+str(self.internal_module_limit)+\
                "\n\tBase productivity: "+str(self.base_productivity)+\
                "\n\tBase inputs of: "+str(self.base_inputs)+\
                "\n\tA Cost of: "+str(self.cost)+\
                "\n\tRequiring: "+str(self.limit)+\
                "\n\tBuilding size of: "+str(self.building['tile_width'])+" by "+str(self.building['tile_height'])


class ManualConstruct:
    """Manual Constructs are hand crafted constructs. This should only be used when there is no other way to progress

    Members
    -------
    ident : str
        Unique identifier
    effect_vector : np.ndarray
        Column vector of this manual action
    limit : TechnologicalLimitation
        Tech level required to complete this manual action
    """
    ident: str
    deltas: CompressedVector
    effect_vector: np.ndarray
    limit: TechnologicalLimitation

    def __init__(self, ident: str, deltas: CompressedVector, limit: TechnologicalLimitation, instance: FactorioInstance):
        """
        Parameters
        ----------
        ident : str
            Unique identifier
        deltas : CompressedVector
            Deltas from running construct
        limit : TechnologicalLimitation
            Tech level required to complete this manual action
        instance : FactorioInstance
            FactorioInstance in use
        """        
        self.ident = ident
        self.deltas = deltas
        self.effect_vector = np.zeros(len(instance.reference_list))
        for k, v in deltas.items():
            self.effect_vector[instance.reference_list.index(k)] = v
        self.limit = limit

    def vector(self, known_technologies: TechnologicalLimitation) -> tuple[np.ndarray, float, str | None]:
        """Gets the vector for this construct

        Parameters
        ----------
        known_technologies : TechnologicalLimitation
            Current tech level

        Returns
        -------
        tuple[np.ndarray, float, str | None]
            Effect vector
            Cost
            Ident
        """        
        if self.limit >= known_technologies:
            return self.effect_vector, 1, None
        else:
            return np.zeros_like(self.effect_vector), 0, self.ident
        
    @staticmethod
    def vectors(all_constructs: tuple[ManualConstruct, ...], known_technologies: TechnologicalLimitation) -> ColumnTable:
        """Calculates the vectors for all ManualConstructs

        Parameters
        ----------
        all_constructs : tuple[ManualConstruct, ...]
            All ManualConstructs to calculate vectors for
        known_technologies : TechnologicalLimitation
            Current tech level

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray[CompressedVector, Any]]
            Matrix of effect vectors,
            Vector of costs,
            Matrix of exact costs,
            Ident vectors
        """        
        construct_arr: np.ndarray[ManualConstruct, Any] = np.array(all_constructs, dtype=ManualConstruct)
        mask = np.array([known_technologies >= construct.limit for construct in construct_arr])
        construct_arr = construct_arr[mask]

        vectors = np.vstack([construct.effect_vector for construct in construct_arr]).T
        costs = np.ones_like(construct_arr, dtype=np.float64)
        true_costs = np.vstack([np.zeros_like(construct.effect_vector) for construct in construct_arr]).T
        idents = np.concatenate([np.array([CompressedVector({construct.ident: 1})]) for construct in construct_arr])

        return ColumnTable(vectors, costs, true_costs, idents)
    
    def __repr__(self) -> str:
        return str(self.ident)+\
                "\n\tWith a deltas of: "+str(self.deltas)+\
                "\n\tRequiring: "+str(self.limit)
                #"\n\tWith a vector of: "+str(self.effect_vector)+\

def module_setup_generator(allowed_modules: list[tuple[str, bool, bool]], internal_module_limit: int, building_size: tuple[int, int], beacon: dict | None = None) -> Generator[tuple[CompressedVector, CompressedVector], None, None]:
    """Returns an generator over the set of possible module setups.

    Parameters
    ----------
    allowed_modules : list[tuple[str, bool, bool]]
        Each tuple represents a module, if it can be used inside the building, and if it can be used in beacons for the building
    internal_module_limit : int
        The number of internal module slots
    building_size : tuple[int, int]
        Size of building's tile
    beacon : dict | None, optional
        Beacon being used, by default None

    Yields
    ------
    Generator[tuple[CompressedVector, CompressedVector], None, None]
        All the modules effecting an average building
        Costs per building
    """    
    if len(allowed_modules)==0 or DEBUG_BLOCK_MODULES:
        yield CompressedVector(), CompressedVector()
    else:
        internal_modules = [m for m, i, _ in allowed_modules if i]
        external_modules = [m for m, _, e in allowed_modules if e]

        if not beacon is None and not DEBUG_BLOCK_BEACONS:
            for effecting_beacon_count, beacons_per_building in beacon_setups(building_size, beacon):
                for internal_mod_count in range(internal_module_limit+1):
                    for internal_mod_setup in itertools.combinations_with_replacement(internal_modules, internal_mod_count):
                        #for external_mod_setup in itertools.combinations_with_replacement(exteral_modules, beacon_count*beacon['module_specification']['module_slots']): #too big
                        for external_mod in external_modules:
                            effect_vector = CompressedVector()
                            cost_vector = CompressedVector()
                            for mod in internal_mod_setup:
                                effect_vector = effect_vector + CompressedVector({mod+"|i": 1})
                                cost_vector = cost_vector + CompressedVector({mod: 1})
                            yield effect_vector + CompressedVector({external_mod+"|e": effecting_beacon_count * beacon['module_specification']['module_slots']}), \
                                  cost_vector + CompressedVector({beacon['name']: 1 * beacons_per_building}) + CompressedVector({external_mod: 1 * beacons_per_building * beacon['module_specification']['module_slots']}) # type: ignore
        else:
            for internal_mod_count in range(internal_module_limit+1):
                for internal_mod_setup in itertools.combinations_with_replacement(internal_modules, internal_mod_count):
                    effect_vector = CompressedVector()
                    cost_vector = CompressedVector()
                    for mod in internal_mod_setup:
                        effect_vector = effect_vector + CompressedVector({mod+"|i": 1})
                        cost_vector = cost_vector + CompressedVector({mod: 1})
                    yield effect_vector, cost_vector

def model_point_generator(allowed_internal_module_count: int, internal_module_limit: int, 
                          allowed_external_module_count: int, beacon_module_limit: int, 
                          beacon_dimensions: tuple[int, ...]) -> np.ndarray:
    if allowed_internal_module_count==0:
        return np.zeros((0, 0))
    #print(allowed_internal_module_count)
    #print(internal_module_limit)
    #print(allowed_external_module_count)
    #print(beacon_module_limit)
    #print(beacon_dimensions)
    internal_module_setups = np.column_stack([g.ravel() for g in np.meshgrid(*([np.arange(internal_module_limit+1)]*allowed_internal_module_count))])
    internal_module_setups = internal_module_setups[internal_module_setups.sum(axis=1)<=internal_module_limit,:]
    #print(internal_module_setups.shape)

    external_module_internals = np.column_stack([g.ravel() for g in np.meshgrid(*([np.arange(beacon_module_limit+1)]*allowed_external_module_count))])
    external_module_internals = external_module_internals[
        np.logical_and(external_module_internals.sum(axis=1)<=beacon_module_limit, (external_module_internals==beacon_module_limit).sum(axis=1)>0)
        ,:]
    #print(external_module_internals.shape)

    beacon_designs = np.zeros((1+sum(beacon_dimensions), len(beacon_dimensions)))
    i = 1
    for j in range(len(beacon_dimensions)):
        for k in range(1, 1+beacon_dimensions[j]):
            beacon_designs[i, j] = k
    #print(beacon_designs.shape)
    
    internals = np.repeat(internal_module_setups, external_module_internals.shape[0] * beacon_designs.shape[0], axis=0)
    externals = np.repeat(external_module_internals, internal_module_setups.shape[0] * beacon_designs.shape[0], axis=0)
    beacons = np.repeat(beacon_designs, internal_module_setups.shape[0] * external_module_internals.shape[0], axis=0)
    #print(internals.shape)
    #print(externals.shape)
    #print(beacons.shape)
    return np.concatenate([internals, externals, beacons], axis=1)

def generate_module_vector_lambdas(allowed_internal_modules: tuple[dict, ...], allowed_external_modules: tuple[dict, ...], beacon_designs: tuple[tuple[dict, tuple[tuple[Fraction, Fraction], ...]], ...],
                                   reference_list: tuple[str, ...]) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    allowed_internal_module_count = len(allowed_internal_modules)
    allowed_external_module_count = len(allowed_external_modules)

    beacon_design_effect_table = np.zeros((len(beacon_designs), max([max([len(design) for design in beacon]) for beacon in beacon_designs])))
    beacon_design_cost_table = np.zeros((len(beacon_designs), max([max([len(design) for design in beacon]) for beacon in beacon_designs])))
    beacon_design_beacon_cost = np.zeros((len(beacon_designs), max([max([len(design) for design in beacon]) for beacon in beacon_designs])), dtype=int)
    beacon_design_electric_cost = np.zeros((len(beacon_designs), max([max([len(design) for design in beacon]) for beacon in beacon_designs])), dtype=int)
    for i, (beacon, designs) in enumerate(beacon_designs):
        for j, design in enumerate(designs):
            beacon_design_effect_table[i, j] = design[0]
            beacon_design_cost_table[i, j] = design[1]
            beacon_design_beacon_cost[i, j] = reference_list.index(beacon['name'])
            beacon_design_electric_cost[i, j] = -1 * beacon['energy_usage_raw']
    electric_index = reference_list.index('electric')

    internal_effect_matrix = np.zeros((len(ALL_MODULE_EFFECTS), allowed_internal_module_count))
    internal_cost_matrix = np.zeros((len(reference_list), allowed_internal_module_count))
    for i, module in enumerate(allowed_internal_modules):
        for j, effect_name in enumerate(ALL_MODULE_EFFECTS):
            if effect_name in module['effect'].keys():
                internal_effect_matrix[j, i] = module['effect'][effect_name]['bonus']

    external_effect_matrix = np.zeros((len(ALL_MODULE_EFFECTS), allowed_external_module_count))
    external_cost_matrix = np.zeros((len(reference_list), allowed_external_module_count))
    for i, module in enumerate(allowed_external_modules):
        for j, effect_name in enumerate(ALL_MODULE_EFFECTS):
            if effect_name in module['effect'].keys():
                external_effect_matrix[j, i] = module['effect'][effect_name]['bonus']

    def dual_func(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(points.shape)==1 or points.shape[1]==1:
            points = points.reshape((1, -1))

        if points.shape[1]==0:
            return np.zeros((0, len(ALL_MODULE_EFFECTS))), np.zeros((0, len(reference_list))), np.zeros((0, len(ALL_MODULE_EFFECTS)))

        internal_module_portion = points[:, :allowed_internal_module_count]
        beacon_module_portion = points[:, allowed_internal_module_count:allowed_internal_module_count+allowed_external_module_count]
        beacon_design_portion = points[:, allowed_internal_module_count+allowed_external_module_count:]

        chosen_beacon_design = (-1 * beacon_design_portion.sum(axis=1)==0 + np.argmax(beacon_design_portion, axis=1) + 1).astype(int)
        chosen_beacon_design_index = beacon_design_portion.sum(axis=1).astype(int)
        beacon_effect_multi = beacon_design_effect_table[chosen_beacon_design, chosen_beacon_design_index]
        beacon_cost_multi = beacon_design_cost_table[chosen_beacon_design, chosen_beacon_design_index]
        beacon_direct_cost = np.zeros((len(reference_list), points.shape[0]))
        beacon_direct_cost[beacon_design_beacon_cost[chosen_beacon_design, chosen_beacon_design_index], :] = beacon_cost_multi
        beacon_electric_effect = np.zeros((len(reference_list), points.shape[0]))
        beacon_electric_effect[electric_index, :] = beacon_cost_multi * beacon_design_electric_cost[chosen_beacon_design, chosen_beacon_design_index]

        internal_effect = internal_effect_matrix @ internal_module_portion.T
        internal_cost = internal_cost_matrix @ internal_module_portion.T

        external_effect = (external_effect_matrix @ beacon_module_portion.T) * beacon_effect_multi
        external_cost = (external_cost_matrix @ beacon_module_portion.T) * beacon_cost_multi + beacon_direct_cost

        multilinear_effect = np.ones([points.shape[0]]+[2]*len(ALL_MODULE_EFFECTS))
        for i in range(points.shape[0]):
            multilinear_effect[i] += MODULE_MULTILINEAR_EFFECT_SELECTORS[i] * (internal_effect + external_effect) #mins and maxes

        return multilinear_effect.reshape((points.shape[0], -1)), beacon_electric_effect.T, (internal_cost + external_cost).T

    return dual_func

def beacon_setups(building_size: tuple[int, int], beacon: dict) -> list[tuple[Fraction, Fraction]]:
    """Determines the possible optimal beacon setups for a building and beacon

    Parameters
    ----------
    building_size : tuple[int, int]
        Size of building's tile
    beacon : dict
        Beacon buffing the building

    Returns
    -------
    list[tuple[int, Fraction]]
        List of tuples with beacons hitting each building and beacons/building in the tesselation

    Raises
    ------
    ValueError
        When cannot calculate the building size properly
    """    
    try:
        M_plus = max(building_size)
        M_minus = min(building_size)
    except:
        raise ValueError(building_size)
    B_plus = max(beacon['tile_width'], beacon['tile_height'])
    B_minus = min(beacon['tile_width'], beacon['tile_height'])
    E_plus = int(beacon['supply_area_distance'])*2+B_plus
    E_minus = int(beacon['supply_area_distance'])*2+B_minus

    setups = []
    #surrounded buildings: same direction
    surrounded_buildings_same_direction_side_A = math.floor((E_plus - B_plus - 2 + M_plus)*1.0/B_minus)
    surrounded_buildings_same_direction_side_B = math.floor((E_minus - B_minus - 2 + M_minus)*1.0/B_minus)
    setups.append((4+2*surrounded_buildings_same_direction_side_A+2*surrounded_buildings_same_direction_side_B,
                   1*Fraction(2+surrounded_buildings_same_direction_side_A+surrounded_buildings_same_direction_side_B)))
    #surrounded buildings: opposite direction
    surrounded_buildings_opp_direction_side_A = math.floor((E_plus - B_plus - 2 + M_minus)*1.0/B_minus)
    surrounded_buildings_opp_direction_side_B = math.floor((E_minus - B_minus - 2 + M_plus)*1.0/B_minus)
    setups.append((4+2*surrounded_buildings_opp_direction_side_A+2*surrounded_buildings_opp_direction_side_B,
                   1*Fraction(2+surrounded_buildings_opp_direction_side_A+surrounded_buildings_opp_direction_side_B)))
    #optimized rows: beacons long way
    setups.append((2*math.ceil((1+math.ceil((E_plus-1)*1.0/M_minus))*1.0/math.ceil(B_plus*1.0/M_minus)),
                   1*Fraction(1, math.ceil(B_plus*1.0/M_minus))))
    #optimized rows: beacons short way
    setups.append((2*math.ceil((1+math.ceil((E_minus-1)*1.0/M_minus))*1.0/math.ceil(B_minus*1.0/M_minus)),
                   1*Fraction(1, math.ceil(B_minus*1.0/M_minus))))
    
    mask = [True]*4
    for i in range(4): #ew
        for j in range(4):
            if i!=j:
                if (setups[i][0] >= setups[j][0] and setups[i][1] > setups[j][1]) or (setups[i][0] > setups[j][0] and setups[i][1] >= setups[j][1]):
                    mask[j] = False
    filt_setups = []
    for i in range(4):
        if mask[i]:
            filt_setups.append(setups[i])

    return list(set(filt_setups))

def create_reference_list(uncompiled_constructs: Collection[UncompiledConstruct]) -> tuple[str, ...]:
    """Creates a reference list given a collection of UncompiledConstructs

    Parameters
    ----------
    uncompiled_constructs : Collection[UncompiledConstruct]
        Collection of UncompiledConstructs to create reference from

    Returns
    -------
    tuple[str, ...]
        A reference list containg every value of CompressedVector within
    """
    logging.info("Creating a reference list from a total of %d constructs.", len(uncompiled_constructs))
    reference_list = set()
    for construct in uncompiled_constructs:
        reference_list.update(set(construct.drain.keys()))
        reference_list.update(set(construct.deltas.keys()))
        reference_list.update(set(construct.base_inputs.keys()))
        reference_list.update(set(construct.cost.keys()))
        for val, _, _ in construct.allowed_modules:
            reference_list.add(val)
    reference_list = list(reference_list)
    reference_list.sort()
    logging.info("A total of %d items/fluids were found for the reference list.", len(reference_list))
    logging.debug(reference_list)
    return tuple(reference_list)

def determine_catalysts(uncompiled_construct_list: Collection[UncompiledConstruct], reference_list: tuple[str, ...]) -> tuple[str, ...]:
    """Determines the catalysts of a collection of UncompiledConstructs
    TODO: Detecting do nothing loops
    
    Parameters
    ----------
    uncompiled_construct_list : Collection[UncompiledConstruct]
        UncompiledConstructs to create catalysts from
    reference_list : tuple[str, ...]
        The universal reference list

    Returns
    -------
    tuple[str, ...]
        A list of catalyst items and fluids
    """
    logging.debug("Determining the catalysts present in a total of %d constructs.", len(uncompiled_construct_list))
    
    graph = {}
    for item in reference_list:
        graph[item] = set()
    for ident in [construct.ident for construct in uncompiled_construct_list]:
        graph[ident+"=construct"] = set()
        
    for construct in uncompiled_construct_list:
        for k, v in list(construct.deltas.items()) + list(construct.base_inputs.items()):
            if v > 0:
                graph[construct.ident+"=construct"].add(k)
            if v < 0:
                graph[k].add(construct.ident+"=construct")
    
    def all_descendants(node):
        descendants = copy.deepcopy(graph[node])
        length = 0
        while length!=len(descendants):
            length = len(descendants)
            old_descendants = copy.deepcopy(descendants)
            for desc in old_descendants:
                for new_desc in graph[desc]:
                    descendants.add(new_desc)
        return descendants
    
    catalyst_list = tuple([item for item in reference_list if item in all_descendants(item)])
    
    logging.debug("A total of %d catalysts were found.", len(catalyst_list))
    logging.debug("Catalysts found: %s", str(catalyst_list))
    return catalyst_list

def calculate_actives(reference_list: tuple[str, ...],catalyst_list: tuple[str, ...], data: dict) -> tuple[str, ...]:
    """Calculates all items that should be actively produce in a material factory. 
    Includes catalysts and any item that can be placed on the ground

    Parameters
    ----------
    reference_list : tuple[str, ...]
        The universal reference list
    catalyst_list : tuple[str, ...]
        The catalyst list
    data : dict
        The whole of data.raw

    Returns
    -------
    tuple[str, ...]
        Items that need to be actively produced in a material factory
    """    
    actives = set(copy.deepcopy(catalyst_list))

    for item in data['item'].values():
        if 'place_result' in item.keys() and item['name'] in reference_list:
            actives.add(item['name'])
    
    for module in data['module'].values():
        if module['name'] in reference_list:
            actives.add(module['name'])
        
    actives = list(actives)
    actives.sort()

    return tuple(actives)
