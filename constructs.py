from __future__ import annotations

from globalsandimports import *
from utils import *

if TYPE_CHECKING:
    from tools import FactorioInstance

class PointEvaluations(NamedTuple):
    """Evaluation matricies for a set of lookup table points

    Members
    -------
    multilinear_effect : np.ndarray
        Multilinear effect at each point
    running_cost : np.ndarray
        Running cost at each point
    evaulated_cost : np.ndarray
        Cost singular value at each point
    effective_area : np.ndarray
        Area used by machine at each point
    """    
    multilinear_effect: np.ndarray
    running_cost: np.ndarray
    evaulated_cost: np.ndarray
    effective_area: np.ndarray

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
    _column: np.ndarray
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
        self._column = np.zeros(len(instance.reference_list))
        for k, v in deltas.items():
            self._column[instance.reference_list.index(k)] = v
        self.limit = limit

    def column(self, known_technologies: TechnologicalLimitation) -> tuple[np.ndarray, float, str | None]:
        """Gets the column for this construct

        Parameters
        ----------
        known_technologies : TechnologicalLimitation
            Current tech level

        Returns
        -------
        tuple[np.ndarray, float, str | None]
            Effect column
            Cost
            Ident
        """        
        if self.limit >= known_technologies:
            return self._column, 1, None
        else:
            return np.zeros_like(self._column), 0, self.ident
        
    @staticmethod
    def columns(all_constructs: Collection[ManualConstruct], known_technologies: TechnologicalLimitation) -> ColumnTable:
        """Calculates the columns for all ManualConstructs

        Parameters
        ----------
        all_constructs : Collection[ManualConstruct]
            All ManualConstructs to calculate columns for
        known_technologies : TechnologicalLimitation
            Current tech level

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray[CompressedVector, Any]]
            Matrix of effect columns,
            Vector of costs,
            Matrix of exact costs,
            Ident columns
        """        
        construct_arr: np.ndarray[ManualConstruct, Any] = np.array(all_constructs, dtype=ManualConstruct)
        mask = np.array([known_technologies >= construct.limit for construct in construct_arr])
        construct_arr = construct_arr[mask]

        columns = np.vstack([construct.effect_vector for construct in construct_arr]).T
        costs = np.ones_like(construct_arr, dtype=np.float64)
        true_costs = np.vstack([np.zeros_like(construct.effect_vector) for construct in construct_arr]).T
        idents = np.concatenate([np.array([CompressedVector({construct.ident: 1})]) for construct in construct_arr])

        return ColumnTable(columns, costs, true_costs, idents)
    
    def __repr__(self) -> str:
        return str(self.ident)+\
                "\n\tWith a deltas of: "+str(self.deltas)+\
                "\n\tRequiring: "+str(self.limit)
                #"\n\tWith a vector of: "+str(self.effect_vector)+\

def beacon_designs(building_size: tuple[int, int], beacon: dict) -> list[tuple[str, tuple[Fraction, Fraction]]]:
    """Determines the possible optimal beacon designs for a building and beacon

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
    B_plus = int(max(beacon['tile_width'], beacon['tile_height']))
    B_minus = int(min(beacon['tile_width'], beacon['tile_height']))
    E_plus = int(beacon['supply_area_distance'])*2+B_plus
    E_minus = int(beacon['supply_area_distance'])*2+B_minus

    designs = []
    #surrounded buildings: same direction
    surrounded_buildings_same_direction_side_A = math.floor(float(np.ceil(E_plus/2) - np.ceil(B_plus/2) + M_plus - 1)/B_minus)
    surrounded_buildings_same_direction_side_B = math.floor(float(np.ceil(E_minus/2) - np.ceil(B_minus/2) + M_minus - 1)/B_minus)
    designs.append(("surrounded-beacons same-direction",
                    (4+2*surrounded_buildings_same_direction_side_A+2*surrounded_buildings_same_direction_side_B,
                     2+surrounded_buildings_same_direction_side_A+surrounded_buildings_same_direction_side_B)))
    #surrounded buildings: opposite direction
    surrounded_buildings_opp_direction_side_A = math.floor(float(np.ceil(E_plus/2) - np.ceil(B_plus/2) + M_minus - 1)/B_minus)
    surrounded_buildings_opp_direction_side_B = math.floor(float(np.ceil(E_minus/2) - np.ceil(B_minus/2) + M_plus - 1)/B_minus)
    designs.append(("surrounded-beacons opposite-direction",
                    (4+2*surrounded_buildings_opp_direction_side_A+2*surrounded_buildings_opp_direction_side_B,
                     1*2+surrounded_buildings_opp_direction_side_A+surrounded_buildings_opp_direction_side_B)))
    #efficient rows: beacons long way
    efficient_rows_long_way_D = int(M_minus * np.ceil(B_plus / M_minus) - B_plus)
    efficient_rows_long_way_LCM = int(np.lcm(M_minus, B_plus + efficient_rows_long_way_D))
    efficient_rows_long_way_sum = Fraction(np.array([np.floor((i*M_minus+M_minus+E_plus-2)/(B_plus + efficient_rows_long_way_D))-np.ceil(i*M_minus/(B_plus + efficient_rows_long_way_D))+1 for i in np.arange(efficient_rows_long_way_LCM)]).sum()/float(efficient_rows_long_way_LCM)).limit_denominator()
    designs.append(("efficient-rows long-way",
                    (efficient_rows_long_way_sum,
                     float(efficient_rows_long_way_LCM)/(B_plus + efficient_rows_long_way_D))))
    #efficient rows: beacons short way
    efficient_rows_short_way_D = int(M_minus * np.ceil(B_minus / M_minus) - B_minus)
    efficient_rows_short_way_LCM = int(np.lcm(M_minus, B_minus + efficient_rows_short_way_D))
    efficient_rows_short_way_sum = Fraction(np.array([np.floor((i*M_minus+M_minus+E_minus-2)/(B_minus + efficient_rows_short_way_D))-np.ceil(i*M_minus/(B_minus + efficient_rows_short_way_D))+1 for i in np.arange(efficient_rows_short_way_LCM)]).sum()/float(efficient_rows_short_way_LCM)).limit_denominator()
    designs.append(("efficient-rows short-way",
                    (efficient_rows_short_way_sum,
                     float(efficient_rows_short_way_LCM)/(B_minus + efficient_rows_short_way_D))))
    
    mask = [True]*4
    for i in range(4): #ew
        for j in range(4):
            if i!=j:
                if (designs[i][0][0] >= designs[j][0][0] and designs[i][0][1] < designs[j][0][1]) or (designs[i][0][0] < designs[j][0][0] and designs[i][0][1] >= designs[j][0][1]):
                    mask[j] = False
    filtered_designs = []
    for i in range(4):
        if mask[i]:
            filtered_designs.append(designs[i])

    return list(set(filtered_designs))

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

def determine_catalysts(uncompiled_construct_list: Collection[UncompiledConstruct], reference_list: Sequence[str]) -> tuple[str, ...]:
    """Determines the catalysts of a collection of UncompiledConstructs
    TODO: Detecting do nothing loops
    
    Parameters
    ----------
    uncompiled_construct_list : Collection[UncompiledConstruct]
        UncompiledConstructs to create catalysts from
    reference_list : Sequence[str]
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

def calculate_actives(reference_list: Sequence[str],catalyst_list: Sequence[str], data: dict) -> tuple[str, ...]:
    """Calculates all items that should be actively produce in a material factory. 
    Includes catalysts and any item that can be placed on the ground

    Parameters
    ----------
    reference_list : Sequence[str]
        The universal reference list
    catalyst_list : Sequence[str]
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
