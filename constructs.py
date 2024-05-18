from __future__ import annotations

from globalsandimports import *
from utils import *

class LinearConstruct:
    """
    A compiled construct. Simply having its identifier, vector, cost, and limit.
    Has a Transformation from a count to an Effect
    Has a Transformation from a count to a Cost

    Members
    -------
    ident:
        Unique identifier for the construct
    vector:
        The action vector of the construct
    cost:
        The cost vector of the construct
    limit:
        TechnologicalLimitation of using the construct
    """
    ident: str
    vector: sparse.coo_array
    cost: sparse.coo_array
    limit: TechnologicalLimitation

    def __init__(self, ident: str, vector: sparse.coo_array, cost: sparse.coo_array, limit: TechnologicalLimitation) -> None:
        assert isinstance(ident, str)
        assert isinstance(vector, sparse.coo_array)
        assert isinstance(cost, sparse.coo_array)
        assert isinstance(limit, TechnologicalLimitation)
        self.ident = ident
        self.vector = vector
        self.cost = cost
        self.limit = limit
        
    def __repr__(self) -> str:
        return str(self.ident)+\
                "\n\tWith a vector of: "+str(self.vector)+\
                "\n\tA Cost of: "+str(self.cost)+\
                "\n\tLimit of: "+str(self.limit)

class UncompiledConstruct:
    """
    An uncompiled construct, contains all the information about the different ways a machine doing something (like running a recipe)
    can function.

    Members
    -------
    ident:
        Unique identifier.
    drain:
        Passive drain of a single instance.
    deltas:
        Changes to product environment from running construct.
    effect_effects:
        Dict of list of (module effect) names of affected deltas.
        Specifies how this construct is affected by modules.
    allowed_modules:
        List of tuples, each tuple representing a module and the amount of said module that is allowed.
    internal_module_limit
        Number of module slots inside construct.
    base_inputs:
        The base inputs for future catalyst calculations.
    cost:
        The cost of a single instance. (without any modules)
    limit:
        TechnologicalLimitation to make this construct. (without any modules)
    building:
        Link the the building entity for tile size values
    base_productivity:
        https://lua-api.factorio.com/latest/prototypes/CraftingMachinePrototype.html#base_productivity
        https://lua-api.factorio.com/latest/prototypes/MiningDrillPrototype.html#base_productivity
    """
    ident: str
    drain: CompressedVector
    deltas: CompressedVector
    effect_effects: dict[str, list[str]]
    allowed_modules: list[tuple[str, bool, bool]]
    internal_module_limit: int
    base_inputs: CompressedVector
    cost: CompressedVector
    limit: TechnologicalLimitation
    building: dict
    base_productivity: Fraction

    def __init__(self, ident: str, drain: CompressedVector, deltas: CompressedVector, effect_effects: dict[str, list[str]], 
                 allowed_modules: list[tuple[str, bool, bool]], internal_module_limit: int, base_inputs: CompressedVector, cost: CompressedVector, 
                 limit: TechnologicalLimitation, building: dict, base_productivity: Fraction = Fraction(0)) -> None:
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
        
    def __repr__(self) -> str:
        return str(self.ident)+\
                "\n\tWith a vector of: "+str(self.deltas)+\
                "\n\tAn added drain of: "+str(self.drain)+\
                "\n\tAn effect matrix of: "+str(self.effect_effects)+\
                "\n\tAllowed Modules: "+str(self.allowed_modules)+\
                "\n\tBase Productivity: "+str(self.base_productivity)+\
                "\n\tBase Inputs of: "+str(self.base_inputs)+\
                "\n\tA Cost of: "+str(self.cost)+\
                "\n\tRequiring: "+str(self.limit)
    
    def compile_to_lambda(self, catalyzing_deltas: tuple[str, ...], module_data: dict, reference_list: tuple[str, ...], beacons: list[dict]) -> Callable[[TechnologicalLimitation], Callable[[np.ndarray], Tuple[sparse.csr_array, sparse.csr_array, str]]]:
        """
        Returns a nested function.
        First function takes a TechnologicalLimitation and outputs a Second function.
        Second function takes a pricing model and returns the best possible column, its cost column, and its recovery name.
        """
        raise NotImplementedError("TODO")
        return None

    def compile(self, catalyzing_deltas: tuple[str, ...], module_data: dict, reference_list: tuple[str, ...], beacons: list[dict]) -> None:
        """
        Returns a SingularConstruct of this UncompiledConstruct

        Parameters
        ---------
        catalyzing_deltas:
            List of catalysts to count in the cost.
        module_data:
            List of modules from data.raw.
        reference_list:
            Universal reference list to use for value orderings.
        beacons:
            List of beacon dicts from data.raw.

        Returns
        -------
        A ModulatedConstruct.
        """
        raise DeprecationWarning("Don't use anymore")
        constructs: list[LinearConstruct] = []
        logging.info("Creating LinearConstructs for "+self.ident)
        logging.info("\nFound a total of "+str(sum([len(list(module_setup_generator(self.allowed_modules, self.internal_module_limit, (self.building['tile_width'], self.building['tile_height']), beacon))) for beacon in [None]+beacons]))+" mod setups")
        for beacon in [None]+beacons: #None=No beacons
            for module_set, beacon_cost, module_cost in module_setup_generator(self.allowed_modules, self.internal_module_limit, (self.building['tile_width'], self.building['tile_height']), beacon):
                logging.debug("Generating a linear construct for %s given module setup %s given module costs %s given beacon cost %s", self.ident, module_set, module_cost, beacon_cost)
                ident = self.ident + (" with module setup: " + " & ".join([str(v)+"x "+k for k, v in module_set.items()]) if len(module_set)>0 else "")

                limit = self.limit + (beacon['limit'] if isinstance(beacon, dict) else TechnologicalLimitation([]))

                effect_vector = np.zeros(len(MODULE_EFFECTS))
                for mod, count in module_set.items():
                    mod_name, mod_region = mod.split("|")
                    if mod_region=="i":
                        effect_vector += count * module_data[mod_name]['effect_vector'].astype(float)
                    if mod_region=="e":
                        assert beacon is not None
                        effect_vector += count * beacon['distribution_effectivity'] * module_data[mod_name]['effect_vector'].astype(float)
                    limit = limit + module_data[mod.split("|")[0]]['limit']
                effect_vector[MODULE_EFFECTS.index('productivity')] += float(self.base_productivity)
                
                effect_vector = np.maximum(effect_vector, MODULE_EFFECT_MINIMUMS_NUMPY.astype(float))
                
                effected_deltas = CompressedVector()
                for item, count in self.deltas.items():
                    for effect, effected in self.effect_effects.items():
                        if item in effected:
                            count *= 1 + effect_vector[MODULE_EFFECTS.index(effect)]
                    effected_deltas[item] = count
                effected_deltas = effected_deltas + self.drain

                effected_cost = self.cost + beacon_cost
                for mod, count in module_cost.items():
                    effected_cost = effected_cost + CompressedVector({mod.split("|")[0]: count})
                for item in catalyzing_deltas:
                    if item in self.base_inputs.keys():
                        effected_cost = effected_cost + CompressedVector({item: -1 * self.base_inputs[item]})

                assert all([v > 0 for v in effected_cost.values()]), effected_cost
                sparse_deltas = sparse.coo_array(([e for e in effected_deltas.values()], ([reference_list.index(d) for d in effected_deltas.keys()], [0 for _ in effected_deltas])), shape=(len(reference_list),1), dtype=np.longdouble)
                sparse_cost = sparse.coo_array(([e for e in effected_cost.values()], ([reference_list.index(d) for d in effected_cost.keys()], [0 for _ in effected_cost])), shape=(len(reference_list),1), dtype=np.longdouble)

                logging.debug("\tFound an vector of %s", effected_deltas)
                logging.debug("\tFound an cost_vector of %s", effected_cost)
                constructs.append(LinearConstruct(ident, sparse_deltas, sparse_cost, limit))

        return SingularConstruct(ModulatedConstruct(constructs, self.ident))

def module_setup_generator(modules: list[tuple[str, bool, bool]], internal_limit: int, building_size: tuple[int, int], beacon: dict | None = None) -> Generator[tuple[CompressedVector, CompressedVector, CompressedVector], None, None]:
    """
    Returns an generator over the set of possible module setups.

    Parameters
    ---------
    modules:
        List of tuples representing avaiable modules and if its allowed in internal/external.
    internal_limit:
        The remaining number of internal module slots.
    building_size:
        Size of building's tile.
    beacon:
        Beacon being used.
    
    Yields
    ------
    vector:
        Integral CompressedVectors representing module setup options
    beacon_cost:
        CompressedVector with keys of beacons and values of beacon count per building
    external_module_cost:
        CompressedVector with keys of modules and values of module count per building
    """
    if len(modules)==0 or DEBUG_BLOCK_MODULES:
        yield CompressedVector(), CompressedVector(), CompressedVector()
    else:
        internal_modules = [m for m, i, _ in modules if i]
        external_modules = [m for m, _, e in modules if e]

        if not beacon is None and not DEBUG_BLOCK_BEACONS:
            for beacon_count, beacon_cost in beacon_setups(building_size, beacon):
                for internal_mod_count in range(internal_limit+1):
                    for internal_mod_setup in itertools.combinations_with_replacement(internal_modules, internal_mod_count):
                        #for external_mod_setup in itertools.combinations_with_replacement(exteral_modules, beacon_count*beacon['module_specification']['module_slots']): #too big
                        for external_mod in external_modules:
                            vector = CompressedVector()
                            for mod in internal_mod_setup:
                                vector += CompressedVector({mod+"|i": 1})
                            yield vector + CompressedVector({external_mod+"|e": beacon_count * beacon['module_specification']['module_slots']}), \
                                  CompressedVector({beacon['name']: -1 * beacon_cost}), \
                                  vector + CompressedVector({external_mod+"|e": -1 * beacon_cost * beacon['module_specification']['module_slots']}), 
        else:
            for internal_mod_count in range(internal_limit+1):
                for internal_mod_setup in itertools.combinations_with_replacement(internal_modules, internal_mod_count):
                    for external_mod in external_modules:
                        vector = CompressedVector()
                        for mod in internal_mod_setup:
                            vector += CompressedVector({mod+"|i": 1})
                        yield vector, CompressedVector(), vector

def create_reference_list(uncompiled_construct_list: tuple[UncompiledConstruct, ...]) -> tuple[str, ...]:
    """
    Creates a reference list given a list of UncompiledConstructs

    Parameters
    ---------
    uncompiled_construct_list:
        List of UncompiledConstructs.
    
    Returns
    -------
    A reference list containg every value of CompressedVector within.
    """
    logging.info("Creating a reference list for a total of %d constructs.", len(uncompiled_construct_list))
    reference_list = set()
    for construct in uncompiled_construct_list:
        reference_list = reference_list.union(set(construct.drain.keys()))
        reference_list = reference_list.union(set(construct.deltas.keys()))
        reference_list = reference_list.union(set(construct.base_inputs.keys()))
        reference_list = reference_list.union(set(construct.cost.keys()))
        for val, _, _ in construct.allowed_modules:
            reference_list.add(val)
    reference_list = list(reference_list)
    reference_list.sort()
    logging.info("A total of %d items/fluids were found for the reference list.", len(reference_list))
    logging.debug(reference_list)
    return tuple(reference_list)

def determine_catalysts(uncompiled_construct_list: tuple[UncompiledConstruct, ...], reference_list: tuple[str, ...]) -> tuple[str, ...]:
    """
    Determines the catalysts of a list of UncompiledConstructs.
    TODO: Detecting do nothing loops.

    Parameters
    ---------
    uncompiled_construct_list:
        List of UncompiledConstructs.
    reference_list:
        The universal reference list for vector orderings

    Returns
    -------
    A list of catalyst items and fluids
    """
    logging.debug("Determining the catalysts present in a total of %d constructs.", len(uncompiled_construct_list))
    
    graph = {}
    for item in reference_list:
        graph[item] = set()
    for ident in [construct.ident for construct in uncompiled_construct_list]:
        graph[ident+"-construct"] = set()
        
    for construct in uncompiled_construct_list:
        for k, v in list(construct.deltas.items()) + list(construct.base_inputs.items()):
            if v > 0:
                graph[construct.ident+"-construct"].add(k)
            if v < 0:
                graph[k].add(construct.ident+"-construct")
    
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

def generate_references_and_catalysts(uncompiled_construct_list: tuple[UncompiledConstruct, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """
    Generates the reference list, catalyst list, and all the construct families for a list of uncompiled constructs.

    Parameters
    ----------
    uncompiled_construct_list:
        List of UncompiledConstructs.

    Returns
    -------
    reference_list:
        A reference list containing every relevent value.
    catalyst_list:
        A subset of the reference_list that contains catalytic items/fluids.
    """
    logging.info("Generating linear construct families from an uncompiled construct list of length %d.", len(uncompiled_construct_list))
    reference_list = create_reference_list(uncompiled_construct_list)
    catalyst_list = determine_catalysts(uncompiled_construct_list, reference_list)
    return reference_list, catalyst_list

