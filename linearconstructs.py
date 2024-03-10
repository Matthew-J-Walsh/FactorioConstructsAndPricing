from __future__ import annotations

from utils import *
from globalsandimports import *

class LinearConstruct:
    """
    A compiled construct. Simply having its identifier, vector, cost, and limit

    Members
    -------
    ident:
        Unique identifier for the construct
    vector:
        Vector of the construct representing its action
    cost:
        The cost vector of the construct
    limit:
        TechnologicalLimitation of using the construct
    """
    ident: str
    vector: CompressedVector
    cost: CompressedVector
    limit: TechnologicalLimitation

    def __init__(self, ident: str, vector: CompressedVector, cost: CompressedVector, limit: TechnologicalLimitation) -> None:
        assert isinstance(ident, str)
        assert isinstance(vector, CompressedVector)
        assert isinstance(cost, CompressedVector)
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

class ConstructTransform:
    """
    List of constructs empowered with a reference list. Formed for more easy generation and manipulation of Matricies.

    Members
    -------
    constructs:
        List of the LinearConstructs making up this transform.
    reference_list:
        Reference list of items in the CompressedVectors.
    """
    constructs: list[LinearConstruct]
    reference_list: list[str]

    def __init__(self, constructs: list[LinearConstruct], reference_list: list[str]) -> None:
        self.constructs = constructs
        self.reference_list = reference_list
    
    def to_sparse(self):
        """
        TODO: May want sparse version for faster LP once we get consistent.
        """
        raise NotImplementedError
    
    def to_dense(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Forms a dense effect and cost Matrix from the ConstructTransform

        Returns
        -------
        effect:
            Linear transformation from amounts of constructs to total effect
        cost:
            Linear transformation from amounts of constructs to cost
        """
        effect = np.zeros((len(self.constructs), len(self.reference_list)), dtype=Fraction)
        cost = np.zeros((len(self.constructs), len(self.reference_list)), dtype=Fraction)
        for i in range(len(self.constructs)):
            for k, v in self.constructs[i].vector.items():
                effect[i, self.reference_list.index(k)] = v
                assert effect[i, self.reference_list.index(k)]==v
            for k, v in self.constructs[i].cost.items():
                cost[i, self.reference_list.index(k)] = v
                assert cost[i, self.reference_list.index(k)]==v

        assert (effect != 0).any()
        assert (cost != 0).any()
        return effect, cost

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
    base_productivity:
        https://lua-api.factorio.com/latest/prototypes/CraftingMachinePrototype.html#base_productivity
        https://lua-api.factorio.com/latest/prototypes/MiningDrillPrototype.html#base_productivity
    catalyzing_deltas:
        The catalyst list to use for added construct costs.
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
    base_productivity: Fraction
    catalyzing_deltas: list[str]

    def __init__(self, ident: str, drain: CompressedVector, deltas: CompressedVector, effect_effects: dict[str, list[str]], 
                 allowed_modules: list[tuple[str, int]], internal_module_limit: int, base_inputs: CompressedVector, cost: CompressedVector, 
                 limit: TechnologicalLimitation, base_productivity: Fraction = Fraction(0)) -> None:
        self.ident = ident
        self.drain = drain
        self.deltas = deltas
        self.effect_effects = effect_effects
        self.allowed_modules = allowed_modules
        self.internal_module_limit = internal_module_limit
        self.base_inputs = base_inputs
        self.cost = cost
        self.limit = limit
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

    def get_constructs(self, catalyzing_deltas: list[str], module_data: dict, max_external_mods: int = 0, 
                       beacon_multiplier: Fraction = Fraction(0)) -> list[LinearConstruct]:
        """
        Returns a list of LinearConstructs for all possible module setups of this UncompiledConstruct

        Parameters
        ---------
        catalyzing_deltas:
            List of catalysts to count in the cost.
        module_data:
            List of modules from data.raw.
        max_external_mods:
            External (beacon) module limit.
        beacon_multiplier:
            Multiplier for beacon-ed modules.

        Returns
        -------
        List of LinearConstructs
        """
        constructs = []
        logging.info("Creating LinearConstructs for "+self.ident)
        logging.info("\nFound a total of "+str(len(list(module_setup_generator(self.allowed_modules, self.internal_module_limit, max_external_mods))))+" mod setups")
        for mod_set in module_setup_generator(self.allowed_modules, self.internal_module_limit, max_external_mods):
            logging.debug("Generating a linear construct for %s given module setup %s", self.ident, mod_set)
            ident = self.ident + (" with module setup: " + " & ".join([str(v)+"x "+k for k, v in mod_set.items()]) if len(mod_set)>0 else "")

            limit = self.limit

            effect_vector = np.zeros(len(MODULE_EFFECTS))
            for mod, count in mod_set.items():
                mod_name, mod_region = mod.split("|")
                if mod_region=="i":
                    effect_vector += count * module_data[mod_name]['effect_vector'].astype(float)
                if mod_region=="e":
                    effect_vector += count * beacon_multiplier * module_data[mod_name]['effect_vector'].astype(float)
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

            effected_cost = self.cost + CompressedVector({mod.split("|")[0]: count for mod, count in mod_set.items()})
            for item in catalyzing_deltas:
                if item in self.base_inputs.keys():
                    effected_cost = effected_cost + CompressedVector({item: -1 * self.base_inputs[item]})

            logging.debug("\tFound an vector of %s", effected_deltas)
            logging.debug("\tFound an cost_vector of %s", effected_cost)
            constructs.append(LinearConstruct(ident, effected_deltas, effected_cost, limit))

        return constructs

def module_setup_generator(modules: list[tuple[str, bool, bool]], internal_limit: int, external_limit: int) -> Generator[CompressedVector, None, None]:
    """
    Returns an generator over the set of possible module setups.

    Parameters
    ---------
    modules:
        List of tuples representing avaiable modules and if its allowed in internal/external.
    internal_limit:
        The remaining number of internal module slots.
    external_limit:
        The remaining number of external module slots.
    
    Yields
    ------
    Integral CompressedVectors representing module setup options
    """
    if len(modules)==0:
        yield CompressedVector()
    else:
        internal_modules = [m for m, i, _ in modules if i]
        exteral_modules = [m for m, _, e in modules if e]

        for internal_mod_count in range(internal_limit+1):
            for external_mod_count in range(external_limit+1):
                for internal_mod_setup in itertools.combinations_with_replacement(internal_modules, internal_mod_count):
                    for external_mod_setup in itertools.combinations_with_replacement(exteral_modules, external_mod_count):
                        vector = CompressedVector()
                        for mod in internal_mod_setup:
                            vector += CompressedVector({mod+"|i": 1})
                        for mod in external_mod_setup:
                            vector += CompressedVector({mod+"|e": 1})
                        yield vector

def create_reference_list(uncompiled_construct_list: list[UncompiledConstruct]) -> list[str]:
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
    return reference_list

def determine_catalysts(uncompiled_construct_list: list[UncompiledConstruct], reference_list: list[str]) -> list[str]:
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
    
    catalyst_list = [item for item in reference_list if item in all_descendants(item)]
    
    logging.debug("A total of %d catalysts were found.", len(catalyst_list))
    logging.debug("Catalysts found: %s", str(catalyst_list))
    return catalyst_list

def generate_references_and_catalysts(uncompiled_construct_list: list[UncompiledConstruct]) -> tuple[list[str], list[str]]:
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

