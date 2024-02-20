from __future__ import annotations
import scipy as sp
import scipy.sparse
import numpy as np
import copy
from utils import *
import functools
import itertools
from globalvalues import *
from typing import Generator

class Module:
    """
    Class for each module, contains all the information on a module that is needed for math.

    Members
    -------
    name:
        Name of module
    cost:
        Cost vector of the module
    effect_vector:
        Numpy array of the module's effects
    limit:
        TechnologicalLimitation of the module
    """
    name: str
    cost: dict
    effect_vecotr: np.ndarray
    limit: TechnologicalLimitation

    def __init__(self, module: dict) -> None:
        """
        Parameters
        ----------
        module:
            A module instance. https://lua-api.factorio.com/latest/prototypes/ModulePrototype.html
        """
        self.name = module['name']
        self.cost = {module['name']: 1}
        self.effect_vector = np.array([(module['effect'][e] if e in module['effect'].keys() else 0) for e in MODULE_EFFECTS]) #https://lua-api.factorio.com/latest/prototypes/ModulePrototype.html#effect
        #https://lua-api.factorio.com/latest/types/Effect.html
        self.limit = module['limit']

class LinearConstructFamily:#family of linear constructs (usually just have different modules allowed)
    """
    A family of linear constructs made from a single UncompiledConstruct. Each construct group in the family
    represents a different research state. Currently doesn't have much of the functionality wanted.
    
    Members
    -------
    uncompiled_construct:
        The base UncompiledConstruct of the construct family
    universal_reference_list:
        The universal reference list to use
    catalyzing_deltas:
        The catalyst list to use for added construct costs

    TODO notes: basically we are going to make a DAG that when we run get_constructs will return a cover of the DAG
    """
    uncompiled_construct: UncompiledConstruct
    universal_reference_list: list[str]
    catalyzing_deltas: list[str]

    def __init__(self, uncompiled_construct, universal_reference_list, catalyzing_deltas) -> None:
        self.uncompiled_construct = uncompiled_construct
        self.universal_reference_list = universal_reference_list
        self.catalyzing_deltas = catalyzing_deltas
    
    def get_constructs(self, known_technologies: TechnologicalLimitation) -> list[LinearConstruct]:
        allowed_modules = []
        for mod, mcount in self.uncompiled_construct.allowed_modules:
            if mod['limit'] < known_technologies:
                allowed_modules.append((mod, mcount))
        return self.uncompiled_construct.true_constructs(self.universal_reference_list, self.catalyzing_deltas, allowed_modules)
        
    def __repr__(self) -> str:
        return self.uncompiled_construct.__repr__()

class LinearConstruct:
    """
    A compiled construct. Simply having its identifier, vector, and cost

    Members
    -------
    ident:
        Unique identifier for the construct
    vector:
        Vector of the construct representing its action
    cost:
        The cost vector of the construct
    """
    ident: str
    vector: sp.sparray
    cost: sp.sparray

    def __init__(self, ident: str, vector: sp.sparray, cost: sp.sparray) -> None:
        self.ident = ident
        self.vector = vector
        self.cost = cost
        
    def __repr__(self) -> str:
        return str(self.ident)+\
                "\n\tWith a vector of: "+str(self.vector)+\
                "\n\tA Cost of: "+str(self.cost)

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
        Number of module slots inside construct
    base_inputs:
        The base inputs for future catalyst calculations.
    cost:
        The cost of a single instance. (without any modules)
    limit:
        TechnologicalLimitation to make this construct. (without any modules)
    base_productivity:
        https://lua-api.factorio.com/latest/prototypes/CraftingMachinePrototype.html#base_productivity
        https://lua-api.factorio.com/latest/prototypes/MiningDrillPrototype.html#base_productivity
    """
    ident: str
    drain: CompressedVector
    deltas: CompressedVector
    effect_effects: dict[str, list]
    allowed_modules: list[tuple[str, bool, bool]]
    internal_module_limit: int
    base_inputs: CompressedVector
    cost: CompressedVector
    limit: TechnologicalLimitation
    base_productivity: Fraction

    def __init__(self, ident: str, drain: CompressedVector, deltas: CompressedVector, effect_effects: dict[str, list], 
                 allowed_modules: list[tuple[str, int]], internal_module_limit: int, base_inputs: CompressedVector, cost: CompressedVector, 
                 limit: TechnologicalLimitation, base_productivity: Fraction = 0) -> None:
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
                "\n\tAllowed Modules: "+str([e[0]['name'] for e in self.allowed_modules])+\
                "\n\tBase Productivity: "+str(self.base_productivity)+\
                "\n\tBase Inputs of: "+str(self.base_inputs)+\
                "\n\tA Cost of: "+str(self.cost)+\
                "\n\tRequiring: "+str(self.limit)
    
    def true_constructs(self, universal_reference_list: list[str], catalyzing_deltas: list[str], permitted_modules: list[tuple[str, bool, bool]], 
                        max_internal_mods: int, max_external_mods: int, beacon_multiplier: Fraction) -> list[LinearConstruct]:
        """
        Returns a list of LinearConstructs for all possible module setups of this UncompiledConstruct

        Parameters
        ---------
        universal_reference_list:
            The universal reference list for vector orderings
        catalyzing_deltas:
            List of catalysts to count in the cost
        permitted_modules:
            List of allowed modules and if they are allow inside the construct or only beaconed
        max_internal_mods:
            Internal module limit.
        max_external_mods:
            External (beacon) module limit.
        beacon_multiplier:
            Multiplier for beacon-ed modules.

        Returns
        -------
        List of LinearConstructs
        """
        n = len(universal_reference_list)
        
        constructs = []
        for mod_set in module_setup_generator(permitted_modules, max_internal_mods, max_external_mods):
            logging.debug("Generating a linear construct for %s given module setup %s", self.ident, mod_set)
            effect_vector = np.zeros(len(MODULE_EFFECTS))
            for mod, count in mod_set.items():
                mod_name, mod_region = mod.split("|")
                if mod_region=="i":
                    effect_vector += count * MODULE_REFERENCE[mod_name].effect_vector
                if mod_region=="e":
                    effect_vector += count * beacon_multiplier * MODULE_REFERENCE[mod_name].effect_vector
            effect_vector[MODULE_EFFECTS.index('productivity')] += self.base_productivity
            
            effect_vector = np.maximum(effect_vector, MODULE_EFFECT_MINIMUMS_NUMPY)
            
            effected_deltas = CompressedVector()
            for item, count in self.deltas.items():
                for effect, effected in self.effect_effects.items():
                    if item in effected:
                        count *= (1+effect_vector[MODULE_EFFECTS.index(effect)])
                effected_deltas.update({item: count})
            effected_deltas = effected_deltas + self.drain
            effected_vector = sp.sparse.csr_array((n, 1), dtype=Fraction)
            for item, delta in effected_deltas.items():
                effected_vector[universal_reference_list.index(item), 0] = delta

            effected_cost = self.cost + CompressedVector({mod.name: count for mod, count in mod_set})
            for item in catalyzing_deltas:
                if item in self.base_inputs.keys():
                    effected_cost = effected_cost + CompressedVector({item: self.base_inputs[item]})
            effected_cost_vector = sp.sparse.csr_array((n, 1), dtype=Fraction)
            for item, delta in effected_cost.items():
                effected_cost_vector[universal_reference_list.index(item), 0] = delta

            logging.debug("\tFound an vector of %s", effected_vector)
            logging.debug("\tFound an cost_vector of %s", effected_cost_vector)
            constructs.append(LinearConstruct(self.ident, effected_vector, effected_cost_vector))

        return constructs

def module_setup_generator(modules: list[tuple[str, bool, bool]], internal_limit: int, external_limit: int) -> Generator[CompressedVector, None, None]:
    """
    Returns an generator over the set of possible module setups

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
    internal_modules = [m for m, i, e in modules if i]
    exteral_modules = [m for m, i, e in modules if e]

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
    logging.debug("Creating a reference list for a total of %d constructs.", len(uncompiled_construct_list))
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
    logging.debug("A total of %d items/fluids were found for the reference list.", len(reference_list))
    logging.debug(reference_list)
    return reference_list

def determine_catalysts(uncompiled_construct_list: list[UncompiledConstruct], universal_reference_list: list[str]) -> list[str]:
    """
    Determines the catalysts of a list of UncompiledConstructs.
    Uses FALSE_CATALYST_METHODS and FALSE_CATALYST_LINKS to block facile catalysts.

    Parameters
    ---------
    uncompiled_construct_list:
        List of UncompiledConstructs.
    universal_reference_list:
        The universal reference list for vector orderings

    Returns
    -------
    A list of catalyst items and fluids
    """
    logging.debug("Determining the catalysts present in a total of %d constructs.", len(uncompiled_construct_list))
    
    graph = {}
    for item in universal_reference_list:
        graph[item] = set()
    for ident in [construct.ident for construct in uncompiled_construct_list]:
        graph[ident+"-construct"] = set()
        
    for construct in uncompiled_construct_list:
        for k, v in list(construct.deltas.items()) + list(construct.base_inputs.items()):
            if v>0:
                graph[construct.ident+"-construct"].add(k)
            if v<0:
                graph[k].add(construct.ident+"-construct")
    
    def all_descendants(node):
        descendants = copy.deepcopy(graph[node])
        length = 0
        while length!=len(descendants):
            length = len(descendants)
            old_descendants = copy.deepcopy(descendants)
            for desc in old_descendants: #TODO: Make faster?
                for new_desc in graph[desc]:
                    descendants.add(new_desc)
        return descendants
    
    catalyst_list = [item for item in universal_reference_list if item in all_descendants(item)]
    
    logging.debug("A total of %d catalysts were found.", len(catalyst_list))
    logging.debug("Catalysts found: %s", str(catalyst_list))
    return catalyst_list

def generate_all_construct_families(uncompiled_construct_list: list[UncompiledConstruct]) -> tuple[list[LinearConstructFamily], list[str], list[str]]:
    """
    Generates the reference list, catalyst list, and all the construct families for a list of uncompiled constructs.

    Parameters
    ----------
    uncompiled_construct_list:
        List of UncompiledConstructs.

    Returns
    -------
    linear_construct_families:
        List of LinearConstructFamilys,
    reference_list:
        A reference list containing every relevent value.
    catalyst_list:
        A subset of the reference_list that contains catalytic items/fluids.
    """
    logging.debug("Generating linear construct families from an uncompiled construct list of length %d.", len(uncompiled_construct_list))
    reference_list = create_reference_list(uncompiled_construct_list)
    catalyst_list = determine_catalysts(uncompiled_construct_list, reference_list)
    return [LinearConstructFamily(construct, reference_list, catalyst_list) for construct in uncompiled_construct_list], reference_list, catalyst_list

