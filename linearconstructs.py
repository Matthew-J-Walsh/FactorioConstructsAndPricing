from utils import *
from sparsetensors import *
import functools
from globalvalues import *

class Module:
    """
    Class for each module, contains all the information on a module that is needed for math.
    """
    def __init__(self, entry):
        self.name = entry['name']
        self.cost = {entry['name']: 1}
        self.effect_vector = np.array([(entry['effect'][e] if e in entry['effect'].keys() else 0) for e in MODULE_EFFECTS])
        self.limit = entry['limit']

class LinearConstructFamily:#family of linear constructs (usually just have different modules allowed)
    """
    A family of linear constructs made from a single UncompiledConstruct. Each construct group in the family
    represents a different research state. Currently doesn't have much of the functionality wanted.
    
    TODO notes: basically we are going to make a DAG that when we run get_constructs will return a cover of the DAG
    """
    def __init__(self, uncompiled_construct, universal_reference_list, catalyzing_deltas):
        assert isinstance(uncompiled_construct, UncompiledConstruct)
        assert isinstance(universal_reference_list, list)
        assert isinstance(catalyzing_deltas, list)

        self.uncompiled_construct = uncompiled_construct
        self.universal_reference_list = universal_reference_list
        self.catalyzing_deltas = catalyzing_deltas
        #base_modules = []
        #for mod, mcount in uncompiled_construct.allowed_modules:
        #    if mod['limit'] < uncompiled_construct.limit:
        #        base_modules.append((mod, mcount))
        #self.base_case = uncompiled_construct.true_linear_constructs(universal_reference_list, catalyzing_deltas, base_modules)
        #self.omni_case = uncompiled_construct.true_linear_constructs(universal_reference_list, catalyzing_deltas, uncompiled_construct.allowed_modules)
    
    def get_constructs(self, known_technologies):
        allowed_modules = []
        for mod, mcount in self.uncompiled_construct.allowed_modules:
            if mod['limit'] < known_technologies:
                allowed_modules.append((mod, mcount))
        return self.uncompiled_construct.true_constructs(self.universal_reference_list, self.catalyzing_deltas, allowed_modules)
        #return self.base_case
        
    def __repr__(self):
        return self.uncompiled_construct.__repr__()

class LinearConstruct:
    """
    A compiled construct, only linear allowed so no freezing needed.
    """
    def __init__(self, ident, vector, cost):
        assert isinstance(vector, SparseTensor)
        assert isinstance(cost, SparseTensor)

        self.ident = ident
        self.vector = vector
        self.cost = cost
        
    def __repr__(self):
        return str(self.ident)+\
                "\n\tWith a vector of: "+str(self.vector)+\
                "\n\tA Cost of: "+str(self.cost)

class UncompiledConstruct:
    """
    An uncompiled construct, contains all the information about the different ways a machine doing something (like running a recipe)
    can function.
    """
    def __init__(self, ident, drain, deltas, effect_effects, allowed_modules, base_inputs={}, cost={}, limit=None):
    
        assert isinstance(drain, dict)
        assert isinstance(deltas, dict)
        assert isinstance(effect_effects, dict)
        assert isinstance(allowed_modules, list)
        assert isinstance(base_inputs, dict)
        assert isinstance(cost, dict)
        assert isinstance(limit, TechnologicalLimitation)
        
        self.ident = ident
        self.drain = drain
        self.deltas = deltas
        self.effect_effects = effect_effects #dict of list of names of affected deltas
        self.allowed_modules = allowed_modules #list of tuples of allowed modules and maximum count. if the maximum counts are unequal the lowest is assumed to be
                                               #minimum in the actual machine and the rest are beacon only
        self.base_inputs = base_inputs #held for catalyst cost calcs later
        self.cost = cost
        self.limit = limit
        
    def __repr__(self):
        return str(self.ident)+\
                "\n\tWith a vector of: "+str(self.deltas)+\
                "\n\tAn added drain of: "+str(self.drain)+\
                "\n\tAn effect matrix of: "+str(self.effect_effects)+\
                "\n\tAllowed Modules: "+str([e[0]['name'] for e in self.allowed_modules])+\
                "\n\tBase Inputs of: "+str(self.base_inputs)+\
                "\n\tA Cost of: "+str(self.cost)+\
                "\n\tRequiring: "+str(self.limit)
    
    def true_constructs(self, universal_reference_list, catalyzing_deltas, permitted_modules):
        n = len(universal_reference_list)
        
        constructs = []
        for mod_set in integer_module_options(permitted_modules, min([c for _, c in self.allowed_modules])):
            logging.debug("Generating a linear construct for %s given module setup %s", self.ident, mod_set)
            effect_vector = np.zeros(len(MODULE_EFFECTS))
            for mod, count in mod_set:
                effect_vector += count * mod.effect_vector
            
            effect_vector = np.maximum(effect_vector, MODULE_EFFECT_MINIMUMS_NUMPY)
            
            effected_deltas = {}
            for item, count in self.deltas.items():
                for effect, effected in self.effect_effects.items():
                    if item in effected:
                        count *= (1+effect_vector[MODULE_EFFECTS.index(effect)])
                effected_deltas.update({item: count})
            effected_deltas = add_dicts(effected_deltas, self.drain)
            effected_vector = SparseTensor((n,))
            for item, delta in effected_deltas.items():
                effected_vector[universal_reference_list.index(item)] = delta

            effected_cost = add_dicts(self.cost, {mod.name: count for mod, count in mod_set})
            effected_cost_vector = SparseTensor((n,))
            for item, delta in effected_cost.items():
                effected_cost_vector[universal_reference_list.index(item)] = delta

            logging.debug("\tFound an vector of %s", effected_vector)
            logging.debug("\tFound an cost_vector of %s", effected_cost_vector)
            constructs.append(LinearConstruct(self.ident, effected_vector, effected_cost_vector))

        return constructs

def integer_module_options(modules, internal_limit):
    """
    Returns an iterator over the set of possible module setups

    [(item, count), (item, count), (item, count), ...]
    """
    if len(modules)==0:
        yield []
    for i in range(len(modules)):
        for j in range(modules[i][1]+1):
            for module_set in integer_module_options([(mod, count-j) for mod, count in modules if count-j>0], max(0, internal_limit-j) if modules[i][1]==internal_limit else internal_limit):
                yield [(modules[i][0], j)] + module_set

def list_of_ucs_by_key(list_of_ucs, key, value): #yields elements of list_of_ucs if it has an attribute with the value
    for d in list_of_ucs:
        if getattr(d, key)==value:
            yield d

def create_reference_list(uncompiled_construct_list):
    logging.debug("Creating a reference list for a total of %d constructs.", len(uncompiled_construct_list))
    reference_list = set()
    for construct in uncompiled_construct_list:
        for val in construct.drain.keys():
            reference_list.add(val)
        for val in construct.deltas.keys():
            reference_list.add(val)
        for val in construct.base_inputs.keys():
            reference_list.add(val)
        for val in construct.cost.keys():
            reference_list.add(val)
        for val, _ in construct.allowed_modules:
            reference_list.add(val['name'])
    reference_list = list(reference_list)
    reference_list.sort()
    logging.debug("A total of %d items/fluids were found for the reference list.", len(reference_list))
    logging.debug(reference_list)
    return reference_list

def determine_catalysts(uncompiled_construct_list):
    logging.debug("Determining the catalysts present in a total of %d constructs.", len(uncompiled_construct_list))
    
    graph = {}
    ref_list = create_reference_list(uncompiled_construct_list)
    for item in ref_list:
        graph.update({item: []})
    for ident in [construct.ident for construct in uncompiled_construct_list]:
        graph.update({ident+"-construct": []})
        
    def update_graph(c, v):
        if not c in FALSE_CATALYST_LINKS:
            if v>0:
                graph[construct.ident+"-construct"].append(c)
            if v<0:
                graph[c].append(construct.ident+"-construct")
    for construct in uncompiled_construct_list:
        for c, v in construct.deltas.items():
            update_graph(c, v)
        for c, v in construct.base_inputs.items():
            update_graph(c, v)
    
    def all_descendants(node):
        descendants = set([v for v in graph[node] if not (any([e in v for e in FALSE_CATALYST_METHODS[0]]) and any([e in v for e in FALSE_CATALYST_METHODS[1]]))])
        length = 0
        while length!=len(descendants):
            length = len(descendants)
            old_descendants = list(descendants)
            for desc in old_descendants:
                for grand in graph[desc]:
                    descendants.add(grand)
        return descendants
    
    catalyst_list = [item for item in ref_list if item in all_descendants(item)]
    
    logging.debug("A total of %d catalysts were found.", len(catalyst_list))
    logging.debug("Catalysts found: %s", str(catalyst_list))
    #for item in ref_list:
    #    logging.debug("%s:\n\t%s\n\t%s\n", item, str(graph[item]), str(all_descendants(item)))
    #for ident in [construct.ident for construct in uncompiled_construct_list]:
    #    logging.debug("%s:\n\t%s\n\t%s\n", ident+"-construct", str(graph[ident+"-construct"]), str(all_descendants(ident+"-construct")))
    return catalyst_list

def generate_all_construct_families(uncompiled_construct_list):
    logging.debug("Generating construct families from an uncompiled construct list of length %d.", len(uncompiled_construct_list))
    reference_list = create_reference_list(uncompiled_construct_list)
    catalysts_list = determine_catalysts(uncompiled_construct_list)
    return functools.reduce(lambda x, y: x+[ConstructFamily(y, reference_list, catalysts_list)], uncompiled_construct_list, []), reference_list

def generate_all_construct_families_linear(uncompiled_construct_list):
    logging.debug("Generating linear construct families from an uncompiled construct list of length %d.", len(uncompiled_construct_list))
    reference_list = create_reference_list(uncompiled_construct_list)
    catalysts_list = determine_catalysts(uncompiled_construct_list)
    return functools.reduce(lambda x, y: x+[LinearConstructFamily(y, reference_list, catalysts_list)], uncompiled_construct_list, []), reference_list

