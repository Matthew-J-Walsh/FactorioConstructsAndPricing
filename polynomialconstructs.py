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

class ConstructFamily: #family of constructs (usually just have different modules allowed)
    """
    A family of compiled constructs made from a single UncompiledConstruct. Each construct in the family
    represents a different research state. Currently doesn't have much of the functionality wanted.
    
    TODO notes: basically we are going to make a DAG that when we run get_constructs will return a cover of the DAG
    """
    def __init__(self, uncompiled_construct, universal_reference_list, catalyzing_deltas):
        self.uncompiled_construct = uncompiled_construct
        self.universal_reference_list = universal_reference_list
        self.catalyzing_deltas = catalyzing_deltas
        base_modules = []
        for mod, mcount in uncompiled_construct.allowed_modules:
            if mod['limit'] < uncompiled_construct.limit:
                base_modules.append((mod, mcount))
        self.base_case = uncompiled_construct.true_construct(universal_reference_list, catalyzing_deltas, base_modules)
        self.omni_case = uncompiled_construct.true_construct(universal_reference_list, catalyzing_deltas, uncompiled_construct.allowed_modules)
    
    def get_construct(self, known_technologies):
        allowed_modules = []
        for mod, mcount in self.uncompiled_construct.allowed_modules:
            if mod['limit'] < known_technologies:
                allowed_modules.append((mod, mcount))
        return self.uncompiled_construct.true_construct(self.universal_reference_list, self.catalyzing_deltas, allowed_modules)
        #return self.base_case
        
    def __repr__(self):
        return self.uncompiled_construct.__repr__()

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
        return self.uncompiled_construct.true_linear_constructs(self.universal_reference_list, self.catalyzing_deltas, allowed_modules)
        #return self.base_case
        
    def __repr__(self):
        return self.uncompiled_construct.__repr__()


class Construct:
    """
    A Construct as defined in "Factorio Stalling, Management, and Constructs".
    """
    def __init__(self, ident, phi_tensors, cost_tensors, theta_tensors, effect_tensors, limit):
        for ten in phi_tensors:
            assert isinstance(ten, SparseTensor), ten
        for ten in cost_tensors:
            assert isinstance(ten, SparseTensor), ten
        for ten in theta_tensors:
            assert isinstance(ten, SparseTensor), ten
        for ten in effect_tensors:
            assert isinstance(ten, SparseTensor), ten
        
        
        self.ident = ident #some sort of identifier for naming and displays
        self.phi_tensors = phi_tensors #usually just linear and translation
        self.cost_tensors = cost_tensors #usually just linear and translation
        self.theta_tensors = theta_tensors #2+RELEVENT_MODULE_EFFECTS total tensors, index order is h,d,b,b,...
        self.effect_tensors = effect_tensors #2+RELEVENT_MODULE_EFFECTS total tensors, index order is n,d,b,b,...
        self.limit = limit #tech stuff

    def __mul__(self, other): #multiply by scalar
        return Construct(self.ident, 
                         [e for e in self.phi_tensors],
                         [e*other for e in self.cost_tensors],
                         [e for e in self.theta_tensors],
                         [e*other for e in self.effect_tensors],
                         limit=self.limit)

    def __rmul__(self, other): #multiply by scalar
        return self * other
    
    def dump_shapes(self):
        logging.info("%s with phi shapes: %s, cost shapes: %s, theta shapes: %s, and effect shapes: %s", self.ident,
                     str([ten.shape for ten in self.phi_tensors]),
                     str([ten.shape for ten in self.cost_tensors]),
                     str([ten.shape for ten in self.theta_tensors]),
                     str([ten.shape for ten in self.effect_tensors]))
    
    def __repr__(self):
        return str(self.ident)+"\nRequiring: "+\
                " or\n".join([" and ".join([y['name'] for y in x]) for x in self.limit])
    
    def freeze(self, phi):
        cost = self.cost_tensors[0].to_dense() + einstein_summation('ij,j', self.cost_tensors[1], phi)
        tts = [self.theta_tensors[0], self.theta_tensors[1]]
        for i in range(2,len(self.theta_tensors)):
            tts[1].add(slice(None,None,None),
                       einstein_summation(''.join([chr(ord('i')+j) for j in range(i+1)])+','+','.join([chr(ord('k')+j) for j in range(i-1)]), 
                                          self.theta_tensors[i], *([phi]*(i-1)), stay_sparse=True))
        ets = [self.effect_tensors[0], self.effect_tensors[1]]
        for i in range(2,len(self.effect_tensors)):
            ets[1].add(slice(None,None,None),
                       einstein_summation(''.join([chr(ord('i')+j) for j in range(i+1)])+','+','.join([chr(ord('k')+j) for j in range(i-1)]), 
                                          self.effect_tensors[i], *([phi]*(i-1)), stay_sparse=True))
        return FrozenConstruct(self.ident, cost, tts, ets)

class FrozenConstruct: #commonly called 'R's
    """
    A construct after a module setup has been supplied.
    """
    def __init__(self, ident, cost, theta_tensors, effect_tensors):
        assert len(theta_tensors)==2
        assert len(effect_tensors)==2
        assert isinstance(cost, np.ndarray)
        for ten in theta_tensors:
            assert isinstance(ten, SparseTensor)
        for ten in effect_tensors:
            assert isinstance(ten, SparseTensor)
        
        self.ident = ident #some sort of identifier for naming and displays
        self.cost = cost #just linear at this point
        self.theta_tensors = theta_tensors #2 tensors, index order is h,d
        self.effect_tensors = effect_tensors #2 tensors, index order is n,d
    
    def __mul__(self, other): #multiply by scalar
        return FrozenConstruct(self.ident, 
                               self.cost*other,
                               [e for e in self.theta_tensors],
                               [e*other for e in self.effect_tensors])

    def __rmul__(self, other): #multiply by scalar
        return self * other
    
    def apply(self, theta):
        bounds = self.theta_tensors[0].to_dense() + einstein_summation('ij,j', self.theta_tensors[1], theta)
        effect = self.effect_tensors[0].to_dense() + einstein_summation('ij,j', self.effect_tensors[1], theta)
        return effect, np.all(np.logical_or(bounds<=0,np.isclose(bounds,np.zeros_like(bounds))))

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
    
    def true_linear_constructs(self, universal_reference_list, catalyzing_deltas, permitted_modules):
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

    def true_construct(self, universal_reference_list, catalyzing_deltas, permitted_modules):
        """
        Makes a Construct from this UncompiledConstruct based on the universal reference list's item ordering,
        catalysts, and allowed modules. permitted_modules must be a subset of self.allowed_modules.
        
        This isn't just the Construct.__init__() function because there are other ways of making a construct.
        """
        n = len(universal_reference_list)
        
        b = len(permitted_modules)
        theta_tensors = [SparseTensor([2])]
        for i in range(2+len(MODULE_EFFECTS)):
            theta_tensors.append(SparseTensor([2, 1]+[b]*i))
        theta_tensors[0][1] = -1 #now [0,-1]
        theta_tensors[1][0, 0] = -1 #giving us -t<=0
        theta_tensors[1][1, 0] = 1 #giving us t+(-1)<=0
        
        logging.debug("Starting compilation of a %s construct, n=%d and b=%d.\n\tThere are %d different modules", self.ident, n, b, len(permitted_modules))
        
        #for phi we need:
        #1 half shell per allowed module (amount>=0)
        #1 half shell to cut maximum amount inclduing beacons
        #1 half shell per allowed module that isnt at beacon maximum, we could do for all but the ones that are at beacon maximum are already subsumed
        #1 half shell per effect for effect restrictions
        #modmax = max(map(lambda i: i[1], permitted_modules))
        #modmaxcount = sum(i[1] == modmax for i in permitted_modules)
        #modsubmaxcount = sum(i[1] <= modmax for i in permitted_modules)
        phi_half_shell_count = 2 * b + 1 + len(MODULE_EFFECTS)
        phi_tensors = [SparseTensor([phi_half_shell_count]), 
                       SparseTensor([phi_half_shell_count, b])]
        i_p = 0
        while (i_p < b):
            phi_tensors[1][i_p, i_p] = -1 #amount>=0
            i_p += 1
        if len(permitted_modules)>0:
            phi_tensors[0][i_p] = -1 * max(map(lambda i: i[1], permitted_modules))
            phi_tensors[1][i_p, :] = 1 #maximum cut + beacons
            i_p += 1
            while (i_p < 2 * b + 1):
                phi_tensors[0][i_p] = -1 * permitted_modules[i_p - b - 1][1]
                phi_tensors[1][i_p, i_p - b - 1] = 1
                i_p += 1
            while (i_p < 2 * b + 1 + len(MODULE_EFFECTS)): #for loop me!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                effect = MODULE_EFFECTS[i_p - 2 * b - 1]
                phi_tensors[0][i_p] = MODULE_EFFECT_MINIMUMS[effect] - 1
                phi_tensors[1][i_p, :] = -1 * np.array([[(mod['effect'][effect]['bonus'] if effect in mod['effect'].keys() else 0) for mod, mcount in permitted_modules]])
                i_p += 1
        
        
        cost_tensors = [SparseTensor([n]), 
                        SparseTensor([n, b])]
        for k, v in self.cost.items():
            cost_tensors[0][universal_reference_list.index(k)] = v
        for i in range(b):
            cost_tensors[1][universal_reference_list.index(permitted_modules[i][0]['name']), i] = 1
        
        
        effect_tensors = [SparseTensor([n])]
        for i in range(2+len(MODULE_EFFECTS)):
            effect_tensors.append(SparseTensor([n, 1]+[b]*i))
            
        for k, v in self.drain.items():
            effect_tensors[0][universal_reference_list.index(k)] = v
        for k, v in self.deltas.items():
            effect_tensors[1][universal_reference_list.index(k), 0] = v
        module_vectors = {}
        for effect in MODULE_EFFECTS:
            module_vectors.update({effect: np.array([(mod['effect'][effect]['bonus'] if effect in mod['effect'].keys() else 0) for mod, mcount in permitted_modules])})
        #now the fun begins:
        for i in range(2, 2+len(MODULE_EFFECTS)): #choosing a polynomial level
            for k, v in self.deltas.items(): #choosing an item/fluid
                sub_tensor = np.zeros([b]*(i-1))
                for effs in itertools.combinations(MODULE_EFFECTS, i-1): #effect combinations
                    if all([k in self.effect_effects[eff] for eff in effs]): #all effects in this combo actually effect
                        sub_tensor += v * functools.reduce(lambda x, y: np.tensordot(x, y, 0), [module_vectors[eff] for eff in effs])
                effect_tensors[i][universal_reference_list.index(k), 0] = sub_tensor.reshape(1, 1, *tuple([b]*(i-1)))
        
        return Construct(self.ident, phi_tensors, cost_tensors, theta_tensors, effect_tensors, self.limit)

def Build_New_Construct(constructs, multipliers, stability_conditions=[]): #list of tuples, number represents index and a 1 or -1 for positive or negative
    """
    Makes a new construct from a list of constructs. 
    For information on stability_conditions see "Factorio Stalling, Management, and Constructs."
    
    Parameters
    ----------
    constructs:
        List of Constructs to be combined into the new construct.
    multipliers:
        List of multipliers to for each construct in constructs.
    stability_conditions: (optional)
        A list of tuples, the first value being the index of the item that must be stabilized,
        the second value is a 1 or -1 depending on if the stability condition should be positive or negative
    
    Returns
    -------
    new_construct:
        The new construct formed.
    """
    multiplied_constructs = []
    for i in range(len(constructs)):
        multiplied_constructs.append(constructs[i]*multipliers[i])
        multiplied_constructs[i].dump_shapes()

    new_ident = "+".join([e.ident for e in multiplied_constructs])
    new_phi_tensors = []
    new_cost_tensors = []
    for i in range(2):
        new_phi_tensors.append(extra_dimensional_projection([e.phi_tensors[i] for e in multiplied_constructs]))
        new_cost_tensors.append(extra_dimensional_projection([e.cost_tensors[i] for e in multiplied_constructs], [0]))
    new_presc_theta_tensors = []
    new_effect_tensors = []
    for i in range(2+len(MODULE_EFFECTS)):
        new_presc_theta_tensors.append(extra_dimensional_projection([e.theta_tensors[i] for e in multiplied_constructs]))
        new_effect_tensors.append(extra_dimensional_projection([e.effect_tensors[i] for e in multiplied_constructs], [0]))
    #ah yes... the stability conditions
    new_theta_tensors = []
    for i in range(len(new_presc_theta_tensors)):
        new_shape = list(new_presc_theta_tensors[i].shape)
        new_shape[0] += len(stability_conditions)
        new_theta_tensors.append(SparseTensor(new_shape))
        new_theta_tensors[i].copy_from(new_presc_theta_tensors[i])
        for j in range(len(stability_conditions)):
            index, parity = stability_conditions[j]
            lf = np.zeros(new_effect_tensors[0].shape[0])
            lf[index] = 1
            temp = parity * einstein_summation(''.join([chr(ord('i')+j) for j in range(i+1)])+',i', 
                                               new_effect_tensors[i], lf, stay_sparse=True)
            if isinstance(temp, SparseTensor): #fixme somehow
                temp = temp.add_rank(0)
                new_theta_tensors[i].__setitem__(tuple([new_presc_theta_tensors[i].shape[0]+j]+[slice(None, None, None)]*i), temp)
            else:
                new_theta_tensors[i][new_presc_theta_tensors[i].shape[0]+j] = temp
            
    
    new_limit = sum([e.limit for e in multiplied_constructs[1:]], multiplied_constructs[0].limit)
    
    new_construct = Construct(new_ident, new_phi_tensors, new_cost_tensors, new_theta_tensors, new_effect_tensors, new_limit)
    new_construct.dump_shapes()
    return new_construct

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

