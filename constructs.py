from __future__ import annotations

from utils import *
from globalsandimports import *

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

class ModulatedConstruct:
    """
    A list of lists of LinearConstructs, each list contains a list of LinearConstructs of the same characteriztion (lying in the same orthant). 
    Allows for easy presolving. Formed from an UncompiledConstruct.

    Members
    -------
    subconstructs:
        List of lists of LinearConstructs
    ident:
        Identifier for this set of LinearConstructs
    """
    subconstructs: tuple[tuple[LinearConstruct, ...], ...]
    ident: str

    def __init__(self, constructs: tuple[LinearConstruct] | list[LinearConstruct], ident: str) -> None:
        temp_subcs: list[list[LinearConstruct]] = []
        orthants: dict[Hashable, int] = {}
        for construct in constructs:
            orth: Hashable = vectors_orthant(construct.vector)
            if orth in orthants.keys():
                temp_subcs[orthants[orth]].append(construct)
            else:
                orthants[orth] = len(temp_subcs)
                temp_subcs.append([construct])
        self.subconstructs = tuple([tuple(orth) for orth in temp_subcs])
        self.ident = ident
    
    def compile(self, pricing_model: np.ndarray, pricing_keys: list[int], known_technologies: TechnologicalLimitation) -> tuple[sparse.coo_matrix, np.ndarray, Callable[[np.ndarray, np.ndarray], tuple[CompressedVector, CompressedVector]]]:
        """
        Compiles the modulated construct given a pricing model. Uses the pricing model to compute a Pareto Frontier of LinearConstructs.

        Parameters
        ----------
        pricing_model:
            Reference pricing model, a CompressedVector with reference_list indicies as keys
        known_technologies:
            Tech level to compile at
        
        Returns
        -------
        A:
            A Transformation from counts to an Effect
        c:
            A Transformation from counts to a Cost
        Recovery:
            A Recovery function from counts to CompressedVector of Factory setup
        """
        constructs: list[list[LinearConstruct]] = [[construct for construct in orth if known_technologies>=construct.limit and all([k in pricing_keys for k in construct.cost.row])] for orth in self.subconstructs]
        effects: list[list[sparse.coo_array]] = [[construct.vector for construct in orth] for orth in constructs]
        costs: list[list[float]] = [[np.dot(construct.cost.todense().flatten(), pricing_model) for construct in orth] for orth in constructs]
        cost_weighted_effects: list[list[sparse.coo_array]]  = [[eff / c for eff, c in zip(effs, cs)] for effs, cs in zip(effects, costs)]
        masks: list[np.ndarray] = [pareto_frontier(d) for d in cost_weighted_effects]

        flattened_effects: list[sparse.coo_array] = sum(effects, [])
        compressed_effects: sparse.coo_matrix = sparse.hstack(flattened_effects).T if len(flattened_effects)>0 else sparse.coo_matrix((0, self.subconstructs[0][0].vector.shape[0])) #saved for recovery
        flattened_costs: np.ndarray = np.array(sum(costs, [])) #saved for recovery
        flattened_mask: np.ndarray = np.where(np.concatenate([np.in1d(np.arange(len(effects[i])), masks[i]) for i in range(len(constructs))]))[0]
        flattened_names: list[str] = sum([[construct.ident for construct in orth] for orth in constructs], [])
        #assert flattened_mask.shape[0]==compressed_effects.shape[0]
        assert flattened_costs.shape[0]==compressed_effects.shape[0]

        if len(constructs[0])==0:
            A = sparse.coo_matrix((pricing_model.shape[0], 0))
            c = np.zeros(0)
            names = np.array([])
        else:
            A: sparse.coo_matrix = sparse.coo_matrix(sparse.hstack([sparse.hstack(np.array(effs)[mask], format="coo") for effs, mask in zip(effects, masks)], format="coo"))
            c: np.ndarray = np.vstack([np.vstack(np.array(costs)[mask]) for costs, mask in zip(costs, masks)]).flatten() # type: ignore
            names: np.ndarray = np.vstack([np.vstack(np.array([construct.ident for construct in orth])[mask]) for orth, mask in zip(constructs, masks)]).flatten() # type: ignore

        def recovery(s: np.ndarray, p: np.ndarray) -> tuple[CompressedVector, CompressedVector]:
            s_fixed: np.ndarray = np.zeros(compressed_effects.shape[0])
            s_fixed[flattened_mask] = s
            assert s.shape[0]==c.shape[0], "Given miss-shapen factory in recovery"
            assert p.shape[0]==compressed_effects.shape[1], "Given miss-shapen pricing model in recovery"
            fac = CompressedVector({flattened_names[i]: s_fixed[i] for i in range(s_fixed.shape[0]) if not np.isclose(s_fixed[i], 0)})
            loss_array: np.ndarray = compressed_effects @ p / flattened_costs
            loss = CompressedVector({flattened_names[i]: loss_array[i] for i in range(compressed_effects.shape[0])})
            return fac, loss
        
        return A, c, recovery
    
    def __repr__(self) -> str:
        return self.ident + "\n\tContaining " + str(sum([len(l) for l in self.subconstructs])) + " LinearConstructs split between " + str(len(self.subconstructs)) + " orthants."

class ComplexConstruct:
    """
    A true construct. A formation of subconstructs with stabilization values.

    Members
    -------
    subconstructs:
        Other ComplexConstructs that make up this construct
    stabilization:
        What inputs and outputs are stabilized (total input, output, or both must be zero) in this construct.
    """
    subconstructs: tuple[ComplexConstruct, ...] | tuple[ModulatedConstruct, ...]
    stabilization: dict
    ident: str

    def __init__(self, subconstructs: tuple[ComplexConstruct, ...] | tuple[ModulatedConstruct, ...], ident: str, stabilization: dict | None = None) -> None:
        self.subconstructs = subconstructs
        self.ident = ident
        if stabilization is None:
            self.stabilization = {}
        else:
            self.stabilization = stabilization

    def compile(self, pricing_model: np.ndarray, pricing_keys: list[int], known_technologies: TechnologicalLimitation) -> tuple[sparse.coo_matrix, np.ndarray, sparse.coo_matrix, sparse.coo_array, Callable[[np.ndarray, np.ndarray], tuple[CompressedVector, CompressedVector]]]:
        """
        Compiles the complex construct given a pricing model.

        Parameters
        ----------
        pricing_model:
            Reference pricing model
        known_technologies:
            Current tech level
        
        Returns
        -------
        A:
            A Transformation from variables to an Effect.
        c:
            A Transformation from variables to a Cost.
        N1:
            A Transformation on variables. Combined with T0 forms a constraint on variables.
        N0:
            A Vector. Combined with T0 forms a constraint on variables.
        Recovery:
            A Recovery function from variables and a pricing model to CompressedVector of Factory setup and CompressedVector of unused construct's efficiencies.
        """
        if len(self.stabilization.keys())!=0:
            raise NotImplementedError("Stabilization not implemented yet. TODO")

        compiled_subconstructs = [sc.compile(pricing_model, pricing_keys, known_technologies) for sc in self.subconstructs]
        As, cs, N1s, N0s, Rs = zip(*compiled_subconstructs)

        A = sparse.coo_matrix(sparse.hstack(As, format="coo"))
        assert A.shape[0]==As[0].shape[0]
        c = np.concatenate(cs)
        N1 = sparse.coo_matrix(sparse.hstack(N1s, format="coo"))
        N0 = sparse.coo_array(sparse.hstack(N0s, format="coo"))
        assert all([n.shape[0]==0 for n in N1s])
        assert all([n1.shape[0]==n0.shape[0] for n1, n0 in zip(N1s, N0s)])
        assert N1.shape[0]==0
        assert N0.shape[0]==0
        
        splits_diff = np.array([a.shape[1] for a in As])
        splits_high = np.cumsum(splits_diff)
        splits_low = np.concatenate([np.array([0]), splits_high[:-1]])

        def Recovery(s: np.ndarray, p: np.ndarray) -> tuple[CompressedVector, CompressedVector]:
            assert s.shape[0] == A.shape[1], "Given miss-shapen factory in recovery"
            assert p.shape[0] == A.shape[0], "Given miss-shapen pricing model in recovery"
            fac = CompressedVector()
            loss = CompressedVector()
            for i in range(len(Rs)):
                assert s[splits_low[i]: splits_high[i]].shape[0]==cs[i].shape[0], "Splitting is wrong "+str(splits_low[i])+" "+str(splits_high[i])+" "+str(s[splits_low[i]: splits_high[i]].shape[0])+" "+str(cs[i].shape[0])
                f, l = Rs[i](s[splits_low[i]: splits_high[i]], p)
                fac = fac + f
                loss = loss + l
            return fac, loss

        return A, c, N1, N0, Recovery
    
    def reduce(self, pricing_model: np.ndarray, pricing_keys: list[int], known_technologies: TechnologicalLimitation) -> tuple[sparse.coo_matrix, np.ndarray, sparse.coo_matrix, sparse.coo_array, Callable[[np.ndarray, np.ndarray], tuple[CompressedVector, CompressedVector]]]:
        """
        Compiles the complex construct given a pricing model. Additionally removes columns that cannot be used because their inputs cannot be made.

        Parameters
        ----------
        pricing_model:
            Reference pricing model
        known_technologies:
            Current tech level
        
        Returns
        -------
        A:
            A Transformation from variables to an Effect.
        c:
            A Transformation from variables to a Cost.
        N1:
            A Transformation on variables. Combined with T0 forms a constraint on variables.
        N0:
            A Vector. Combined with T0 forms a constraint on variables.
        Recovery:
            A Recovery function from variables and a pricing model to CompressedVector of Factory setup and CompressedVector of unused construct's efficiencies.
        """
        Af, cf, N1f, N0f, Recoveryf = self.compile(pricing_model, pricing_keys, known_technologies)
        Af = Af.tocsr()
        mask = np.full(Af.shape[1], True, dtype=bool)
        valid_rows = np.asarray((Af[:, np.where(mask)[0]] > 0).sum(axis=1)).flatten() > 0 #sum is equivalent to any

        logging.info("Beginning reduction of "+str(np.count_nonzero(mask))+" constructs with "+str(np.count_nonzero(valid_rows))+" counted outputs.")
        last_mask = np.full(Af.shape[1], False, dtype=bool)
        while (last_mask!=mask).any():
            last_mask = mask.copy()
            valid_rows = np.asarray((Af[:, np.where(mask)[0]] > 0).sum(axis=1)).flatten() > 0
            mask = np.logical_and(mask, np.logical_not(np.asarray((Af[np.where(~valid_rows)[0], :] < 0).sum(axis=0)).flatten()))
            logging.info("Reduced to "+str(np.count_nonzero(mask))+" constructs with "+str(np.count_nonzero(valid_rows))+" counted outputs.")
        
        A = sparse.coo_matrix(Af[:, mask].tocoo())
        c = cf[mask]
        N1 = N1f
        N0 = N0f
        def Recovery(s: np.ndarray, p: np.ndarray) -> tuple[CompressedVector, CompressedVector]:
            assert s.shape[0]==np.count_nonzero(mask), "Given miss-shapen factory in recovery"
            assert p.shape[0]==Af.shape[0], "Given miss-shapen pricing model in recovery"
            s_fixed = np.zeros_like(mask, dtype=s.dtype)
            s_fixed[np.where(mask)[0]] = s
            return Recoveryf(s_fixed, p)
        
        return A, c, N1, N0, Recovery

    def stabilize(self, column: int, direction: int) -> None:
        """
        Applies stabilization on this ComplexConstruct.

        Parameters
        ----------
        column:
            Which column to stabilize
        direction:
            Direction of stabilization. 1: Positive, 0: Positive and Negative, -1: Negative
        """
        if column in self.stabilization.keys():
            if direction==0 or self.stabilization[column]==0 or direction!=self.stabilization[column]:
                self.stabilization[column] = 0
        else:
            self.stabilization[column] = direction
    
    def __repr__(self) -> str:
        return self.ident + " with " + str(len(self.subconstructs)) + " subconstructs." + \
               ("\n\tWith Stabilization: "+str(self.stabilization) if len(self.stabilization.keys()) > 0 else "")

class SingularConstruct(ComplexConstruct):
    """
    Base case ComplexConstruct, only a single UncompiledConstruct is used to create.
    """

    def __init__(self, modulated_construct: ModulatedConstruct):
        super().__init__((modulated_construct,), modulated_construct.ident)
    
    def compile(self, pricing_model: np.ndarray, pricing_keys: list[int], known_technologies: TechnologicalLimitation) -> tuple[sparse.coo_matrix, np.ndarray, sparse.coo_matrix, sparse.coo_array, Callable[[np.ndarray, np.ndarray], tuple[CompressedVector, CompressedVector]]]:
        """
        Compiles the complex construct given a pricing model.

        Parameters
        ----------
        pricing_model:
            Reference pricing model
        known_technologies:
            Current tech level
        
        Returns
        -------
        A:
            A Transformation from variables to an Effect.
        c:
            A Transformation from variables to a Cost.
        N1:
            A Transformation on variables. Combined with T0 forms a constraint on variables.
        N0:
            A Vector. Combined with T0 forms a constraint on variables.
        Recovery:
            A Recovery function from variables and a pricing model to CompressedVector of Factory setup and CompressedVector of unused construct's efficiencies.
        """
        A: sparse.coo_matrix
        c: np.ndarray
        R: Callable[[np.ndarray, np.ndarray], tuple[CompressedVector, CompressedVector]]
        assert len(self.subconstructs)==1
        assert isinstance(self.subconstructs[0], ModulatedConstruct)
        A, c, R = self.subconstructs[0].compile(pricing_model, pricing_keys, known_technologies)
        return A, c, sparse.coo_matrix((0, c.shape[0])), sparse.coo_array((0, 1)), R
        
    def stabilize(self, column: int, direction: int) -> None:
        raise RuntimeError("SingularConstructs cannot be stabilized")

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

    def compile(self, catalyzing_deltas: tuple[str, ...], module_data: dict, reference_list: tuple[str, ...], beacons: list[dict]) -> SingularConstruct:
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
        constructs: list[LinearConstruct] = []
        logging.info("Creating LinearConstructs for "+self.ident)
        logging.info("\nFound a total of "+str(sum([len(list(module_setup_generator(self.allowed_modules, self.internal_module_limit, self.building, beacon))) for beacon in [None]+beacons]))+" mod setups")
        for beacon in [None]+beacons: #None=No beacons
            for module_set, beacon_cost, module_cost in module_setup_generator(self.allowed_modules, self.internal_module_limit, self.building, beacon):
                logging.debug("Generating a linear construct for %s given module setup %s given module costs %s given beacon cost %s", self.ident, module_set, module_cost, beacon_cost)
                ident = self.ident + (" with module setup: " + " & ".join([str(v)+"x "+k for k, v in module_set.items()]) if len(module_set)>0 else "")

                limit = self.limit

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

def module_setup_generator(modules: list[tuple[str, bool, bool]], internal_limit: int, building: dict, beacon: dict | None = None) -> Generator[tuple[CompressedVector, CompressedVector, CompressedVector], None, None]:
    """
    Returns an generator over the set of possible module setups.

    Parameters
    ---------
    modules:
        List of tuples representing avaiable modules and if its allowed in internal/external.
    internal_limit:
        The remaining number of internal module slots.
    building:
        Building being used.
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
            for beacon_count, beacon_cost in beacon_setups(building, beacon):
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

