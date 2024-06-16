from __future__ import annotations

from constructs import *
from globalsandimports import *
from lpsolvers import *
#from tools import FactorioInstance
#import tools
from utils import *


def encode_effects_vector_to_multilinear(effect_vector: np.ndarray) -> np.ndarray:
    """Takes an effects vector and puts it into multilinear effect form

    Parameters
    ----------
    effect_vector : np.ndarray
        Input effect vector to put into multilinear form

    Returns
    -------
    np.ndarray
        Vector in multilinear form
    """
    multilinear = np.full(len(MODULE_EFFECT_ORDERING), np.nan)
    for i in range(len(MODULE_EFFECT_ORDERING)):
        multilinear[i] = 1
        if len(MODULE_EFFECT_ORDERING[i])!=0:
            for j in MODULE_EFFECT_ORDERING[i]:
                multilinear[i] *= effect_vector[j]
    return multilinear

def encode_effect_deltas_to_multilinear(deltas: CompressedVector, effect_effects: dict[str, list[str]], reference_list: tuple[str, ...]) -> sparse.csr_matrix:
    """Calculates the multilinear effect form of deltas

    Parameters
    ----------
    deltas : CompressedVector
        Changes imbued by a construct
    effect_effects : dict[str, list[str]]
        Specifies how the construct is affected by module effects
    reference_list : tuple[str, ...]
        Ordering for reference items

    Returns
    -------
    sparse.csr_matrix
        Multilinear form of the deltas
    """
    multilinear = sparse.lil_matrix((len(MODULE_EFFECT_ORDERING), len(reference_list)))
    for k, v in deltas.items():
        keffects = set([ACTIVE_MODULE_EFFECTS.index(eff_name) for eff_name, effeff in effect_effects.items() if k in effeff])
        for i in range(len(MODULE_EFFECT_ORDERING)):
            if MODULE_EFFECT_ORDERING[i]==keffects:
                multilinear[i, reference_list.index(k)] = v #pretty slow for csr_matrix, should we speed up with lil and convert back?
                break
    return sparse.csr_matrix(multilinear)


class ModuleLookupTable:
    """A lookup table for many CompiledConstructs, used to determine the optimal module setups
    Left ([0], [0, :]) value is the baseline

    Members
    -------
    module_count : int
        Number of modules in this lookup table
    building_width : int
        Smaller side of building
    building_height : int
        Larger side of building
    avaiable_modules : list[tuple[str, bool, bool]]
        Modules that can be used for this lookup table
    instance : FactorioInstance
        Origin FactorioInstance
    base_productivity : Fraction
        Base productivity for this lookup table
    multilinear_effect_transform : np.ndarray
        Multilinear effect transformation for this lookup table
    effect_transform : np.ndarray
        Effect transformation for this lookup table
    cost_transform : np.ndarray
        Cost transformation for this lookup table
    module_setups : np.ndarray
        Module setup table for this lookup table
    effect_table : np.ndarray
        Module effect table for this lookup table
    effective_area_table : np.ndarray
        Area usage table, the size of the beacons in the setup
    module_names : list[str]
        Reference for modules in this lookup table
    limits : np.ndarray
        Array of required technologies for each potential module and beacon setup
    """
    module_count: int
    building_width: int
    building_height: int
    avaiable_modules: list[tuple[str, bool, bool]]
    base_productivity: Fraction
    multilinear_effect_transform: np.ndarray
    effect_transform: sparse.csr_matrix
    cost_transform: np.ndarray
    module_setups: np.ndarray
    effect_table: np.ndarray
    effective_area_table: np.ndarray
    module_names: list[str]
    limits: np.ndarray
    connected_points: list[np.ndarray]
    extreme_points: np.ndarray

    def __init__(self, module_count: int, building_size: tuple[int, int], avaiable_modules: list[tuple[str, bool, bool]], instance, base_productivity: Fraction) -> None:
        """
        Parameters
        ----------
        module_count : int
            Number of modules in this lookup table
        building_size : tuple[int, int]
            Size of building for this lookup table
        avaiable_modules : list[tuple[str, bool, bool]]
            Modules that can be used for this lookup table
        instance : FactorioInstance
            Origin FactorioInstance
        base_productivity : Fraction
            Base productivity for this lookup table
        """        
        self.module_count = module_count
        self.building_width = min(building_size)
        self.building_height = max(building_size)
        self.avaiable_modules = avaiable_modules
        self.base_productivity = base_productivity
        self.module_names = []
        for module_name, internal, external in avaiable_modules:
            if internal:
                self.module_names.append(module_name+"|i")
            if external:
                self.module_names.append(module_name+"|e")

        beacon_module_setups = [(beacon, list(module_setup_generator(avaiable_modules, module_count, (self.building_width, self.building_height), beacon))) for beacon in [None]+list(instance.data_raw['beacon'].values())]
        count = sum([len(bms[1]) for bms in beacon_module_setups])

        self.multilinear_effect_transform = np.zeros((count, len(MODULE_EFFECT_ORDERING)))
        effect_transform = sparse.lil_matrix((count, len(instance.reference_list)))
        self.cost_transform = np.zeros((count, len(instance.reference_list)))
        #self.paired_transform = sparse.csr_matrix((count, ))
        self.module_setups = np.zeros((count, len(self.module_names)), dtype=int)
        self.effect_table = np.zeros((count, len(ACTIVE_MODULE_EFFECTS)))
        self.effective_area_table = np.zeros(count)
        self.limits = np.array([TechnologicalLimitation(instance.tech_tree, []) for _ in range(count)], dtype=object)

        i = 0
        #for beacon in [None]+list(instance.data_raw['beacon'].values()):
        #    for module_setup, module_costs in module_setup_generator(avaiable_modules, module_count, (self.building_width, self.building_height), beacon):
        for beacon, ms in beacon_module_setups:
            for module_setup, module_costs in ms:
                module_setup_vector = np.zeros(len(self.module_names))
                for mod, count in module_setup.items():
                    module_setup_vector[self.module_names.index(mod)] = count

                self.limits[i] = self.limits[i] + (beacon['limit'] if isinstance(beacon, dict) else TechnologicalLimitation(instance.tech_tree, []))

                effect_vector = np.ones(len(ACTIVE_MODULE_EFFECTS))
                effect_vector[ACTIVE_MODULE_EFFECTS.index("productivity")] += float(self.base_productivity)
                for mod, count in module_setup.items():
                    mod_name, mod_region = mod.split("|")
                    if mod_region=="i":
                        effect_vector += count * instance.data_raw['module'][mod_name]['effect_vector'].astype(float)
                    if mod_region=="e":
                        assert beacon is not None
                        effect_vector += count * beacon['distribution_effectivity'] * instance.data_raw['module'][mod_name]['effect_vector'].astype(float)
                    self.limits[i] = self.limits[i] + instance.data_raw['module'][mod.split("|")[0]]['limit']
                effect_vector = np.maximum(effect_vector, MODULE_EFFECT_MINIMUMS_NUMPY.astype(float))

                
                self.multilinear_effect_transform[i, :] = encode_effects_vector_to_multilinear(effect_vector)
                #the two following lines are very slow. lil_matrix?
                #self.cost_transform[i, :] = sparse.csr_array(([e for e in effected_cost.values()], ([0 for _ in effected_cost], [instance.reference_list.index(d) for d in effected_cost.keys()])), shape=(1, len(instance.reference_list)), dtype=np.longdouble)
                for k, v in module_costs.items():
                    assert v >= 0, module_costs
                    self.cost_transform[i, instance.reference_list.index(k)] = v
                    if beacon is not None and beacon['name']==k:
                        self.effective_area_table[i] += v * beacon['tile_width'] * beacon['tile_height']
                        effect_transform[i, instance.reference_list.index('electric')] = -1 * v * beacon['energy_usage_raw']
                self.module_setups[i, :] = module_setup_vector
                self.effect_table[i, :] = effect_vector

                i += 1
                
        self.effect_transform = sparse.csr_matrix(effect_transform)
        assert (self.cost_transform>=0).all()

        return #TODO: Finish later.
        allowed_internal_modules = tuple([instance.data_raw['module'][mod] for mod, i, e in self.avaiable_modules if i])
        allowed_external_modules = tuple([instance.data_raw['module'][mod] for mod, i, e in self.avaiable_modules if e])
        beacon_designs: tuple[tuple[dict, tuple[tuple[Fraction, Fraction], ...]], ...] = tuple([(beacon, tuple(beacon_setups((self.building_height, self.building_height), beacon))) for beacon in instance.data_raw['beacon'].values()])
        new_effects, new_added_effect, new_cost = generate_module_vector_lambdas(allowed_internal_modules, allowed_external_modules, 
                                                                                 beacon_designs, instance.reference_list)(model_point_generator(len(allowed_internal_modules), self.module_count, len(allowed_external_modules), 2, 
                                                                                                                                                tuple([len(designs) for beacon, designs in beacon_designs])))

        if len(self.avaiable_modules) > 3:
            print(self.multilinear_effect_transform.shape)
            print(self.effect_transform.shape)
            print(self.cost_transform.shape)

            print(new_effects.shape)
            print(new_added_effect.shape)
            print(new_cost.shape)

            raise ValueError()


    def evaluate(self, effect_vector: np.ndarray, priced_indices: np.ndarray, dual_vector: np.ndarray) -> np.ndarray:
        """Evaluates a effect weighting vector pair to find the best module combination

        Parameters
        ----------
        effect_vector : np.ndarray
            A vector containing the weighting of each multilinear combination of effects
        priced_indices : np.ndarray
            Indicies of cost vector elements that are priced
        dual_vector : np.ndarray
            Dual vector to calculate with (for beacon energy costs)

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Numerator of evaluations (-np.inf if cannot be made),
            Denominator of evaluations
        """
        inverse_priced_indices_arr = np.ones(self.cost_transform.shape[1])
        inverse_priced_indices_arr[priced_indices] = 0
        mask = self.cost_transform @ inverse_priced_indices_arr > 0
        e = self.multilinear_effect_transform @ effect_vector + self.effect_transform @ dual_vector # type: ignore
        e[mask] = -np.inf
        return e
    
    def search(self, construct: CompiledConstruct, cost_function: Callable[[CompiledConstruct, np.ndarray], np.ndarray], priced_indices: np.ndarray, dual_vector: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, str]:
        if cost_function(construct, np.array([0]))[0]==0:
            extremes = [self._point_search_zc(start_point, construct, cost_function, priced_indices, dual_vector) for start_point in self.extreme_points]
            argmax = 0
            maxval = extremes[0][0] @ dual_vector
            for i in range(1, len(extremes)):
                if extremes[i][0] @ dual_vector > maxval:
                    argmax = i
                    maxval = extremes[i][0] @ dual_vector
            return extremes[argmax]
        else:
            extremes = [self._point_search_nnzc(start_point, construct, cost_function, priced_indices, dual_vector) for start_point in self.extreme_points]
            argmax = 0
            maxval = extremes[0][0] @ dual_vector / extremes[0][1]
            for i in range(1, len(extremes)):
                if extremes[i][0] @ dual_vector / extremes[i][1] > maxval:
                    argmax = i
                    maxval = extremes[i][0] @ dual_vector / extremes[i][1]
            return extremes[argmax]
    
    def _point_search_nnzc(self, start_point: int, construct: CompiledConstruct, cost_function: Callable[[CompiledConstruct, np.ndarray], np.ndarray], 
                           priced_indices: np.ndarray, dual_vector: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, str]:
        inverse_priced_indices_arr = np.ones(self.cost_transform.shape[1])
        inverse_priced_indices_arr[priced_indices] = 0
        mask = self.cost_transform @ inverse_priced_indices_arr > 0
        effect_transform = (construct.effect_transform @ dual_vector)

        current_point = start_point
        current_eval: float = self.multilinear_effect_transform[current_point] @ effect_transform + self.effect_transform[current_point] @ dual_vector
        current_cost: float = cost_function(construct, np.array([current_point]))[0]
        while True:
            new_points = self.connected_points[current_point]
            new_evals: np.ndarray = self.multilinear_effect_transform[new_points] @ effect_transform + self.effect_transform[new_points] @ dual_vector 
            new_costs: np.ndarray = cost_function(construct, new_points)
            new_best_local: int = int(np.argmax(new_evals/new_costs))
            new_best_global: int = new_points[new_best_local]

            if new_evals[new_best_local]/new_costs[new_best_local]>=current_eval/current_cost:
                current_point = new_best_global
                current_eval = new_evals[new_best_local]
                current_cost = new_costs[new_best_local]
            else:
                break

        return self.multilinear_effect_transform[current_point] @ construct.effect_transform + self.effect_transform[current_point], \
               current_cost, self.cost_transform[current_point] + np.dot(construct.paired_cost_transform, self.multilinear_effect_transform[current_point]) + construct.base_cost_vector, \
               construct.origin.ident + (" with module setup: " + " & ".join([str(v)+"x "+self.module_names[i] for i, v in enumerate(self.module_setups[current_point]) if v>0]) if np.sum(self.module_setups[current_point])>0 else "")

    def _point_search_zc(self, start_point: int, construct: CompiledConstruct, cost_function: Callable[[CompiledConstruct, np.ndarray], np.ndarray], 
                           priced_indices: np.ndarray, dual_vector: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, str]:
        raise NotImplementedError()

    def __repr__(self) -> str:
        return "Lookup table with parameters: "+str([self.module_count, self.building_width, self.building_height, self.avaiable_modules, self.base_productivity])+" totalling "+str(self.cost_transform.shape[0])

_LOOKUP_TABLES: list[ModuleLookupTable] = []
def link_lookup_table(module_count: int, building_size: tuple[int, int], avaiable_modules: list[tuple[str, bool, bool]], instance, base_productivity: Fraction) -> ModuleLookupTable:
    """Finds or creates a ModuleLookupTable to return

    Parameters
    ----------
    module_count : int
        Number of modules in the lookup table
    building_size : tuple[int, int]
        Size of building for the lookup table
    avaiable_modules : list[tuple[str, bool, bool]]
        Modules that can be used for the lookup table
    instance : FactorioInstance
        Origin FactorioInstance
    base_productivity : Fraction
        Base productivity for the lookup table

    Returns
    -------
    ModuleLookupTable
        The specified module lookup table
    """
    for table in _LOOKUP_TABLES:
        if module_count == table.module_count and min(building_size) == table.building_width and \
           max(building_size) == table.building_height and set([module[0] for module in avaiable_modules]) == set(table.module_names): #Total time: 10.8034 s * 30%. This set operation sucks?
            return table
        
    new_table = ModuleLookupTable(module_count, building_size, avaiable_modules, instance, base_productivity)
    _LOOKUP_TABLES.append(new_table)
    return new_table
            

class CompiledConstruct:
    """A compiled UncompiledConstruct for high speed and low memory column generation.

    Members
    -------
    origin : UncompiledConstruct
        Construct to compile
    lookup_table:  ModuleLookupTable
        Lookup table associated with this Construct
    effect_transform : sparse.csr_matrix
        Effect this construct has in multilinear form
    base_cost_vector : np.ndarray
        Cost vector associated with the module-less and beacon-less construct
    required_price_indices : np.ndarray
        TODO: Do this differently
    paired_cost_transform : np.ndarray
        Additional cost vector from effects, Currently 0 for various reasons
    effective_area : int
        Area usage of an instance without beacons.
    isa_mining_drill : bool
        If this construct should be priced based on size when calculating in size restricted mode
    """
    origin: UncompiledConstruct
    lookup_table: ModuleLookupTable
    effect_transform: sparse.csr_matrix
    base_cost_vector: np.ndarray
    required_price_indices: np.ndarray
    paired_cost_transform: np.ndarray
    effective_area: int
    isa_mining_drill: bool

    def __init__(self, origin: UncompiledConstruct, instance):
        """
        Parameters
        ----------
        origin : UncompiledConstruct
            Construct to compile
        instance : FactorioInstance
            Origin FactorioInstance
        """        
        self.origin = origin

        self.lookup_table = link_lookup_table(origin.internal_module_limit, (origin.building['tile_width'], origin.building['tile_height']), origin.allowed_modules, instance, origin.base_productivity)

        self.effect_transform = encode_effect_deltas_to_multilinear(origin.deltas, origin.effect_effects, instance.reference_list)
        
        true_cost: CompressedVector = copy.deepcopy(origin.cost)
        for item in instance.catalyst_list:
            if item in origin.base_inputs.keys():
                true_cost = true_cost + CompressedVector({item: -1 * origin.base_inputs[item]})
        
        self.base_cost_vector = np.zeros(len(instance.reference_list))
        for k, v in true_cost.items():
            self.base_cost_vector[instance.reference_list.index(k)] = v
        
        self.required_price_indices = np.array([instance.reference_list.index(k) for k in true_cost.keys()])

        self.paired_cost_transform = np.zeros((len(instance.reference_list), len(MODULE_EFFECT_ORDERING)))
        #for transport_building in LOGISTICAL_COST_MULTIPLIERS.keys():
        #    if transport_building=="pipe":
        #        base_throughput = sum([v for k, v in origin.deltas.items() if k in instance.data_raw['fluid'].keys()])
        #    else:
        #        base_throughput = sum([v for k, v in origin.deltas.items() if k not in instance.data_raw['fluid'].keys()])

        self.effective_area = origin.building['tile_width'] * origin.building['tile_height'] + min(origin.building['tile_width'], origin.building['tile_height'])

        self.isa_mining_drill = origin.building['type']=="mining-drill"
            
    def vector(self, cost_function: Callable[[CompiledConstruct, np.ndarray], np.ndarray], priced_indices: np.ndarray, dual_vector: np.ndarray | None, known_technologies: TechnologicalLimitation) -> tuple[np.ndarray, float, np.ndarray, str | None]:
        """Produces the best vector possible given a pricing model

        Parameters
        ----------
        cost_function : Callable[[CompiledConstruct, np.ndarray], np.ndarray]
            A compiled cost function
        priced_indices : np.ndarray
            What indices of the pricing vector are actually priced
        dual_vector : np.ndarray | None
            Dual vector to calculate with, if None is given, give the module-less beacon-less setup
        known_technologies : TechnologicalLimitation
            Current tech level to calculate for

        Returns
        -------
        tuple[np.ndarray, float, np.ndarray, str | None]
            Best construct values:
            Effect Vector,
            Scalar Cost,
            True Cost Vector,
            Name

            Returns empty arrays, nans, and None if construct cannot be made
        """
        #TODO: make priced_indices just a mask instead of the np.where-ified version
        if not (known_technologies >= self.origin.limit) or not np.isin(self.required_price_indices, priced_indices, assume_unique=True).all(): #rough line, ordered?
            column, cost, true_cost, ident = np.zeros((self.base_cost_vector.shape[0], 0)), np.nan, np.zeros((self.base_cost_vector.shape[0], 0)), None
        elif dual_vector is None:
            column, true_cost, ident = self._generate_vector(0)
            cost = cost_function(self, np.array([0]))[0]
        else:
            e, c = self._evaluate(cost_function, priced_indices, dual_vector)
                
            if np.isclose(c, 0).any():
                assert np.isclose(c, 0).all(), self.origin.ident
                index = int(np.argmax(e))
            else:
                index = int(np.argmax(e / c))
            cost = c[index]
            column, true_cost, ident = self._generate_vector(index)
        
        return column, cost, true_cost, ident

    def _evaluate(self, cost_function: Callable[[CompiledConstruct, np.ndarray], np.ndarray], priced_indices: np.ndarray, dual_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluation of this construct

        Parameters
        ----------
        cost_function : Callable[[CompiledConstruct, np.ndarray], np.ndarray]
            A compiled cost function
        priced_indices : np.ndarray
            What indices of the pricing vector are actually priced
        dual_vector : dual_vector
            Dual vector to calculate with
            
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Numerator of evaluations (-np.inf if cannot be made),
            Denominator of evaluations
        """
        e = self.lookup_table.evaluate(self.effect_transform @ dual_vector, priced_indices, dual_vector)
        c = cost_function(self, np.arange(self.lookup_table.multilinear_effect_transform.shape[0]))
        return e, c

    def _generate_vector(self, index: int) -> tuple[np.ndarray, np.ndarray, str]:
        """Calculates the vector information of a module setup

        Parameters
        ----------
        index : int
            Index of the module setup to use

        Returns
        -------
        tuple[np.ndarray, np.ndarray, str]
            A column vector, 
            its true cost vector, 
            its ident
        """
        module_setup = self.lookup_table.module_setups[index]
        ident = self.origin.ident + (" with module setup: " + " & ".join([str(v)+"x "+self.lookup_table.module_names[i] for i, v in enumerate(module_setup) if v>0]) if np.sum(module_setup)>0 else "")
        
        column: np.ndarray = (self.lookup_table.multilinear_effect_transform[index] @ self.effect_transform + np.asarray(self.lookup_table.effect_transform[index].todense())).flatten()
        cost: np.ndarray = self.lookup_table.cost_transform[index] + np.dot(self.paired_cost_transform, self.lookup_table.multilinear_effect_transform[index]) + self.base_cost_vector

        assert (cost>=0).all(), self.origin.ident

        #return sparse.csr_array(column).T, sparse.csr_array(cost).T, ident # type: ignore
        return np.reshape(column, (-1, 1)), np.reshape(cost, (-1, 1)), ident
    
    def efficency_dump(self, cost_function: Callable[[CompiledConstruct, np.ndarray], np.ndarray], priced_indices: np.ndarray, dual_vector: np.ndarray, known_technologies: TechnologicalLimitation) -> CompressedVector:
        """Dumps the efficiency of all possible constructs

        Parameters
        ----------
        cost_function : Callable[[CompiledConstruct, np.ndarray], np.ndarray]
            A compiled cost function
        priced_indices : np.ndarray
            What indices of the pricing vector are actually priced
        dual_vector : np.ndarray | None
            Dual vector to calculate with, if None is given, give the module-less beacon-less setup
        known_technologies : TechnologicalLimitation
            Current tech level to calculate for

        Returns
        -------
        CompressedVector
            Efficiency Table

        Raises
        ------
        ValueError
            Debugging issues
        """
        if not (known_technologies >= self.origin.limit) or not np.isin(self.required_price_indices, priced_indices, assume_unique=True).all(): #rough line, ordered?
            return CompressedVector()
        else:
            e, c = self._evaluate(cost_function, priced_indices, dual_vector)
            
            output = CompressedVector({'base_vector': self.effect_transform @ dual_vector})
            if np.isclose(c, 0).any():
                assert np.isclose(c, 0).all(), self.origin.ident
                evaluation = e
            else:
                evaluation = (e / c)
            try:
                assert not np.isnan(evaluation).any()
            except:
                logging.debug(self.effect_transform @ dual_vector)
                logging.debug(np.isclose(c, 0))
                logging.debug(e)
                logging.debug(c)
                raise ValueError(self.origin.ident)
            for i in range(evaluation.shape[0]):
                output.update({self._generate_vector(i)[2]: evaluation[i]})

            return output

    def __repr__(self) -> str:
        return self.origin.ident + " CompiledConstruct with "+repr(self.lookup_table)+" as its table."

class ComplexConstruct:
    """A true construct. A formation of subconstructs with stabilization values.

    Members
    -------
    subconstructs : list[ComplexConstruct] | list[CompiledConstruct]
        ComplexConstructs that makeup this Complex Construct
    stabilization : dict
        What inputs and outputs are stabilized (total input, output, or both must be zero) in this construct
    ident : str
        Name for this construct
    """
    subconstructs: list[ComplexConstruct] | list[CompiledConstruct]
    stabilization: dict
    ident: str

    def __init__(self, subconstructs: list[ComplexConstruct], ident: str) -> None:
        """
        Parameters
        ----------
        subconstructs : list[ComplexConstruct]
            ComplexConstructs that makeup this Complex Construct
        ident : str
            Name for this construct
        """
        self.subconstructs = subconstructs
        self.stabilization = {}
        self.ident = ident

    def stabilize(self, row: int, direction: int) -> None:
        """Applies stabilization on this ComplexConstruct

        Parameters
        ----------
        row : int
            Which row to stabilize
        direction : int
            Direction of stabilization. 1: Positive, 0: Positive and Negative, -1: Negative
        """
        if row in self.stabilization.keys():
            if direction==0 or self.stabilization[row]==0 or direction!=self.stabilization[row]:
                self.stabilization[row] = 0
        else:
            self.stabilization[row] = direction

    def vectors(self, cost_function: Callable[[CompiledConstruct, np.ndarray], np.ndarray], priced_indices: np.ndarray, dual_vector: np.ndarray | None, known_technologies: TechnologicalLimitation) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray[CompressedVector, Any]]:
        """Produces the best vectors possible given a pricing model

        Parameters
        ----------
        cost_function : Callable[[CompiledConstruct, np.ndarray], np.ndarray]
            A compiled cost function
        priced_indices : np.ndarray
            What indices of the pricing vector are actually priced
        dual_vector : np.ndarray | None
            Dual vector to calculate with, if None is given, give the module-less beacon-less setup
        known_technologies : TechnologicalLimitation
            Current tech level to calculate for

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray[CompressedVector, Any]]
            Matrix of effect vectors,
            Vector of costs,
            Matrix of exact costs,
            Ident vectors
        """
        assert len(self.stabilization)==0, "Stabilization not implemented yet." #linear combinations
        vectors, costs, true_costs, idents = zip(*[sc.vectors(cost_function, priced_indices, dual_vector, known_technologies) for sc in self.subconstructs]) # type: ignore
        vector = np.concatenate(vectors, axis=1)#sparse.csr_matrix(sparse.hstack(vectors))
        cost = np.concatenate(costs)
        true_cost = np.concatenate(true_costs, axis=1)#sparse.csr_matrix(sparse.hstack(costs))
        ident: np.ndarray[CompressedVector, Any] = np.concatenate(idents)

        for stab_row, stab_dir in self.stabilization.items():
            raise NotImplementedError("Cost true cost issue.")
            if stab_dir >= 0:
                violating_columns = np.where(vector[:, stab_row] < 0)[0]
                unviolating_columns = np.where(vector[:, stab_row] > 0)[0]
                assert len(unviolating_columns)>0, "Impossible stabilization? "+str(stab_row)
                fixed_columns: list[np.ndarray] = [vector[unviolating_columns]]
                fixed_costs: list[np.ndarray] = [true_cost[unviolating_columns]]
                fixed_idents: np.ndarray[CompressedVector, Any] = ident[unviolating_columns]
                for vcol, ucol in itertools.product(violating_columns, unviolating_columns):
                    fixed_columns.append(vector[ucol] - (vector[vcol, stab_row] / vector[ucol, stab_row]) * vector[vcol])
                    assert fixed_columns[-1][stab_row]==0 #todo remove me
                    fixed_costs.append(true_cost[ucol] - (vector[vcol, stab_row] / vector[ucol, stab_row]) * true_cost[vcol])
                    fixed_idents = np.concatenate((fixed_idents, np.array([ident[ucol] - (vector[vcol, stab_row] / vector[ucol, stab_row]) *ident[ucol]])))
                vector = np.concatenate(fixed_columns, axis=1)#sparse.csr_matrix(sparse.hstack(fixed_columns))
                true_cost = np.concatenate(fixed_costs, axis=1)#sparse.csr_matrix(sparse.hstack(fixed_costs))
                ident = fixed_idents
            if stab_dir <= 0:
                violating_columns = np.where(vector[:, stab_row] > 0)[0]
                unviolating_columns = np.where(vector[:, stab_row] < 0)[0]
                assert len(unviolating_columns)>0, "Impossible stabilization? "+str(stab_row)
                fixed_columns: list[np.ndarray] = [vector[unviolating_columns]]
                fixed_costs: list[np.ndarray] = [true_cost[unviolating_columns]]
                fixed_idents: np.ndarray[CompressedVector, Any] = ident[unviolating_columns]
                for vcol, ucol in itertools.product(violating_columns, unviolating_columns):
                    fixed_columns.append(vector[ucol] - (vector[vcol, stab_row] / vector[ucol, stab_row]) * vector[vcol])
                    assert fixed_columns[-1][stab_row]==0 #todo remove me
                    fixed_costs.append(true_cost[ucol] - (vector[vcol, stab_row] / vector[ucol, stab_row]) * true_cost[vcol])
                    fixed_idents = np.concatenate((fixed_idents, np.array([ident[ucol] - (vector[vcol, stab_row] / vector[ucol, stab_row]) *ident[ucol]])))
                vector = np.concatenate(fixed_columns, axis=1)#sparse.csr_matrix(sparse.hstack(fixed_columns))
                true_cost = np.concatenate(fixed_costs, axis=1)#sparse.csr_matrix(sparse.hstack(fixed_costs))
                ident = fixed_idents

        return vector, cost, true_cost, ident

    def reduce(self, cost_function: Callable[[CompiledConstruct, np.ndarray], np.ndarray], priced_indices: np.ndarray, dual_vector: np.ndarray | None, known_technologies: TechnologicalLimitation) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray[CompressedVector, Any]]:
        """Produces the best vectors possible given a pricing model. 
        Additionally removes columns that cannot be used because their inputs cannot be made.
        Additionally sorts the columns (based on their hash hehe).

        Parameters
        ----------
        cost_function : Callable[[CompiledConstruct, np.ndarray], np.ndarray]
            A compiled cost function
        priced_indices : np.ndarray
            What indices of the pricing vector are actually priced
        dual_vector : np.ndarray | None
            Dual vector to calculate with, if None is given, give the module-less beacon-less setup
        known_technologies : TechnologicalLimitation
            Current tech level to calculate for

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray[CompressedVector, Any]]
            Matrix of effect vectors,
            Vector of costs,
            Matrix of exact costs,
            Ident vectors
        """
        vector, cost, true_cost, ident = self.vectors(cost_function, priced_indices, dual_vector, known_technologies)
        mask = np.full(vector.shape[1], True, dtype=bool)

        valid_rows = np.asarray((vector[:, np.where(mask)[0]] > 0).sum(axis=1)).flatten() > 0 #sum is equivalent to any
        logging.debug("Beginning reduction of "+str(np.count_nonzero(mask))+" constructs with "+str(np.count_nonzero(valid_rows))+" counted outputs.")
        last_mask = np.full(vector.shape[1], False, dtype=bool)
        while (last_mask!=mask).any():
            last_mask = mask.copy()
            valid_rows = np.asarray((vector[:, np.where(mask)[0]] > 0).sum(axis=1)).flatten() > 0
            mask = np.logical_and(mask, np.logical_not(np.asarray((vector[np.where(~valid_rows)[0], :] < 0).sum(axis=0)).flatten()))
            logging.debug("Reduced to "+str(np.count_nonzero(mask))+" constructs with "+str(np.count_nonzero(valid_rows))+" counted outputs.")
    
        vector = vector[:, mask]
        cost = cost[mask]
        true_cost = true_cost[:, mask]
        ident = ident[mask]

        ident_hashes = np.array([hash(ide) for ide in ident])
        sort_list = ident_hashes.argsort()

        #return vector, cost, ident
        return vector[:, sort_list], cost[sort_list], true_cost[:, sort_list], ident[sort_list]

    def efficiency_analysis(self, cost_function: Callable[[CompiledConstruct, np.ndarray], np.ndarray], priced_indices: np.ndarray, dual_vector: np.ndarray, 
                            known_technologies: TechnologicalLimitation, valid_rows: np.ndarray, post_analyses: dict[str, dict[int, float]]) -> float:
        """Determines the best possible realizable efficiency of the construct

        Parameters
        ----------
        cost_function : Callable[[CompiledConstruct, np.ndarray], np.ndarray]
            A compiled cost function
        priced_indices : np.ndarray
            What indices of the pricing vector are actually priced
        dual_vector : np.ndarray | None
            Dual vector to calculate with, if None is given, give the module-less beacon-less setup
        known_technologies : TechnologicalLimitation
            Current tech level to calculate for
        valid_rows : np.ndarray
            Outputing rows of the dual

        Returns
        -------
        float
            Efficiency decimal, 1 should mean as efficient as optimal factory elements

        Raises
        ------
        RuntimeError
            Error with optimization that shouldn't happen
        """
        vector, cost, true_cost, ident = self.vectors(cost_function, priced_indices, dual_vector, known_technologies)

        mask = np.logical_not(np.asarray((vector[np.where(~valid_rows)[0], :] < 0).sum(axis=0)).flatten())

        vector = vector[:, mask]
        cost = cost[mask]
        true_cost = true_cost[:, mask]
        ident = ident[mask]

        if vector.shape[1]==0:
            return np.nan
        
        if not self.ident in post_analyses.keys(): #if this flag is set we don't maximize stability before calculating the efficiency.
            return np.max(np.divide(vector.T @ dual_vector, cost)) # type: ignore
        else:
            logging.debug("Doing special post analysis calculating for: "+self.ident)
            stabilizable_rows = np.where(np.logical_and(np.asarray((vector > 0).sum(axis=1)), np.asarray((vector < 0).sum(axis=1))))[0]
            stabilizable_rows = np.delete(stabilizable_rows, np.where(np.in1d(stabilizable_rows, np.array(post_analyses[self.ident].keys())))[0])

            R = vector[np.concatenate([np.array([k for k in post_analyses[self.ident].keys()]), stabilizable_rows]), :]
            u = np.concatenate([np.array([v for v in post_analyses[self.ident].values()]), np.zeros_like(stabilizable_rows)])
            c = cost - (vector.T @ dual_vector)

            primal_diluted, dual = BEST_LP_SOLVER(R, u, c)
            if primal_diluted is None or dual is None:
                logging.debug("Efficiency analysis for "+self.ident+" was unable to solve initial problem, returning nan.")
                return np.nan

            Rp = np.concatenate([c.reshape((1, -1)), R], axis=0)
            up = np.concatenate([np.array([np.dot(c, primal_diluted) * (1 + SOLVER_TOLERANCES['rtol']) - SOLVER_TOLERANCES['atol']]), u])

            primal, dual = BEST_LP_SOLVER(Rp, up, np.ones(c.shape[0]), g=primal_diluted)
            if primal is None or dual is None:
                assert linear_transform_is_gt(R, primal_diluted, u).all()
                assert linear_transform_is_gt(Rp, primal_diluted, up).all()
                raise RuntimeError("Alegedly no primal found but we have one.")

            return np.dot(vector.T @ dual_vector, primal) / np.dot(c, primal)

    def __repr__(self) -> str:
        return self.ident + " with " + str(len(self.subconstructs)) + " subconstructs." + \
               ("\n\tWith Stabilization: "+str(self.stabilization) if len(self.stabilization.keys()) > 0 else "")

class SingularConstruct(ComplexConstruct):
    """Base case ComplexConstruct, only a single UncompiledConstruct is used to create.
    """

    def __init__(self, subconstruct: CompiledConstruct) -> None:
        """_summary_

        Parameters
        ----------
        subconstruct : CompiledConstruct
            The singular element of this construct.
        """        
        self.subconstructs = [subconstruct]
        self.stabilization = {}
        self.ident = subconstruct.origin.ident

    def stabilize(self, row: int, direction: int) -> None:
        """Cannot stabilize a singular constuct

        Parameters
        ----------
        row : int
            Don't use
        direction : int
            Don't use

        Raises
        ------
        RuntimeError
            Cannot stabilize a singular constuct
        """        
        raise RuntimeError("Cannot stabilize a singular constuct.")

    def vectors(self, cost_function: Callable[[CompiledConstruct, np.ndarray], np.ndarray], priced_indices: np.ndarray, dual_vector: np.ndarray | None, known_technologies: TechnologicalLimitation) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray[CompressedVector, Any]]:
        """Produces the best vector possible given a pricing model

        Parameters
        ----------
        cost_function : Callable[[CompiledConstruct, np.ndarray], np.ndarray]
            A compiled cost function
        priced_indices : np.ndarray
            What indices of the pricing vector are actually priced
        dual_vector : np.ndarray | None
            Dual vector to calculate with, if None is given, give the module-less beacon-less setup
        known_technologies : TechnologicalLimitation
            Current tech level to calculate for

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray[CompressedVector, Any]]
            Matrix of effect vectors,
            Vector of costs,
            Matrix of exact costs,
            Ident vectors
        """
        vector, cost, true_cost, ident = self.subconstructs[0].vector(cost_function, priced_indices,  dual_vector, known_technologies) # type: ignore
        if ident is None:
            return vector, np.array([]), true_cost, np.array([])
        return vector, np.array([cost]), true_cost, np.array([CompressedVector({ident: 1})])

