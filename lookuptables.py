from __future__ import annotations

from constructs import *
from globalsandimports import *
#from tools import FactorioInstance
#import tools
from utils import *


def encode_effects_vector_to_multilinear(effect_vector: np.ndarray):
    """
    Takes an effects vector and puts it into multilinear effect form.
    """
    multilinear = np.empty(len(MODULE_EFFECT_ORDERING))
    for i in range(len(MODULE_EFFECT_ORDERING)):
        multilinear[i] = 1
        if len(MODULE_EFFECT_ORDERING[i])!=0:
            for j in MODULE_EFFECT_ORDERING[i]:
                multilinear[i] *= effect_vector[j]
    return multilinear

def encode_effect_deltas_to_multilinear(deltas: CompressedVector, effect_effects: dict[str, list[str]], reference_list: tuple[str, ...], base_productivity: Fraction) -> sparse.csr_matrix:
    """
    Takes a CompressedVector of changes and a dictionary of how different effects effect the outcome and returns the multilinear effect form.
    """
    multilinear = sparse.csr_matrix((len(MODULE_EFFECT_ORDERING), len(reference_list)))
    for k, v in deltas.items():
        keffects = set([MODULE_EFFECTS.index(eff_name) for eff_name, deltas in effect_effects.items() if v in deltas])
        for i in range(len(MODULE_EFFECT_ORDERING)):
            if MODULE_EFFECT_ORDERING[i]==keffects:
                multilinear[i, reference_list.index(k)] = v #pretty slow for csr_matrix, should we speed up with lil and convert back?
                break

    return multilinear


class ModuleLookupTable:
    """
    A lookup table for many CompiledConstructs, used to determine the optimal module setups.

    'Left' most value is the baseline.
    """
    module_count: int
    building_width: int
    building_height: int
    avaiable_modules: list[tuple[str, bool, bool]]
    base_productivity: Fraction
    effect_transform: np.ndarray
    cost_transform: np.ndarray
    module_setups: np.ndarray
    effect_table: np.ndarray
    module_names: list[str]
    limits: np.ndarray

    def __init__(self, module_count: int, building_size: tuple[int, int], avaiable_modules: list[tuple[str, bool, bool]], instance, base_productivity: Fraction):
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

        count = sum([len(list(module_setup_generator(avaiable_modules, module_count, (self.building_width, self.building_height), beacon))) for beacon in [None]+list(instance.data_raw['beacon'].values())])

        self.effect_transform = np.zeros((count, len(MODULE_EFFECT_ORDERING)))
        self.cost_transform = np.zeros((count, len(instance.reference_list)))
        #self.paired_transform = sparse.csr_matrix((count, ))
        self.module_setups = np.zeros((count, len(self.module_names)), dtype=int)
        self.effect_table = np.zeros((count, len(MODULE_EFFECTS)))
        self.limits = np.array([TechnologicalLimitation(instance.tech_tree, []) for _ in range(count)], dtype=object)

        i = 0
        for beacon in [None]+list(instance.data_raw['beacon'].values()):
            for module_set, beacon_cost, module_cost in module_setup_generator(avaiable_modules, 
                                                                               module_count, (self.building_width, self.building_height), beacon):
                module_setup_vector = np.zeros(len(self.module_names))
                for mod, count in module_set.items():
                    module_setup_vector[self.module_names.index(mod)] = count

                self.limits[i] = self.limits[i] + (beacon['limit'] if isinstance(beacon, dict) else TechnologicalLimitation(instance.tech_tree, []))

                effect_vector = np.ones(len(MODULE_EFFECTS))
                effect_vector[MODULE_EFFECTS.index("productivity")] += float(self.base_productivity)
                effected_cost = beacon_cost
                for mod, count in module_set.items():
                    mod_name, mod_region = mod.split("|")
                    if mod_region=="i":
                        effect_vector += count * instance.data_raw['module'][mod_name]['effect_vector'].astype(float)
                    if mod_region=="e":
                        assert beacon is not None
                        effect_vector += count * beacon['distribution_effectivity'] * instance.data_raw['module'][mod_name]['effect_vector'].astype(float)
                    effected_cost = effected_cost + CompressedVector({mod.split("|")[0]: count}) 
                    self.limits[i] = self.limits[i] + instance.data_raw['module'][mod.split("|")[0]]['limit']
                effect_vector = np.maximum(effect_vector, MODULE_EFFECT_MINIMUMS_NUMPY.astype(float))
                
                self.effect_transform[i, :] = encode_effects_vector_to_multilinear(effect_vector)
                #the two following lines are very slow. lil_matrix?
                #self.cost_transform[i, :] = sparse.csr_array(([e for e in effected_cost.values()], ([0 for _ in effected_cost], [instance.reference_list.index(d) for d in effected_cost.keys()])), shape=(1, len(instance.reference_list)), dtype=np.longdouble)
                for k, v in effected_cost.items():
                    self.cost_transform[i, instance.reference_list.index(k)] = v
                self.module_setups[i, :] = module_setup_vector
                self.effect_table[i, :] = effect_vector

                i += 1

    #Total time: 10.8505 s
    def evaluate(self, effect_vector: np.ndarray, cost_vector: np.ndarray, priced_indices: np.ndarray, paired_cost_vector: np.ndarray, base_cost: float) -> int:
        """
        Evaluates a effect weighting vector pair to find the best module combination.

        Parameters
        ----------
        effect_vector:
            A vector containing the weighting of each multilinear combination of effects.
        paired_vector:
            A vector containing the paired effect->cost costs.
        cost_vector:
            A vector containing the weighting of the cost each module.

        Returns
        -------
        Index of the optimal module configuration
        """
        #mask = np.where([known_technologies >= self.limits[i] for i in range(self.limits.shape[0])])[0] #brutally insanely bad 7986.29/8700s
        priced_indices_arr = np.ones(self.cost_transform.shape[1])
        priced_indices_arr[priced_indices] = 0
        mask = np.logical_not(self.cost_transform @ priced_indices_arr)
        return np.argmax((self.effect_transform @ effect_vector)[mask] / (self.cost_transform @ cost_vector + self.effect_transform @ paired_cost_vector + base_cost)[mask]) # type: ignore

    def generate(self, index: int):
        """
        Generates components for a column vector given a setup.
        """
        module_setup = self.module_setups[index]
        ident = (" with module setup: " + " & ".join([str(v)+"x "+self.module_names[i] for i, v in enumerate(module_setup) if v>0]) if np.sum(module_setup)>0 else "")
        return np.maximum(self.effect_table[index], MODULE_EFFECT_MINIMUMS_NUMPY), ident
    
    def __repr__(self) -> str:
        return "Lookup table with parameters: "+str([self.module_count, self.building_width, self.building_height, self.avaiable_modules, self.base_productivity])

_LOOKUP_TABLES: list[ModuleLookupTable] = []
def link_lookup_table(module_count: int, building_size: tuple[int, int], avaiable_modules: list[tuple[str, bool, bool]], instance, base_productivity: Fraction) -> ModuleLookupTable:
    """
    Finds or creates a ModuleLookupTable to return.

    Parameters
    ----------
    module_count:
        Number of modules in the construct class
    building_size:
        Width and Height of the building in the construct class
    avaiable_modules:
        What modules can be used in the constuct class

    """
    for table in _LOOKUP_TABLES:
        if module_count == table.module_count and min(building_size) == table.building_width and \
           max(building_size) == table.building_height and set([module[0] for module in avaiable_modules]) == set(table.module_names): #Total time: 10.8034 s * 30%. This set operation sucks?
            return table
        
    new_table = ModuleLookupTable(module_count, building_size, avaiable_modules, instance, base_productivity)
    _LOOKUP_TABLES.append(new_table)
    return new_table
            

class CompiledConstruct:
    """
    A compiled UncompiledConstruct for fast and low memory column generation.
    """
    origin: UncompiledConstruct
    lookup_table: ModuleLookupTable
    effect_transform: sparse.csr_matrix
    base_cost_vector: np.ndarray
    required_price_indices: np.ndarray
    paired_cost_transform: np.ndarray

    def __init__(self, origin: UncompiledConstruct, instance):
        self.origin = origin

        self.lookup_table = link_lookup_table(origin.internal_module_limit, (origin.building['tile_width'], origin.building['tile_height']), origin.allowed_modules, instance, origin.base_productivity)

        self.effect_transform = encode_effect_deltas_to_multilinear(origin.deltas, origin.effect_effects, instance.reference_list, origin.base_productivity)
        
        true_cost: CompressedVector = copy.deepcopy(origin.cost)
        for item in instance.catalyst_list:
            if item in origin.base_inputs.keys():
                true_cost = true_cost + CompressedVector({item: -1 * origin.base_inputs[item]})
        
        self.base_cost_vector = np.zeros(len(instance.reference_list))
        for k, v in true_cost.items():
            self.base_cost_vector[instance.reference_list.index(k)] = v
        
        #self.base_cost_vector = sparse.csr_array(([e for e in true_cost.values()], ([instance.reference_list.index(d) for d in true_cost.keys()], [0 for _ in true_cost])), shape=(len(instance.reference_list),1), dtype=np.longdouble)
        
        self.required_price_indices = np.array([instance.reference_list.index(k) for k in true_cost.keys()])

        self.paired_cost_transform = np.zeros((len(instance.reference_list), len(MODULE_EFFECT_ORDERING)))
        #for transport_building in LOGISTICAL_COST_MULTIPLIERS.keys():
        #    if transport_building=="pipe":
        #        base_throughput = sum([v for k, v in origin.deltas.items() if k in instance.data_raw['fluid'].keys()])
        #    else:
        #        base_throughput = sum([v for k, v in origin.deltas.items() if k not in instance.data_raw['fluid'].keys()])
            
    def vector(self, pricing_vector: np.ndarray, priced_indices: np.ndarray, dual_vector: np.ndarray | None, known_technologies: TechnologicalLimitation) -> tuple[np.ndarray, np.ndarray, str | None]:
        """
        Produces the best vector possible given a pricing model.

        Parameters
        ----------
        pricing_vector:
            Pricing model to use. If none is given, give the un-modded version.

        Returns
        -------
        A column vector, its true cost vector, and its ident.
        If the construct surpasses current tech limits it returns null size vectors and None for ident.
        """
        #TODO: make priced_indices just a mask instead of the np.where-ified version
        if not (known_technologies >= self.origin.limit) or not np.isin(self.required_price_indices, priced_indices, assume_unique=True).all(): #rough line, ordered?
            return np.zeros((pricing_vector.shape[0], 0)), np.zeros((pricing_vector.shape[0], 0)), None
        if dual_vector is None:
            return self._generate_vector(0)
        else:
            #rough multiplications
            column, cost, ident = self._generate_vector(self.lookup_table.evaluate(self.effect_transform @ dual_vector, pricing_vector, priced_indices, self.paired_cost_transform.T @ pricing_vector, np.dot(self.base_cost_vector, pricing_vector)))
            return column, cost, ident

    #Total time: 341.247 s UM WHAT?
    def _generate_vector(self, index: int) -> tuple[np.ndarray, np.ndarray, str]:
        """
        Calculates the vector information of a module setup.

        Parameters
        ----------
        index:
            Index of the module setup to use.
        
        Returns
        -------
        A column vector, its true cost vector, and its ident.
        """
        module_setup = self.lookup_table.module_setups[index]
        ident = self.origin.ident + (" with module setup: " + " & ".join([str(v)+"x "+self.lookup_table.module_names[i] for i, v in enumerate(module_setup) if v>0]) if np.sum(module_setup)>0 else "")

        column: np.ndarray = self.lookup_table.effect_transform[index] @ self.effect_transform #slow line
        cost: np.ndarray = self.lookup_table.cost_transform[index] + np.dot(self.paired_cost_transform, self.lookup_table.effect_transform[index]) + self.base_cost_vector

        #return sparse.csr_array(column).T, sparse.csr_array(cost).T, ident # type: ignore
        return np.reshape(column, (-1, 1)), np.reshape(cost, (-1, 1)), ident

    def __repr__(self) -> str:
        return self.origin.ident + " CompiledConstruct with "+repr(self.lookup_table)+" as its table."

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
    subconstructs: list[ComplexConstruct] | list[CompiledConstruct]
    stabilization: dict
    ident: str

    def __init__(self, subconstructs: list[ComplexConstruct], ident: str) -> None:
        self.subconstructs = subconstructs
        self.stabilization = {}
        self.ident = ident

    def stabilize(self, row: int, direction: int) -> None:
        """
        Applies stabilization on this ComplexConstruct.

        Parameters
        ----------
        row:
            Which row to stabilize.
        direction:
            Direction of stabilization. 1: Positive, 0: Positive and Negative, -1: Negative.
        """
        if row in self.stabilization.keys():
            if direction==0 or self.stabilization[row]==0 or direction!=self.stabilization[row]:
                self.stabilization[row] = 0
        else:
            self.stabilization[row] = direction

    def vectors(self, pricing_vector: np.ndarray, priced_indices: np.ndarray, dual_vector: np.ndarray | None, known_technologies: TechnologicalLimitation) -> tuple[np.ndarray, np.ndarray, np.ndarray[CompressedVector, Any]]:
        """
        Produces the best vector possible given a pricing model.

        Parameters
        ----------
        pricing_vector:
            Pricing model to use. If none is given, give the un-modded version.

        Returns
        -------
        A column vector, its true cost vector, and its ident.
        """
        assert len(self.stabilization)==0, "Stabilization not implemented yet." #linear combinations
        vectors, costs, idents = zip(*[sc.vectors(pricing_vector, priced_indices, dual_vector, known_technologies) for sc in self.subconstructs]) # type: ignore
        vector = np.concatenate(vectors, axis=1)#sparse.csr_matrix(sparse.hstack(vectors))
        cost = np.concatenate(costs, axis=1)#sparse.csr_matrix(sparse.hstack(costs))
        ident: np.ndarray[CompressedVector, Any] = np.concatenate(idents)

        for stab_row, stab_dir in self.stabilization.items():
            if stab_dir >= 0:
                violating_columns = np.where(vector[:, stab_row] < 0)[0]
                unviolating_columns = np.where(vector[:, stab_row] > 0)[0]
                assert len(unviolating_columns)>0, "Impossible stabilization? "+str(stab_row)
                fixed_columns: list[np.ndarray] = [vector[unviolating_columns]]
                fixed_costs: list[np.ndarray] = [cost[unviolating_columns]]
                fixed_idents: np.ndarray[CompressedVector, Any] = ident[unviolating_columns]
                for vcol, ucol in itertools.product(violating_columns, unviolating_columns):
                    fixed_columns.append(vector[ucol] - (vector[vcol, stab_row] / vector[ucol, stab_row]) * vector[vcol])
                    assert fixed_columns[-1][stab_row]==0 #todo remove me
                    fixed_costs.append(cost[ucol] - (vector[vcol, stab_row] / vector[ucol, stab_row]) * cost[vcol])
                    fixed_idents = np.concatenate((fixed_idents, np.array([ident[ucol] - (vector[vcol, stab_row] / vector[ucol, stab_row]) *ident[ucol]])))
                vector = np.concatenate(fixed_columns, axis=1)#sparse.csr_matrix(sparse.hstack(fixed_columns))
                cost = np.concatenate(fixed_costs, axis=1)#sparse.csr_matrix(sparse.hstack(fixed_costs))
                ident = fixed_idents
            if stab_dir <= 0:
                violating_columns = np.where(vector[:, stab_row] > 0)[0]
                unviolating_columns = np.where(vector[:, stab_row] < 0)[0]
                assert len(unviolating_columns)>0, "Impossible stabilization? "+str(stab_row)
                fixed_columns: list[np.ndarray] = [vector[unviolating_columns]]
                fixed_costs: list[np.ndarray] = [cost[unviolating_columns]]
                fixed_idents: np.ndarray[CompressedVector, Any] = ident[unviolating_columns]
                for vcol, ucol in itertools.product(violating_columns, unviolating_columns):
                    fixed_columns.append(vector[ucol] - (vector[vcol, stab_row] / vector[ucol, stab_row]) * vector[vcol])
                    assert fixed_columns[-1][stab_row]==0 #todo remove me
                    fixed_costs.append(cost[ucol] - (vector[vcol, stab_row] / vector[ucol, stab_row]) * cost[vcol])
                    fixed_idents = np.concatenate((fixed_idents, np.array([ident[ucol] - (vector[vcol, stab_row] / vector[ucol, stab_row]) *ident[ucol]])))
                vector = np.concatenate(fixed_columns, axis=1)#sparse.csr_matrix(sparse.hstack(fixed_columns))
                cost = np.concatenate(fixed_costs, axis=1)#sparse.csr_matrix(sparse.hstack(fixed_costs))
                ident = fixed_idents

        return vector, cost, ident

    def reduce(self, pricing_vector: np.ndarray, priced_indices: np.ndarray, dual_vector: np.ndarray | None, known_technologies: TechnologicalLimitation) -> tuple[np.ndarray, np.ndarray, np.ndarray[CompressedVector, Any]]:
        """
        Produces the best vector possible given a pricing model. Additionally removes columns that cannot be used because their inputs cannot be made.
        Additionally sorts the columns (based on their hash hehe).

        Parameters
        ----------
        pricing_vector:
            Pricing model to use. If none is given, give the un-modded version.

        Returns
        -------
        A column vector, its true cost vector, and its ident.
        """
        vector, cost, ident = self.vectors(pricing_vector, priced_indices, dual_vector, known_technologies)
        mask = np.full(vector.shape[1], True, dtype=bool)

        valid_rows = np.asarray((vector[:, np.where(mask)[0]] > 0).sum(axis=1)).flatten() > 0 #sum is equivalent to any
        logging.info("Beginning reduction of "+str(np.count_nonzero(mask))+" constructs with "+str(np.count_nonzero(valid_rows))+" counted outputs.")
        last_mask = np.full(vector.shape[1], False, dtype=bool)
        while (last_mask!=mask).any():
            last_mask = mask.copy()
            valid_rows = np.asarray((vector[:, np.where(mask)[0]] > 0).sum(axis=1)).flatten() > 0
            mask = np.logical_and(mask, np.logical_not(np.asarray((vector[np.where(~valid_rows)[0], :] < 0).sum(axis=0)).flatten()))
            logging.info("Reduced to "+str(np.count_nonzero(mask))+" constructs with "+str(np.count_nonzero(valid_rows))+" counted outputs.")
    
        vector = vector[:, mask]
        cost = cost[:, mask]
        ident = ident[mask]

        ident_hashes = np.array([hash(ide) for ide in ident])
        sort_list = ident_hashes.argsort()

        #return vector, cost, ident
        return vector[:, sort_list], cost[:, sort_list], ident[sort_list]

    def efficiency_analysis(self, pricing_vector: np.ndarray, priced_indices: np.ndarray, dual_vector: np.ndarray, known_technologies: TechnologicalLimitation, valid_rows: np.ndarray) -> float:
        """
        Determines the best possible realizable efficiency of the construct.

        Parameters
        ----------
        pricing_vector:
            Pricing model to use.
        dual_vector:
            Dual pricing model to use. If none is given, give the un-modded version.

        Returns
        -------
        Efficiency decimal, 1 should mean as efficient as optimal factory elements.
        """
        vector, cost, ident = self.vectors(pricing_vector, priced_indices, dual_vector, known_technologies)
        mask = np.logical_not(np.asarray((vector[np.where(~valid_rows)[0], :] < 0).sum(axis=0)).flatten())
        
        vector = vector[:, mask]
        cost = cost[:, mask]
        ident = ident[mask]

        if vector.shape[1]==0:
            return np.nan
        return np.max(np.divide(vector.T @ dual_vector, cost.T @ pricing_vector)) # type: ignore

    def __repr__(self) -> str:
        return self.ident + " with " + str(len(self.subconstructs)) + " subconstructs." + \
               ("\n\tWith Stabilization: "+str(self.stabilization) if len(self.stabilization.keys()) > 0 else "")

class SingularConstruct(ComplexConstruct):
    """
    Base case ComplexConstruct, only a single UncompiledConstruct is used to create.
    """

    def __init__(self, subconstruct: CompiledConstruct) -> None:
        self.subconstructs = [subconstruct]
        self.stabilization = {}
        self.ident = subconstruct.origin.ident

    def stabilize(self, row: int, direction: int) -> None:
        raise RuntimeError("Cannot stabilize a singular constuct.")

    def vectors(self, pricing_vector: np.ndarray, priced_indices: np.ndarray, dual_vector: np.ndarray | None, known_technologies: TechnologicalLimitation) -> tuple[np.ndarray, np.ndarray, np.ndarray[CompressedVector, Any]]:
        """
        Produces the best vector possible given a pricing model.

        Parameters
        ----------
        pricing_vector:
            Pricing model to use.
        dual_vector:
            Dual pricing model to use. If none is given, give the un-modded version.

        Returns
        -------
        A column vector, its true cost vector, and its ident.
        """
        vector, cost, ident = self.subconstructs[0].vector(pricing_vector, priced_indices,  dual_vector, known_technologies) # type: ignore
        if ident is None:
            return vector, cost, np.array([])
        return vector, cost, np.array([CompressedVector({ident: 1})])

