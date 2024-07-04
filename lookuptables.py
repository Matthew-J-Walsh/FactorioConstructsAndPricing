from __future__ import annotations

from constructs import *
from globalsandimports import *
from lpsolvers import *
from utils import *
from costfunctions import *

if TYPE_CHECKING:
    from tools import FactorioInstance


def encode_effect_deltas_to_multilinear(deltas: CompressedVector, effect_effects: dict[str, list[str]], reference_list: Sequence[str]) -> sparse.csr_matrix:
    """Calculates the multilinear effect form of deltas

    Parameters
    ----------
    deltas : CompressedVector
        Changes imbued by a construct
    effect_effects : dict[str, list[str]]
        Specifies how the construct is affected by module effects
    reference_list : Sequence[str]
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
    base_productivity : Fraction
        Base productivity for this lookup table
    ref_length : int
        Length of the reference list
    point_length : int
        Length of points of this lookup table
    point_restrictions_transform : sparse.csr_matrix
        Transformation on inverse_pricing_indicies to calculate what indicies of the points are allowed to be nonzero
    string_table : list[str]
        Table of strings for point indicies
    _allowed_internal_module_count : int
        Number of interal modules
    _allowed_external_module_count : int
        Number of exteral modules
    _beacon_design_sizes : np.ndarray
        Array with cardinality of beacon frequency designs for different beacon designs
    _beacon_module_slots : np.ndarray
        Number of slots for each beacon design
    _beacon_design_one_form : np.ndarray
        Transformation to calculate what beacon design is being used
    _beacon_design_effect_multi : np.ndarray
        Effect multi from a beacon design onto the external modules
    _beacon_design_cost_multi : np.ndarray
        Cost multi from a beacon design onto the beacon cost and external module costs
    _beacon_design_beacon_cost_index : np.ndarray
        Index of a beacon design's building cost
    _beacon_design_electric_cost : np.ndarray
        Electricity cost of the beacon
    _electric_index : int
        Index of electricity in the reference list
    _internal_effect_matrix : np.ndarray
        Matrix to calculate the internal module effects given a point
    _internal_cost_matrix : np.ndarray
        Matrix to calculate the internal module costs given a point
    _external_effect_matrix : np.ndarray
        Matrix to calculate the external module effects given a point
    _external_cost_matrix : np.ndarray
        Matrix to calculate the external module costs given a point
    _beacon_areas : np.ndarray
        Area of each beacon design
    """
    module_count: int
    building_width: int
    building_height: int
    avaiable_modules: list[tuple[str, bool, bool]]
    base_productivity: Fraction

    ref_length: int
    point_length: int
    point_restrictions_transform: sparse.csr_matrix
    string_table: list[str]

    _allowed_internal_module_count: int
    _allowed_external_module_count: int
    _beacon_design_sizes: np.ndarray
    _beacon_module_slots: np.ndarray
    _beacon_design_one_form: np.ndarray
    _beacon_design_effect_multi: np.ndarray
    _beacon_design_cost_multi: np.ndarray
    _beacon_design_beacon_cost_index: np.ndarray
    _beacon_design_electric_cost: np.ndarray
    _electric_index: int
    _internal_effect_matrix: np.ndarray
    _internal_cost_matrix: np.ndarray
    _external_effect_matrix: np.ndarray
    _external_cost_matrix: np.ndarray
    _beacon_areas: np.ndarray

    def __init__(self, module_count: int, building_size: tuple[int, int], avaiable_modules: list[tuple[str, bool, bool]], instance: FactorioInstance, base_productivity: Fraction) -> None:
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

        internal_modules: list[dict] = []
        external_modules: list[dict] = []
        for module_name, internal, external in avaiable_modules:
            if internal:
                internal_modules.append(instance.data_raw['module'][module_name])
            if external:
                external_modules.append(instance.data_raw['module'][module_name])

        if len(avaiable_modules)==0:
            beacon_module_designs: list[tuple[dict, list[tuple[Fraction, Fraction]]]] = []
        else:
            beacon_module_designs: list[tuple[dict, list[tuple[Fraction, Fraction]]]] = []
            for beacon in list(instance.data_raw['beacon'].values()):
                for design in beacon_designs((self.building_width, self.building_height), beacon):
                    beacon_module_designs.append((beacon, [design]))

        self.string_table = [module['name']+"|i" for module in internal_modules] + [module['name']+"|e" for module in external_modules] + [beacon['name'] for beacon, designs in beacon_module_designs]
        self.point_length = len(self.string_table)

        point_restrictions_transform = sparse.lil_matrix((self.point_length, len(instance.reference_list)))
        i = 0
        for internal_module in internal_modules:
            point_restrictions_transform[i, instance.reference_list.index(internal_module['name'])] = 1
            i += 1
        for external_module in external_modules:
            point_restrictions_transform[i, instance.reference_list.index(external_module['name'])] = 1
            i += 1
        for beacon, designs in beacon_module_designs:
            point_restrictions_transform[i, instance.reference_list.index(beacon['name'])] = 1
            i += 1
        self.point_restrictions_transform = sparse.csr_matrix(point_restrictions_transform)


        self.ref_length = len(instance.reference_list)


        self._allowed_internal_module_count = len(internal_modules)
        self._allowed_external_module_count = len(external_modules)


        self._beacon_design_sizes = np.array([len(designs) for beacon, designs in beacon_module_designs])
        self._beacon_module_slots = np.array([beacon['module_specification']['module_slots'] for beacon, designs in beacon_module_designs])
        self.point_length = self._allowed_internal_module_count + self._allowed_external_module_count + len(beacon_module_designs)



        self._beacon_design_count = sum([len(designs) for beacon, designs in beacon_module_designs])+1
        self._beacon_design_one_form = np.array([k for k in itertools.accumulate([len(designs) for beacon, designs in beacon_module_designs])])

        self._beacon_design_effect_multi = np.zeros(self._beacon_design_count)
        self._beacon_design_cost_multi = np.zeros(self._beacon_design_count)
        self._beacon_design_beacon_cost_index = np.zeros(self._beacon_design_count, dtype=int)
        self._beacon_design_electric_cost = np.zeros(self._beacon_design_count)
        i = 1
        for j, (beacon, designs) in enumerate(beacon_module_designs):
            for design in designs:
                self._beacon_design_effect_multi[i] = design[0] * beacon['distribution_effectivity']
                self._beacon_design_cost_multi[i] = design[1]
                self._beacon_design_beacon_cost_index[i] = instance.reference_list.index(beacon['name'])
                self._beacon_design_electric_cost[i] = -1 * beacon['energy_usage_raw'] * design[1]
                i += 1
            assert self._beacon_design_one_form[j]==(i-1), self._beacon_design_one_form[j] - (i - 1)
        self._electric_index = instance.reference_list.index('electric')

        self._internal_effect_matrix = np.zeros((len(ACTIVE_MODULE_EFFECTS), self._allowed_internal_module_count))
        self._internal_cost_matrix = np.zeros((self.ref_length, self._allowed_internal_module_count))
        for i, module in enumerate(internal_modules):
            for j, effect_name in enumerate(ACTIVE_MODULE_EFFECTS):
                if effect_name in module['effect'].keys():
                    self._internal_effect_matrix[j, i] = module['effect'][effect_name]['bonus']
            self._internal_cost_matrix[instance.reference_list.index(module['name']), i] = 1

        self._external_effect_matrix = np.zeros((len(ACTIVE_MODULE_EFFECTS), self._allowed_external_module_count))
        self._external_cost_matrix = np.zeros((self.ref_length, self._allowed_external_module_count))
        for i, module in enumerate(external_modules):
            for j, effect_name in enumerate(ACTIVE_MODULE_EFFECTS):
                if effect_name in module['effect'].keys():
                    self._external_effect_matrix[j, i] = module['effect'][effect_name]['bonus']
            self._external_cost_matrix[instance.reference_list.index(module['name']), i] = 1

        beacon_areas = [0]
        for beacon, designs in beacon_module_designs:
            for design in designs:
                beacon_areas.append(design[1] * beacon['tile_height'] * beacon['tile_width'])
        self._beacon_areas = np.array(beacon_areas)

        assert self._beacon_design_one_form.shape[0] == len(beacon_module_designs), str(self._beacon_design_one_form.shape[0])+","+str(len(beacon_module_designs))
        assert self._beacon_areas.shape[0]==self._beacon_design_count, str(self._beacon_areas.shape[0])+","+str(self._beacon_design_count)+"\n"+str(self._beacon_design_sizes)


    def best_point(self, construct: CompiledConstruct, cost_function: CompiledCostFunction, inverse_priced_indices: np.ndarray, dual_vector: np.ndarray | None) -> tuple[PointEvaluations, str]:
        """Calculates the best possible valued column, its associated evaluations, and string

        Parameters
        ----------
        construct : CompiledConstruct
            Construct being calculated for
        cost_function : CompiledCostFunction
            Cost function to use
        inverse_priced_indices : np.ndarray
            What indicies aren't priced and therefor cannot be used in the column
        dual_vector : np.ndarray | None
            Evaluating dual vector

        Returns
        -------
        tuple[PointEvaluations, str]
            PointEvaluations of the best point
            Associated string of the best point
        """        
        if self.point_length==0:
            return PointEvaluations(np.ones((1, 2**len(ACTIVE_MODULE_EFFECTS))), np.zeros((1, self.ref_length)), np.zeros((1, self.ref_length)), np.zeros(1)), ""
        
        if dual_vector is None:
            point = np.zeros(self.point_length)
            evaluation = self.get_point_evaluations(point)
            #print("B")
        else:
            point, evaluation = self.search(construct, cost_function, inverse_priced_indices, dual_vector)
            #print("C")
        module_string = (" with module setup: " + " & ".join([self.string_table[i]+" x"+str(v) for i, v in enumerate(point) if v>0]) if np.sum(point)>0 else "")
        return evaluation, module_string
    
    def search(self, construct: CompiledConstruct, cost_function: CompiledCostFunction, inverse_priced_indices: np.ndarray, dual_vector: np.ndarray) -> tuple[np.ndarray, PointEvaluations]:
        """Searches for the best point

        Parameters
        ----------
        construct : CompiledConstruct
            Construct being calculated for
        cost_function : CompiledCostFunction
            Cost function to use
        inverse_priced_indices : np.ndarray
            What indicies aren't priced and therefor cannot be used in the column
        dual_vector : np.ndarray | None
            Evaluating dual vector

        Returns
        -------
        tuple[np.ndarray, PointEvaluations]
            Best point
            PointEvaluations of the best point

        Raises
        ------
        RuntimeError
            Potential infinite loop
        """        
        point_inverse_restrictions = self.point_restrictions_transform @ inverse_priced_indices

        multilinear_weighting_vector = construct.effect_transform @ dual_vector

        current_point = np.zeros(self.point_length)
        current_point_evaluation = self.get_point_evaluations(current_point)
        current_point_cost = cost_function(construct, current_point_evaluation)[0]
        cost_mode = current_point_cost==0
        if cost_mode:
            current_point_value = (current_point_evaluation.multilinear_effect @ multilinear_weighting_vector)[0]
        else:
            current_point_value = (current_point_evaluation.multilinear_effect @ multilinear_weighting_vector)[0] / current_point_cost

        i = 0
        while True:
            new_points = self.get_neighbors(current_point, point_inverse_restrictions)
            new_point_evaluation = self.get_point_evaluations(new_points)
            best_new_point_value, best_new_point_index, best_point_evaluation = get_best_point(construct, multilinear_weighting_vector, cost_function, new_point_evaluation, cost_mode)
            best_new_point = new_points[best_new_point_index]
            if best_new_point_value < current_point_value or (best_new_point==current_point).all():
                break
            current_point = best_new_point
            current_point_value = best_new_point_value
            current_point_evaluation = best_point_evaluation
            i += 1
            if i>200:
                raise RuntimeError("Potential infinite loop when calculating best point. Something has gone horribly wrong.")

        return current_point, current_point_evaluation
    
    def get_neighbors(self, point: np.ndarray, point_inverse_restrictions: np.ndarray) -> np.ndarray:
        """Calculates the neighbors of a point that are actually usable

        Parameters
        ----------
        point : np.ndarray
            Point to calculate neighbors for
        point_inverse_restrictions : np.ndarray
            What indicies of a point can be nonzero given the pricing model

        Returns
        -------
        np.ndarray
            Neighboring points
        """        
        point = point.reshape(1, -1)
        assert point.shape[1]==self.point_length, point.shape[1]-self.point_length
        
        internal_module_portion = point[:, :self._allowed_internal_module_count]
        external_module_portion = point[:, self._allowed_internal_module_count:self._allowed_internal_module_count+self._allowed_external_module_count]
        beacon_design_portion = point[:, self._allowed_internal_module_count+self._allowed_external_module_count:]

        internal_module_count = internal_module_portion.sum()

        internal_loss_points = np.zeros((0, point.shape[1]))
        if internal_module_count>0:
            nnzs = np.nonzero(internal_module_portion)[1]
            internal_loss_points = np.repeat(point, nnzs.shape[0], axis=0)
            internal_loss_points[np.arange(internal_loss_points.shape[0]), nnzs] -= 1
        internal_gain_points = np.zeros((0, point.shape[1]))
        if internal_module_count<self.module_count:
            internal_gain_points = np.repeat(point, self._allowed_internal_module_count, axis=0)
            internal_gain_points[np.diag_indices(self._allowed_internal_module_count)] += 1
        internal_piviot_points = np.zeros((0, point.shape[1]))
        if internal_module_count==self.module_count:
            nnzs = np.nonzero(internal_module_portion)[1]
            internal_piviot_points = np.repeat(point, nnzs.shape[0] * self._allowed_internal_module_count, axis=0)
            internal_piviot_points[np.arange(internal_piviot_points.shape[0]), np.repeat(nnzs, self._allowed_internal_module_count)] -= 1
            internal_piviot_points[:, :self._allowed_internal_module_count] += np.kron(np.eye(self._allowed_internal_module_count, dtype=int), np.ones(nnzs.shape[0], dtype=int)).T

        beacons_active = beacon_design_portion.sum() > 0

        external_piviot_points = np.zeros((0, point.shape[1]))
        if beacons_active:
            nnzs = self._allowed_internal_module_count + np.nonzero(external_module_portion)[1]
            external_piviot_points = np.repeat(point, nnzs.shape[0] * self._allowed_external_module_count, axis=0)
            external_piviot_points[np.arange(external_piviot_points.shape[0]), np.repeat(nnzs, self._allowed_external_module_count)] -= 1
            external_piviot_points[:, self._allowed_internal_module_count:self._allowed_internal_module_count+self._allowed_external_module_count] += np.kron(np.eye(self._allowed_external_module_count, dtype=int), np.ones(nnzs.shape[0], dtype=int)).T
        beacon_piviot_points = np.zeros((0, point.shape[1]))
        if beacons_active:
            beacon_piviot_points = np.repeat(point, self._beacon_design_sizes.shape[0], axis=0)
            beacon_piviot_points[:, self._allowed_internal_module_count+self._allowed_external_module_count:] = 0
            beacon_piviot_points[:, self._allowed_internal_module_count+self._allowed_external_module_count:][np.diag_indices(self._beacon_design_sizes.shape[0])] = 1
        beacon_pm_points = np.zeros((0, point.shape[1]))
        if beacons_active:
            nnz: int = np.nonzero(beacon_design_portion)[1][0]
            if beacon_design_portion[:, nnz]==self._beacon_design_sizes[nnz]:
                beacon_pm_points = point.copy()
                beacon_pm_points[0, self._allowed_internal_module_count + self._allowed_external_module_count + nnz] -= 1
            else:
                beacon_pm_points = np.repeat(point, 2, axis=0)
                beacon_pm_points[0, self._allowed_internal_module_count + self._allowed_external_module_count + nnz] -= 1
                beacon_pm_points[1, self._allowed_internal_module_count + self._allowed_external_module_count + nnz] += 1
        initial_beacon_piviot_points = np.zeros((0, point.shape[1]))
        if not beacons_active:
            initial_beacon_piviot_points = np.repeat(point, self._allowed_external_module_count * self._beacon_design_sizes.shape[0], axis=0)
            initial_beacon_piviot_points[:, self._allowed_internal_module_count:self._allowed_internal_module_count+self._allowed_external_module_count] = 0
            initial_beacon_piviot_points[np.arange(initial_beacon_piviot_points.shape[0]), self._allowed_internal_module_count + np.tile(np.arange(self._allowed_external_module_count), (self._beacon_design_sizes.shape[0], 1)).flatten()] = self._beacon_module_slots[0]
            initial_beacon_piviot_points[:, self._allowed_internal_module_count+self._allowed_external_module_count:] += np.kron(np.eye(self._beacon_design_sizes.shape[0], dtype=int), np.ones(self._allowed_external_module_count, dtype=int)).T

        out = np.concatenate((point, internal_loss_points, internal_gain_points, internal_piviot_points, external_piviot_points, beacon_piviot_points, beacon_pm_points, initial_beacon_piviot_points), axis=0)

        out = out[(out @ point_inverse_restrictions)==0]

        #for i in range(out.shape[0]):
        #    assert self.valid_point(out[i])

        return out

    def valid_point(self, point: np.ndarray) -> bool:
        """Checks if a given point is a valid point for this lookup table

        Parameters
        ----------
        point : np.ndarray
            Point to check

        Returns
        -------
        bool
            If it's valid
        """        
        point = point.reshape(1, -1)
        if point.shape[1]!=self.point_length:
            return False

        internal_module_portion = point[:, :self._allowed_internal_module_count]
        external_module_portion = point[:, self._allowed_internal_module_count:self._allowed_internal_module_count+self._allowed_external_module_count]
        beacon_design_portion = point[:, self._allowed_internal_module_count+self._allowed_external_module_count:]

        internal_module_count = internal_module_portion.sum()
        if internal_module_count>self.module_count:
            return False
        external_module_count = external_module_portion.sum()
        if beacon_design_portion.sum()>0 and external_module_count>self._beacon_module_slots[beacon_design_portion.argmax()]:
            return False
        if (beacon_design_portion>self._beacon_design_sizes).any():
            return False

        return True

    def get_point_evaluations(self, points: np.ndarray) -> PointEvaluations:
        """Calculates the PointEvaluations of all given points

        Parameters
        ----------
        points : np.ndarray
            points to check

        Returns
        -------
        PointEvaluations
            PointEvaluations of the points

        Raises
        ------
        ValueError
            Misshapen point
        """        
        if len(points.shape)==1 or points.shape[1]==1: #.7
            points = points.reshape(1, -1) #.2

        assert points.shape[1]==self.point_length #.3

        if points.shape[1]==0: #.2
            raise ValueError()

        internal_module_portion = points[:, :self._allowed_internal_module_count] #.5
        external_module_portion = points[:, self._allowed_internal_module_count:self._allowed_internal_module_count+self._allowed_external_module_count] #.5
        beacon_design_portion = points[:, self._allowed_internal_module_count+self._allowed_external_module_count:] #.4

        chosen_beacon_design = (self._beacon_design_one_form @ beacon_design_portion.T).astype(int) #4.2
        beacon_effect_multi = self._beacon_design_effect_multi[chosen_beacon_design] #1.5
        beacon_cost_multi = self._beacon_design_cost_multi[chosen_beacon_design] #.9
        beacon_direct_cost = np.zeros((points.shape[0], self.ref_length)) #2.0
        beacon_direct_cost[:, self._beacon_design_beacon_cost_index[chosen_beacon_design]] = beacon_cost_multi #3.0
        beacon_electric_effect = np.zeros((self.ref_length, points.shape[0])) #1.5
        beacon_electric_effect[self._electric_index, :] = beacon_cost_multi * self._beacon_design_electric_cost[chosen_beacon_design] #2.3

        internal_effect = self._internal_effect_matrix @ internal_module_portion.T #3.2
        internal_cost = internal_module_portion @ self._internal_cost_matrix.T #11.3

        external_effect = (self._external_effect_matrix @ external_module_portion.T) * beacon_effect_multi #2.9
        external_cost = np.einsum("ij,kj,i->ik", external_module_portion, self._external_cost_matrix, beacon_cost_multi) + beacon_direct_cost #40.3

        total_effect = (internal_effect + external_effect).T + 1 #2.6
        total_effect = (total_effect + MODULE_EFFECT_MINIMUMS_NUMPY + np.abs(total_effect - MODULE_EFFECT_MINIMUMS_NUMPY))/2 #4.7

        multilinear_effect = (np.einsum("ij,jklm->ijklm", total_effect, MODULE_MULTILINEAR_EFFECT_SELECTORS) + MODULE_MULTILINEAR_VOID_SELECTORS[None, ]).prod(axis=1) #11.3
        multilinear_effect = multilinear_effect.reshape(-1, 1<<len(ACTIVE_MODULE_EFFECTS)) #1.0

        effective_area = self._beacon_areas[chosen_beacon_design] #1.4

        return PointEvaluations(multilinear_effect.reshape(points.shape[0], -1), beacon_electric_effect.T.reshape(points.shape[0], -1), 
                                (internal_cost + external_cost).reshape(points.shape[0], -1), effective_area.reshape(points.shape[0], -1)) #3.2

    def __repr__(self) -> str:
        return "Lookup table with parameters: "+str([self.module_count, self.building_width, self.building_height, self.avaiable_modules, self.base_productivity])+" totalling "+str("UNKNOWN TODO")

def get_best_point(construct: CompiledConstruct, multilinear_weighting_vector: np.ndarray, cost_function: CompiledCostFunction, point_evaluations: PointEvaluations, cost_mode: bool) -> tuple[float, int, PointEvaluations]:
    """Calculates the best point given a set of poitns and their evaluations

    Parameters
    ----------
    construct : CompiledConstruct
        Construct being looked at
    multilinear_weighting_vector : np.ndarray
        Multilinear effect weightings
    cost_function : CompiledCostFunction
        Cost function being used
    point_evaluations : PointEvaluations
        Evaluations of a bunch of points
    cost_mode : bool
        If the cost is zero and therefor evaluation isn't based on value/cost

    Returns
    -------
    tuple[float, int, PointEvaluations]
        Best point evaluation
        Index of best point
        PointEvaluations of best point
    """    
    point_values = (multilinear_weighting_vector @ point_evaluations.multilinear_effect.T).reshape(-1)
    point_costs = cost_function(construct, point_evaluations)
    if cost_mode:
        best_point_index = int(point_values.argmax())
        best_point_value = point_values[best_point_index]
    else:
        best_point_index = int((point_values / point_costs).argmax())
        best_point_value = point_values[best_point_index] / point_costs[best_point_index]
    #best_new_point_cost = new_point_costs[best_new_point_index]
    best_point_evaluation = PointEvaluations(point_evaluations.multilinear_effect[best_point_index].reshape(1, -1), point_evaluations.running_cost[best_point_index].reshape(1, -1), 
                                             point_evaluations.beacon_cost[best_point_index].reshape(1, -1) ,point_evaluations.effective_area[best_point_index].reshape(1, -1))
    return best_point_value, best_point_index, best_point_evaluation

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
           max(building_size) == table.building_height and set(avaiable_modules) == set(table.avaiable_modules) and base_productivity == table.base_productivity:
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
    technological_lookup_tables : ResearchTable
        ResearchTable containing lookup tables associated with this Construct given a Tech Level
    technological_speed_multipliers : ResearchTable
        ResearchTable containing speed multipliers associated with this Construct given a Tech Level
    effect_transform : sparse.csr_matrix
        Effect this construct has in multilinear form
    base_cost : np.ndarray
        Cost vector associated with the module-less and beacon-less construct
    required_price_indices : np.ndarray
        Indicies that must be priced to build this construct
    paired_cost_transform : np.ndarray
        Additional cost vector from effects, Currently 0 for various reasons
    effective_area : int
        Area usage of an instance without beacons.
    isa_mining_drill : bool
        If this construct should be priced based on size when calculating in size restricted mode
    """
    origin: UncompiledConstruct
    technological_lookup_tables: ResearchTable
    technological_speed_multipliers: ResearchTable
    effect_transform: sparse.csr_matrix
    base_cost: np.ndarray
    required_price_indices: np.ndarray
    effective_area: int
    isa_mining_drill: bool

    def __init__(self, origin: UncompiledConstruct, instance: FactorioInstance):
        """
        Parameters
        ----------
        origin : UncompiledConstruct
            Construct to compile
        instance : FactorioInstance
            Origin FactorioInstance
        """        
        self.origin = origin

        if "laboratory-productivity" in origin.research_effected: #https://lua-api.factorio.com/latest/types/LaboratoryProductivityModifier.html
            self.technological_lookup_tables = ResearchTable()
            for limit, base_prod in instance.research_modifiers['laboratory-productivity']:
                self.technological_lookup_tables.add(limit, link_lookup_table(origin.internal_module_limit, (origin.building['tile_width'], origin.building['tile_height']), origin.allowed_modules, instance, origin.base_productivity+base_prod))
        elif "mining-drill-productivity-bonus" in origin.research_effected: #https://lua-api.factorio.com/latest/types/MiningDrillProductivityBonusModifier.html
            self.technological_lookup_tables = ResearchTable()
            for limit, base_prod in instance.research_modifiers['mining-drill-productivity-bonus']:
                self.technological_lookup_tables.add(limit, link_lookup_table(origin.internal_module_limit, (origin.building['tile_width'], origin.building['tile_height']), origin.allowed_modules, instance, origin.base_productivity+base_prod))
        else:
            self.technological_lookup_tables = ResearchTable()
            self.technological_lookup_tables.add(origin.limit, link_lookup_table(origin.internal_module_limit, (origin.building['tile_width'], origin.building['tile_height']), origin.allowed_modules, instance, origin.base_productivity))
        if "laboratory-speed" in origin.research_effected: #https://lua-api.factorio.com/latest/types/LaboratorySpeedModifier.html
            self.technological_speed_multipliers = instance.research_modifiers['laboratory-speed']
        else:
            self.technological_speed_multipliers = instance.research_modifiers['no-speed-modifier']

        self.effect_transform = encode_effect_deltas_to_multilinear(origin.deltas, origin.effect_effects, instance.reference_list)
        
        true_cost: CompressedVector = copy.deepcopy(origin.cost)
        for item in instance.catalyst_list:
            if item in origin.base_inputs.keys():
                true_cost = true_cost + CompressedVector({item: -1 * origin.base_inputs[item]})
        
        self.base_cost = np.zeros(len(instance.reference_list))
        for k, v in true_cost.items():
            self.base_cost[instance.reference_list.index(k)] = v
        
        self.required_price_indices = np.array([instance.reference_list.index(k) for k in true_cost.keys()])

        self.effective_area = origin.building['tile_width'] * origin.building['tile_height'] + min(origin.building['tile_width'], origin.building['tile_height'])

        self.isa_mining_drill = origin.building['type']=="mining-drill"
            
    def lookup_table(self, known_technologies: TechnologicalLimitation) -> ModuleLookupTable:
        """Calculate the highest ModuleLookupTable that has been unlocked

        Parameters
        ----------
        known_technologies : TechnologicalLimitation
            Current tech level to calculate for 

        Returns
        -------
        ModuleLookupTable
            The highest unlocked lookup table
        """
        return self.technological_lookup_tables.max(known_technologies)
    
    def speed_multiplier(self, known_technologies: TechnologicalLimitation) -> float:
        """Calculate the speed multiplier at a technological level

        Parameters
        ----------
        known_technologies : TechnologicalLimitation
            Current tech level to calculate for

        Returns
        -------
        float
            Multiplier
        """
        return self.technological_speed_multipliers.value(known_technologies)

    def columns(self, cost_function: CompiledCostFunction, inverse_priced_indices: np.ndarray, dual_vector: np.ndarray | None, known_technologies: TechnologicalLimitation) -> ColumnTable:
        """Produces the best column possible given a pricing model

        Parameters
        ----------
        cost_function : CompiledCostFunction
            A compiled cost function
        inverse_priced_indices : np.ndarray
            What indices of the pricing vector aren't priced
        dual_vector : np.ndarray | None
            Dual vector to calculate with, if None is given, give the module-less beacon-less design
        known_technologies : TechnologicalLimitation
            Current tech level to calculate for

        Returns
        -------
        ColumnTable
            Table of column for this construct
        """
        if not (known_technologies >= self.origin.limit) or inverse_priced_indices[self.required_price_indices].sum()>0: #rough line, ordered?
            column, cost, true_cost, ident = np.zeros((self.base_cost.shape[0], 0)), np.zeros(0), np.zeros((self.base_cost.shape[0], 0)), np.zeros(0, dtype=CompressedVector)
            #print("A")
        else:
            lookup_table = self.lookup_table(known_technologies)
            speed_multi = self.speed_multiplier(known_technologies)

            evaluation, module_string = lookup_table.best_point(self, cost_function, inverse_priced_indices, dual_vector)

            column = (evaluation.multilinear_effect @ self.effect_transform + evaluation.running_cost.flatten()).reshape(-1, 1) * speed_multi
            cost = cost_function(self, evaluation)
            true_cost = (self.base_cost + evaluation.beacon_cost).reshape(-1, 1)
            ident = np.array([CompressedVector({self.origin.ident + module_string: 1})])
            #logging.debug(self.origin.ident)
            #logging.debug(evaluation)
            #logging.debug(sparse.csr_matrix(column))
            #logging.debug(cost)
            #logging.debug(sparse.csr_matrix(true_cost))
            #logging.debug(ident)

        assert column.shape[0] == self.base_cost.shape[0]
        assert true_cost.shape[0] == self.base_cost.shape[0]
        assert column.shape[1] == true_cost.shape[1]
        assert column.shape[1] == cost.shape[0]
        assert column.shape[1] == ident.shape[0]

        return ColumnTable(column, cost, true_cost, ident)
    
    def efficency_dump(self, cost_function: CostFunction, inverse_priced_indices: np.ndarray, dual_vector: np.ndarray, known_technologies: TechnologicalLimitation) -> CompressedVector:
        """Dumps the efficiency of all possible constructs

        Parameters
        ----------
        cost_function : CostFunction
            A cost function
        inverse_priced_indices : np.ndarray
            What indices of the pricing vector aren't priced
        dual_vector : np.ndarray | None
            Dual vector to calculate with, if None is given, give the module-less beacon-less design
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
        raise NotImplementedError("Hasn't been reimplemented since change to lookup tables. Likely needs to change to only look at some points.")
        lookup_table = self.lookup_table(known_technologies)
        speed_multi = self.speed_multiplier(known_technologies)
        if not (known_technologies >= self.origin.limit) or inverse_priced_indices[self.required_price_indices].sum()>0: #rough line, ordered?
            return CompressedVector()
        else:
            e, c = self._evaluate(cost_function, inverse_priced_indices, dual_vector, lookup_table, speed_multi)
            
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
                output.update({self._generate_vector(i, lookup_table, speed_multi)[2]: evaluation[i]})

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

    def columns(self, cost_function: CompiledCostFunction, inverse_priced_indices: np.ndarray, dual_vector: np.ndarray | None, known_technologies: TechnologicalLimitation) -> ColumnTable:
        """Produces the best columns possible given a pricing model

        Parameters
        ----------
        cost_function : CompiledCostFunction
            A cost function
        inverse_priced_indices : np.ndarray
            What indices of the pricing vector aren't priced
        dual_vector : np.ndarray | None
            Dual vector to calculate with, if None is given, give the module-less beacon-less design
        known_technologies : TechnologicalLimitation
            Current tech level to calculate for

        Returns
        -------
        ColumnTable
            Table of columns for this construct
        """
        assert len(self.stabilization)==0, "Stabilization not implemented yet." #linear combinations
        table = [sc.columns(cost_function, inverse_priced_indices, dual_vector, known_technologies) for sc in self.subconstructs]
        out = ColumnTable.sum(table, inverse_priced_indices.shape[0])

        assert out.columns.shape[0] == out.true_costs.shape[0]
        assert out.columns.shape[1] == out.true_costs.shape[1]
        assert out.columns.shape[1] == out.costs.shape[0]
        assert out.columns.shape[1] == out.idents.shape[0]

        assert out.columns.shape[0]==self.subconstructs[0].subconstructs[0].base_cost.shape[0] # type: ignore
        return out

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

    def efficiency_analysis(self, cost_function: CompiledCostFunction, inverse_priced_indices: np.ndarray, dual_vector: np.ndarray, 
                            known_technologies: TechnologicalLimitation, valid_rows: np.ndarray, post_analyses: dict[str, dict[int, float]]) -> float:
        """Determines the best possible realizable efficiency of the construct

        Parameters
        ----------
        cost_function : CompiledCostFunction
            A cost function
        inverse_priced_indices : np.ndarray
            What indices of the pricing vector aren't priced
        dual_vector : np.ndarray | None
            Dual vector to calculate with, if None is given, give the module-less beacon-less design
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
        vector_table = self.columns(cost_function, inverse_priced_indices, dual_vector, known_technologies)

        vector_table.mask(np.logical_not(np.asarray((vector_table.columns[np.where(~valid_rows)[0], :] < 0).sum(axis=0)).flatten()))

        if vector_table.columns.shape[1]==0:
            return np.nan
        
        if not self.ident in post_analyses.keys(): #if this flag is set we don't maximize stability before calculating the efficiency.
            return np.max(np.divide(vector_table.columns.T @ dual_vector, vector_table.costs)) # type: ignore
        else:
            logging.debug("Doing special post analysis calculating for: "+self.ident)
            stabilizable_rows = np.where(np.logical_and(np.asarray((vector_table.columns > 0).sum(axis=1)), np.asarray((vector_table.columns < 0).sum(axis=1))))[0]
            stabilizable_rows = np.delete(stabilizable_rows, np.where(np.in1d(stabilizable_rows, np.array(post_analyses[self.ident].keys())))[0])

            R = vector_table.columns[np.concatenate([np.array([k for k in post_analyses[self.ident].keys()]), stabilizable_rows]), :]
            u = np.concatenate([np.array([v for v in post_analyses[self.ident].values()]), np.zeros_like(stabilizable_rows)])
            c = vector_table.costs - (vector_table.columns.T @ dual_vector)

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

            return np.dot(vector_table.columns.T @ dual_vector, primal) / np.dot(c, primal)

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

    def columns(self, cost_function: CompiledCostFunction, inverse_priced_indices: np.ndarray, dual_vector: np.ndarray | None, known_technologies: TechnologicalLimitation) -> ColumnTable:
        """Produces the best column possible given a pricing model

        Parameters
        ----------
        cost_function : CompiledCostFunction
            A cost function
        inverse_priced_indices : np.ndarray
            What indices of the pricing vector aren't priced
        dual_vector : np.ndarray | None
            Dual vector to calculate with, if None is given, give the module-less beacon-less design
        known_technologies : TechnologicalLimitation
            Current tech level to calculate for

        Returns
        -------
        ColumnTable
            Table of columns for this construct
        """
        return self.subconstructs[0].columns(cost_function, inverse_priced_indices,  dual_vector, known_technologies)

