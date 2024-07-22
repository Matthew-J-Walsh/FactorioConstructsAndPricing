from __future__ import annotations

from globalsandimports import *
from lpsolvers import *
from utils import *
from costfunctions import *
from transportation import *

if TYPE_CHECKING:
    from tools import FactorioInstance
    from constructs import CompiledConstruct

class ColumnSpecifier(NamedTuple):
    """All the information needed for a construct to decide what the best column is to generate
    """    
    cost_function: PricedCostFunction
    """Cost function being used, pricing already set"""
    inverse_priced_indices: np.ndarray
    """Array indicating what elements of the reference list are unpriced"""
    dual_vector: np.ndarray | None
    """Dual vector to optimize with"""
    transport_costs: TransportCost
    """Transportation costs that have reached this far"""
    transport_cost_functions: dict[str, TransportationCompiler]
    """Transportation cost functions for adding more transport costs to the residual pair"""
    known_technologies: TechnologicalLimitation
    """Research state to generate for"""

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
    """
    module_count: int
    """Number of modules in this lookup table"""
    building_width: int
    """Smaller side of building"""
    building_height: int
    """Larger side of building"""
    avaiable_modules: list[tuple[str, bool, bool]]
    """Modules that can be used for this lookup table"""
    base_productivity: Fraction
    """Base productivity for this lookup table"""

    _ref_length: int
    """Length of the reference list"""
    _point_length: int
    """Length of points of this lookup table"""
    _point_restrictions_transform: sparse.csr_matrix
    """Transformation on inverse_pricing_indicies to calculate what indicies of the points are allowed to be nonzero"""
    _string_table: list[str]
    """Table of strings for point indicies"""
    _allowed_internal_module_count: int
    """Number of interal modules"""
    _allowed_external_module_count: int
    """Number of exteral modules"""
    _beacon_design_sizes: np.ndarray
    """Array with cardinality of beacon frequency designs for different beacon designs"""
    _beacon_module_slots: np.ndarray
    """Number of slots for each beacon design"""
    _beacon_design_one_form: np.ndarray
    """Transformation to calculate what beacon design is being used"""
    _beacon_design_effect_multi: np.ndarray
    """Effect multi from a beacon design onto the external modules"""
    _beacon_design_cost_multi: np.ndarray
    """Cost multi from a beacon design onto the beacon cost and external module costs"""
    _beacon_design_beacon_cost_index: np.ndarray
    """Index of a beacon design's building cost"""
    _beacon_design_electric_cost: np.ndarray
    """Electricity cost of the beacon"""
    _electric_index: int
    """Index of electricity in the reference list"""
    _base_effect_vector: np.ndarray
    """The effect vector of this ModuleLookupTable without any modules in use"""
    _internal_effect_matrix: np.ndarray
    """Matrix to calculate the internal module effects given a point"""
    _internal_cost_matrix: np.ndarray
    """Matrix to calculate the internal module costs given a point"""
    _external_effect_matrix: np.ndarray
    """Matrix to calculate the external module effects given a point"""
    _external_cost_matrix: np.ndarray
    """Matrix to calculate the external module costs given a point"""
    _beacon_areas: np.ndarray
    """Area of each beacon design"""

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
                internal_modules.append(instance._data_raw['module'][module_name])
            if external:
                external_modules.append(instance._data_raw['module'][module_name])

        if len(avaiable_modules)==0:
            beacon_module_designs: list[tuple[dict, str, list[tuple[Fraction, Fraction]]]] = []
        else:
            beacon_module_designs: list[tuple[dict, str, list[tuple[Fraction, Fraction]]]] = []
            for beacon in list(instance._data_raw['beacon'].values()):
                for design_name, design in beacon_designs((self.building_width, self.building_height), beacon):
                    beacon_module_designs.append((beacon, design_name, [design]))

        self._string_table = [module['name']+"|i" for module in internal_modules] + [module['name']+"|e" for module in external_modules] + [beacon['name']+": "+design_name for beacon, design_name, designs in beacon_module_designs]
        self._point_length = len(self._string_table)

        point_restrictions_transform = sparse.lil_matrix((self._point_length, len(instance.reference_list)))
        i = 0
        for internal_module in internal_modules:
            point_restrictions_transform[i, instance.reference_list.index(internal_module['name'])] = 1
            i += 1
        for external_module in external_modules:
            point_restrictions_transform[i, instance.reference_list.index(external_module['name'])] = 1
            i += 1
        for beacon, design_name, designs in beacon_module_designs:
            point_restrictions_transform[i, instance.reference_list.index(beacon['name'])] = 1
            i += 1
        self._point_restrictions_transform = sparse.csr_matrix(point_restrictions_transform)


        self._ref_length = len(instance.reference_list)


        self._allowed_internal_module_count = len(internal_modules)
        self._allowed_external_module_count = len(external_modules)


        self._beacon_design_sizes = np.array([len(designs) for beacon, design_name, designs in beacon_module_designs])
        self._beacon_module_slots = np.array([beacon['module_specification']['module_slots'] for beacon, design_name, designs in beacon_module_designs])
        self._point_length = self._allowed_internal_module_count + self._allowed_external_module_count + len(beacon_module_designs)



        self._beacon_design_count = sum([len(designs) for beacon, design_name, designs in beacon_module_designs])+1
        self._beacon_design_one_form = np.array([k for k in itertools.accumulate([len(designs) for beacon, design_name, designs in beacon_module_designs])])

        self._beacon_design_effect_multi = np.zeros(self._beacon_design_count)
        self._beacon_design_cost_multi = np.zeros(self._beacon_design_count)
        self._beacon_design_beacon_cost_index = np.zeros(self._beacon_design_count, dtype=int)
        self._beacon_design_electric_cost = np.zeros(self._beacon_design_count)
        i = 1
        for j, (beacon, design_name, designs) in enumerate(beacon_module_designs):
            for design in designs:
                self._beacon_design_effect_multi[i] = design[0] * beacon['distribution_effectivity']
                self._beacon_design_cost_multi[i] = design[1]
                self._beacon_design_beacon_cost_index[i] = instance.reference_list.index(beacon['name'])
                self._beacon_design_electric_cost[i] = -1 * beacon['energy_usage_raw'] * design[1]
                i += 1
            assert self._beacon_design_one_form[j]==(i-1), self._beacon_design_one_form[j] - (i - 1)
        self._electric_index = instance.reference_list.index('electric')

        self._base_effect_vector = np.ones(len(ACTIVE_MODULE_EFFECTS))
        self._base_effect_vector[ACTIVE_MODULE_EFFECTS.index("productivity")] += self.base_productivity

        self._internal_effect_matrix = np.zeros((len(ACTIVE_MODULE_EFFECTS), self._allowed_internal_module_count))
        self._internal_cost_matrix = np.zeros((self._ref_length, self._allowed_internal_module_count))
        for i, module in enumerate(internal_modules):
            for j, effect_name in enumerate(ACTIVE_MODULE_EFFECTS):
                if effect_name in module['effect'].keys():
                    self._internal_effect_matrix[j, i] = module['effect'][effect_name]['bonus']
            self._internal_cost_matrix[instance.reference_list.index(module['name']), i] = 1

        self._external_effect_matrix = np.zeros((len(ACTIVE_MODULE_EFFECTS), self._allowed_external_module_count))
        self._external_cost_matrix = np.zeros((self._ref_length, self._allowed_external_module_count))
        for i, module in enumerate(external_modules):
            for j, effect_name in enumerate(ACTIVE_MODULE_EFFECTS):
                if effect_name in module['effect'].keys():
                    self._external_effect_matrix[j, i] = module['effect'][effect_name]['bonus']
            self._external_cost_matrix[instance.reference_list.index(module['name']), i] = 1

        beacon_areas = [0]
        for beacon, design_name, designs in beacon_module_designs:
            for design in designs:
                beacon_areas.append(design[1] * beacon['tile_height'] * beacon['tile_width'])
        self._beacon_areas = np.array(beacon_areas)

        assert self._beacon_design_one_form.shape[0] == len(beacon_module_designs), str(self._beacon_design_one_form.shape[0])+","+str(len(beacon_module_designs))
        assert self._beacon_areas.shape[0]==self._beacon_design_count, str(self._beacon_areas.shape[0])+","+str(self._beacon_design_count)+"\n"+str(self._beacon_design_sizes)

    def best_point(self, construct: CompiledConstruct, cost_function: CompiledCostFunction, transport_cost: TransportCost, inverse_priced_indices: np.ndarray, dual_vector: np.ndarray | None) -> tuple[PointEvaluations, str]:
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
        if self._point_length==0:
            return PointEvaluations(np.ones((1, 2**len(ACTIVE_MODULE_EFFECTS))), np.zeros((1, self._ref_length)), np.zeros((1, self._ref_length)), np.zeros(1)), ""
        
        if dual_vector is None:
            point = np.zeros(self._point_length)
            evaluation = self.get_point_evaluations(point)
            #print("B")
        else:
            point, evaluation = self.search(construct, cost_function, transport_cost, inverse_priced_indices, dual_vector)
            #print("C")
        module_string = (" with module setup: " + " & ".join([self._string_table[i]+" x"+str(v) for i, v in enumerate(point[:self._allowed_internal_module_count+self._allowed_external_module_count]) if v>0]) + " " + \
                                                       " & ".join([self._string_table[i+self._allowed_internal_module_count+self._allowed_external_module_count]+" density-"+str(v) for i, v in enumerate(point[self._allowed_internal_module_count+self._allowed_external_module_count:]) if v>0]) if np.sum(point)>0 else "")
        return evaluation, module_string
    
    def search(self, construct: CompiledConstruct, cost_function: CompiledCostFunction, transport_cost: TransportCost, inverse_priced_indices: np.ndarray, dual_vector: np.ndarray) -> tuple[np.ndarray, PointEvaluations]:
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
        point_inverse_restrictions = self._point_restrictions_transform @ inverse_priced_indices

        multilinear_weighting_vector = construct.effect_transform @ dual_vector
        multilinear_transport_scaling_effect_vector = (construct.effect_transform @ transport_cost.scaling_effect).T
        static_transport_effect = np.dot(construct.flow_characterization, transport_cost.static_effect)

        current_point = np.zeros(self._point_length)
        current_point_evaluation = self.get_point_evaluations(current_point)
        current_point_cost = cost_function(current_point_evaluation)[0]
        cost_mode = current_point_cost==0
        if cost_mode:
            current_point_value = (current_point_evaluation.multilinear_effect @ multilinear_weighting_vector)[0]
        else:
            current_point_value = (current_point_evaluation.multilinear_effect @ multilinear_weighting_vector)[0] / current_point_cost

        i = 0
        while True:
            new_points = self.get_neighbors(current_point, point_inverse_restrictions)
            new_point_evaluation = self.get_point_evaluations(new_points)
            best_new_point_value, best_new_point_index, best_point_evaluation = get_best_point(multilinear_weighting_vector, multilinear_transport_scaling_effect_vector, static_transport_effect, 
                                                                                               cost_function, transport_cost, new_point_evaluation, cost_mode)
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
        assert point.shape[1]==self._point_length, point.shape[1]-self._point_length
        
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
        if point.shape[1]!=self._point_length:
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

        assert points.shape[1]==self._point_length #.3

        if points.shape[1]==0: #.2
            raise ValueError()

        internal_module_portion = points[:, :self._allowed_internal_module_count] #.5
        external_module_portion = points[:, self._allowed_internal_module_count:self._allowed_internal_module_count+self._allowed_external_module_count] #.5
        beacon_design_portion = points[:, self._allowed_internal_module_count+self._allowed_external_module_count:] #.4

        chosen_beacon_design = (self._beacon_design_one_form @ beacon_design_portion.T).astype(int) #4.2
        beacon_effect_multi = self._beacon_design_effect_multi[chosen_beacon_design] #1.5
        beacon_cost_multi = self._beacon_design_cost_multi[chosen_beacon_design] #.9
        beacon_direct_cost = np.zeros((points.shape[0], self._ref_length)) #2.0
        beacon_direct_cost[:, self._beacon_design_beacon_cost_index[chosen_beacon_design]] = beacon_cost_multi #3.0
        beacon_electric_effect = np.zeros((self._ref_length, points.shape[0])) #1.5
        beacon_electric_effect[self._electric_index, :] = beacon_cost_multi * self._beacon_design_electric_cost[chosen_beacon_design] #2.3

        internal_effect = self._internal_effect_matrix @ internal_module_portion.T #3.2
        internal_cost = internal_module_portion @ self._internal_cost_matrix.T #11.3

        external_effect = (self._external_effect_matrix @ external_module_portion.T) * beacon_effect_multi #2.9
        external_cost = np.einsum("ij,kj,i->ik", external_module_portion, self._external_cost_matrix, beacon_cost_multi) + beacon_direct_cost #40.3

        total_effect = (internal_effect + external_effect).T + self._base_effect_vector #2.6
        total_effect = (total_effect + MODULE_EFFECT_MINIMUMS_NUMPY + np.abs(total_effect - MODULE_EFFECT_MINIMUMS_NUMPY))/2 #4.7

        multilinear_effect = (np.einsum("ij,jklm->ijklm", total_effect, MODULE_MULTILINEAR_EFFECT_SELECTORS) + MODULE_MULTILINEAR_VOID_SELECTORS[None, ]).prod(axis=1) #11.3
        multilinear_effect = multilinear_effect.reshape(-1, 1<<len(ACTIVE_MODULE_EFFECTS)) #1.0

        effective_area = self._beacon_areas[chosen_beacon_design] #1.4

        return PointEvaluations(multilinear_effect.reshape(points.shape[0], -1), beacon_electric_effect.T.reshape(points.shape[0], -1), 
                                (internal_cost + external_cost).reshape(points.shape[0], -1), effective_area.reshape(points.shape[0], -1)) #3.2

    def __repr__(self) -> str:
        return "Lookup table with parameters: "+str([self.module_count, self.building_width, self.building_height, self.avaiable_modules, self.base_productivity])+" totalling "+str("UNKNOWN TODO")

def get_best_point(multilinear_weighting_vector: np.ndarray, multilinear_transport_scaling_effect_vector: np.ndarray, static_transport_effect: float, 
                   cost_function: CompiledCostFunction, transport_cost: TransportCost, point_evaluations: PointEvaluations, cost_mode: bool) -> tuple[float, np.intp, PointEvaluations]:
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
    point_values = ((point_evaluations.multilinear_effect + multilinear_transport_scaling_effect_vector) @ multilinear_weighting_vector).reshape(-1) + static_transport_effect
    point_costs = cost_function(point_evaluations)
    if cost_mode:
        best_point_index = point_values.argmax()
        best_point_value = point_values[best_point_index]
    else:
        best_point_index = (point_values / point_costs).argmax()
        best_point_value = point_values[best_point_index] / point_costs[best_point_index]
    #best_new_point_cost = new_point_costs[best_new_point_index]
    best_point_evaluation = PointEvaluations(point_evaluations.multilinear_effect[best_point_index].reshape(1, -1), point_evaluations.running_cost[best_point_index].reshape(1, -1), 
                                             point_evaluations.evaulated_cost[best_point_index].reshape(1, -1) ,point_evaluations.effective_area[best_point_index].reshape(1, -1))
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

