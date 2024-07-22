from __future__ import annotations

from globalsandimports import *
from utils import *
from lookuptables import *

if TYPE_CHECKING:
    from tools import FactorioInstance

class UncompiledConstruct:
    """An uncompiled construct, contains all the information needed to compile a single construct.
    """

    ident: str
    """Unique identifier"""
    drain: CompressedVector
    """Passive drain to the product space"""
    deltas: CompressedVector
    """Changes to product space from running the construct"""
    effect_effects: dict[str, list[str]]
    """Specifies how this construct is affected by module effects"""
    allowed_modules: list[tuple[str, bool, bool]]
    """Each tuple represents a module, if it can be used inside the building, and if it can be used in beacons for the building"""
    internal_module_limit: int
    """Number of module slots inside the building"""
    base_productivity: Fraction
    """Baseline productivity effect of the building
        https://lua-api.factorio.com/latest/prototypes/CraftingMachinePrototype.html#base_productivity
        https://lua-api.factorio.com/latest/prototypes/MiningDrillPrototype.html#base_productivity"""
    base_inputs: CompressedVector
    """The inputs required to start the machine, used for future catalyst calculations"""
    cost: CompressedVector
    """The cost of a single instance (without any modules)"""
    limit: TechnologicalLimitation
    """Required technological level to make this construct (without any modules)"""
    building: dict
    """Link the the building entity for tile size values
        https://lua-api.factorio.com/latest/prototypes/EntityPrototype.html#tile_width
        https://lua-api.factorio.com/latest/prototypes/EntityPrototype.html#tile_height"""
    research_effected: list[str]
    """What research modifiers effect this construct"""
    surfaces: list[str]
    """What surfaces this construct can be executed on"""

    def __init__(self, ident: str, drain: CompressedVector, deltas: CompressedVector, effect_effects: dict[str, list[str]], 
                 allowed_modules: list[tuple[str, bool, bool]], internal_module_limit: int, base_inputs: CompressedVector, cost: CompressedVector, 
                 limit: TechnologicalLimitation, building: dict, base_productivity: Fraction = Fraction(0), research_effected: list[str] | None = None,
                 surfaces: list[str] | str = "Nauvis") -> None:
        """
        Parameters
        ----------
        ident : str
            Unique identifier
        drain : CompressedVector
            Passive drain to the product space
        deltas : CompressedVector
            Changes to product space from running the construct
        effect_effects : dict[str, list[str]]
            Specifies how this construct is affected by module effects
        allowed_modules : list[tuple[str, bool, bool]]
            Each tuple represents a module, if it can be used inside the building, and if it can be used in beacons for the building
        internal_module_limit : int
            Number of module slots inside the building
        base_inputs : CompressedVector
            The inputs required to start the machine, used for future catalyst calculations
        cost : CompressedVector
            The cost of a single instance (without any modules)
        limit : TechnologicalLimitation
            Required technological level to make this construct (without any modules)
        building : dict
            Link the the building entity for tile size values
            https://lua-api.factorio.com/latest/prototypes/EntityPrototype.html#tile_width
            https://lua-api.factorio.com/latest/prototypes/EntityPrototype.html#tile_height
        base_productivity : Fraction, optional
            Baseline productivity effect of the building
            https://lua-api.factorio.com/latest/prototypes/CraftingMachinePrototype.html#base_productivity
            https://lua-api.factorio.com/latest/prototypes/MiningDrillPrototype.html#base_productivity
            , by default Fraction(0)
        research_effected : list[str], optional
            What research modifiers effect this construct
        surfaces : list[str] | str, optional
            What surfaces this constructs can run on. Default "Nauvis"
        """
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
        if not research_effected is None:
            self.research_effected = research_effected
        else:
            self.research_effected = []
        if isinstance(surfaces, list):
            self.surfaces = surfaces
        else:
            self.surfaces = [surfaces]
        
    def __repr__(self) -> str:
        return str(self.ident)+\
                "\n\tAn added drain of: "+str(self.drain)+\
                "\n\tWith a vector of: "+str(self.deltas)+\
                "\n\tAn effect table of: "+str(self.effect_effects)+\
                "\n\tAllowed modules: "+str(self.allowed_modules)+\
                "\n\tInternal module count: "+str(self.internal_module_limit)+\
                "\n\tBase productivity: "+str(self.base_productivity)+\
                "\n\tBase inputs of: "+str(self.base_inputs)+\
                "\n\tA Cost of: "+str(self.cost)+\
                "\n\tRequiring: "+str(self.limit)+\
                "\n\tBuilding size of: "+str(self.building['tile_width'])+" by "+str(self.building['tile_height'])

class CompiledConstruct:
    """A compiled UncompiledConstruct for high speed and low memory column generation.
    """
    origin: UncompiledConstruct
    """Construct compiled from"""
    _technological_productivity_table: ResearchTable
    """ResearchTable containing the added productivity associated with this Construct given a Tech Level"""
    _technological_speed_multipliers: ResearchTable
    """ResearchTable containing speed multipliers associated with this Construct given a Tech Level"""
    effect_transform: sparse.csr_matrix
    """Effect this construct has in multilinear form"""
    flow_transform: sparse.csr_matrix
    """Absolute value of effect_transform. Used to calculate scaled transport costs"""
    flow_characterization: np.ndarray
    """What reference values are changed by this construct. Used to calculate static transport costs"""
    base_cost: np.ndarray
    """Cost vector associated with the module-less and beacon-less construct"""
    _required_price_indices: np.ndarray
    """Indicies that must be priced to build this construct"""
    effective_area: int
    """Area usage of an instance without beacons"""
    _isa_mining_drill: bool
    """If this construct should be priced based on size when calculating in size restricted mode"""
    _instance: FactorioInstance
    """Instance associated with this construct"""

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
            self._technological_productivity_table = instance.research_modifiers['laboratory-productivity']
        elif "mining-drill-productivity-bonus" in origin.research_effected: #https://lua-api.factorio.com/latest/types/MiningDrillProductivityBonusModifier.html
            self._technological_productivity_table = instance.research_modifiers['mining-drill-productivity-bonus']
        else:
            self._technological_productivity_table = ResearchTable()
        if "laboratory-speed" in origin.research_effected: #https://lua-api.factorio.com/latest/types/LaboratorySpeedModifier.html
            self._technological_speed_multipliers = instance.research_modifiers['laboratory-speed']
        else:
            self._technological_speed_multipliers = instance.research_modifiers['no-speed-modifier']

        self.effect_transform = encode_effect_deltas_to_multilinear(origin.deltas, origin.effect_effects, instance.reference_list)
        self.flow_transform = abs(self.effect_transform)
        self.flow_characterization = ((np.asarray(self.effect_transform.todense())!=0).sum(axis=0)!=0).astype(float).flatten()
        assert self.flow_characterization.shape[0]==len(instance.reference_list), str(self.flow_characterization.shape[0])+", "+str(len(instance.reference_list))
        
        true_cost: CompressedVector = copy.deepcopy(origin.cost)
        for item in instance.catalyst_list:
            if item in origin.base_inputs.keys():
                true_cost = true_cost + CompressedVector({item: -1 * origin.base_inputs[item]})
        
        self.base_cost = np.zeros(len(instance.reference_list))
        for k, v in true_cost.items():
            self.base_cost[instance.reference_list.index(k)] = v
        
        self._required_price_indices = np.array([instance.reference_list.index(k) for k in true_cost.keys()])

        self.effective_area = origin.building['tile_width'] * origin.building['tile_height'] + min(origin.building['tile_width'], origin.building['tile_height'])

        self._isa_mining_drill = origin.building['type']=="mining-drill"

        self._instance = instance
    
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
        #return self._technological_lookup_tables.max(known_technologies)
        return link_lookup_table(self.origin.internal_module_limit, 
                                 (self.origin.building['tile_width'], self.origin.building['tile_height']), 
                                 self.origin.allowed_modules, self._instance, 
                                 self.origin.base_productivity+self._technological_productivity_table.value(known_technologies))
    
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
        return float(self._technological_speed_multipliers.value(known_technologies))

    def columns(self, args: ColumnSpecifier) -> ColumnTable:
        """Produces the best column possible given a pricing model

        Parameters
        ----------
        args : ColumnSpecifier
            ColumnSpecifier to make a column for

        Returns
        -------
        ColumnTable
            Table of column for this construct
        """
        if not (args.known_technologies >= self.origin.limit) or args.inverse_priced_indices[self._required_price_indices].sum()>0: #rough line, ordered?
            column, cost, true_cost, ident = np.zeros((self.base_cost.shape[0], 0)), np.zeros(0), np.zeros((self.base_cost.shape[0], 0)), np.zeros(0, dtype=CompressedVector)
        else:
            cost_function = args.cost_function(self, args.transport_costs)
            lookup_table = self.lookup_table(args.known_technologies)
            speed_multi = self.speed_multiplier(args.known_technologies)

            evaluation, module_string = lookup_table.best_point(self, cost_function, args.transport_costs, args.inverse_priced_indices, args.dual_vector)

            column = (evaluation.multilinear_effect @ self.effect_transform + evaluation.running_cost.flatten()).reshape(-1, 1) * speed_multi
            cost = cost_function(evaluation)
            true_cost = true_cost_function(self, args.transport_costs, evaluation)
            ident = np.array([CompressedVector({self.origin.ident + module_string: 1})])

            assert np.dot(true_cost.flatten(), args.inverse_priced_indices)==0, self.origin.ident+"'s true cost contains unpriced items: " + \
                    " & ".join([self._instance.reference_list[i] for i in range(len(self._instance.reference_list)) if args.inverse_priced_indices[i]!=0 and true_cost[i, 0]!=0])

            column = column.reshape(-1, 1)
            cost = cost.reshape(-1)
            true_cost = true_cost.reshape(-1, 1)
            ident = ident.reshape(-1)

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
        if not (known_technologies >= self.origin.limit) or inverse_priced_indices[self._required_price_indices].sum()>0: #rough line, ordered?
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
    """
    subconstructs: list[ComplexConstruct] | list[CompiledConstruct]
    """ComplexConstructs that makeup this Complex Construct"""
    _stabilization: dict[int, str]
    """What inputs and outputs are stabilized (total input, output, or both must be zero) in this construct"""
    ident: str
    """Name for this construct"""
    _instance: FactorioInstance
    """FactorioInstance associated with this construct"""
    _transport_types: tuple[str, ...]
    """What transport cost table to use when avaiable, uses first avaiable"""
    attributes: dict[Any, Any]
    """Misc attributes"""

    def __init__(self, subconstructs: Sequence[ComplexConstruct], ident: str, transport_types: tuple[str, ...] | str | None = None, 
                 attributes: dict[Any, Any] | None = None) -> None:
        """
        Parameters
        ----------
        subconstructs : Sequence[ComplexConstruct]
            ComplexConstructs that makeup this Complex Construct
        ident : str
            Name for this construct
        transport_types : tuple[str, ...] | str | None
            Which transport type(s) should be added to costs and potentially their priority ordering
        attributes : dict[Any, Any] | None
            Special attributes to hold onto
        """
        self.subconstructs = list(subconstructs)
        self._stabilization = {}
        self.ident = ident
        self._instance = subconstructs[0]._instance
        if isinstance(transport_types, str):
            self._transport_types = (transport_types, )
        elif transport_types is None:
            self._transport_types = tuple()
        else:
            self._transport_types = transport_types
        if attributes:
            self.attributes = attributes
        else:
            self.attributes = {}

    def stabilize(self, row: int, direction: Literal["Positive"] | Literal["Positive and Negative"] | Literal["Negative"]) -> None:
        """Applies stabilization on this ComplexConstruct

        Parameters
        ----------
        row : int
            Which row to stabilize
        direction : str
            Direction of stabilization. "Positive", "Positive and Negative", "Negative"
        """
        if row in self._stabilization.keys():
            if direction=="Positive and Negative" or self._stabilization[row]=="Positive and Negative" or direction!=self._stabilization[row]:
                self._stabilization[row] = "Positive and Negative"
        else:
            self._stabilization[row] = direction

    def columns(self, args: ColumnSpecifier) -> ColumnTable:
        """Produces the best columns possible given a pricing model

        Parameters
        ----------
        args : ColumnSpecifier
            ColumnSpecifier to make columns for

        Returns
        -------
        ColumnTable
            Table of columns for this construct
        """
        assert len(self._stabilization)==0, "Stabilization not implemented yet." #linear combinations
        table: list[ColumnTable] = []

        best_transport_type = None
        for transport_type in self._transport_types:
            if transport_type in args.transport_cost_functions.keys():
                best_transport_type = transport_type
                break
        if best_transport_type:
            transport_residual_pair = args.transport_costs + args.transport_cost_functions[best_transport_type](args.dual_vector)
        else:
            transport_residual_pair = args.transport_costs

        nargs = ColumnSpecifier(args.cost_function, args.inverse_priced_indices, args.dual_vector, transport_residual_pair, args.transport_cost_functions, args.known_technologies)
        for sc in self.subconstructs:
            assert isinstance(sc, ComplexConstruct)
            table.append(sc.columns(nargs))
        out: ColumnTable = ColumnTable.sum(table, args.inverse_priced_indices.shape[0])

        assert out.columns.shape[0] == out.true_costs.shape[0]
        assert out.columns.shape[1] == out.true_costs.shape[1]
        assert out.columns.shape[1] == out.costs.shape[0], str(out.columns.shape[1])+" "+str(out.costs.shape[0])
        assert out.columns.shape[1] == out.idents.shape[0]

        for stab_row, stab_dir in self._stabilization.items():
            if "Positive" in stab_dir:
                out = out.stabilize_row(stab_row, 1)
            if "Negative" in stab_dir:
                out = out.stabilize_row(stab_row, -1)

        return out

    def efficiency_analysis(self, args: ColumnSpecifier, valid_rows: np.ndarray, post_analyses: dict[str, dict[int, float]]) -> float:
        """Determines the best possible realizable efficiency of the construct

        Parameters
        ----------
        args : ColumnSpecifier
            ColumnSpecifier used to make columns for the analysis
        valid_rows : np.ndarray
            Outputing rows of the dual
        post_analyses : dict[str, dict[int, float]]
            All the post analyses needed for efficiency analysis

        Returns
        -------
        float
            Efficiency decimal, 1 should mean as efficient as optimal factory elements

        Raises
        ------
        RuntimeError
            Error with optimization that shouldn't happen
        """
        vector_table = self.columns(args)

        vector_table.mask(np.logical_not(np.asarray((vector_table.columns[np.where(~valid_rows)[0], :] < 0).sum(axis=0)).flatten()))

        if vector_table.columns.shape[1]==0:
            return np.nan
        
        if not self.ident in post_analyses.keys(): #if this flag is set we don't maximize stability before calculating the efficiency.
            return np.max(np.divide(vector_table.columns.T @ args.dual_vector, vector_table.costs)) # type: ignore
        else:
            logging.debug("Doing special post analysis calculating for: "+self.ident)
            stabilizable_rows = np.where(np.logical_and(np.asarray((vector_table.columns > 0).sum(axis=1)), np.asarray((vector_table.columns < 0).sum(axis=1))))[0]
            stabilizable_rows = np.delete(stabilizable_rows, np.where(np.in1d(stabilizable_rows, np.array(post_analyses[self.ident].keys())))[0])

            R = vector_table.columns[np.concatenate([np.array([k for k in post_analyses[self.ident].keys()]), stabilizable_rows]), :]
            u = np.concatenate([np.array([v for v in post_analyses[self.ident].values()]), np.zeros_like(stabilizable_rows)])
            c = vector_table.costs - (vector_table.columns.T @ args.dual_vector)

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

            return np.dot(vector_table.columns.T @ args.dual_vector, primal) / np.dot(c, primal)

    def __repr__(self) -> str:
        return self.ident + " with " + str(len(self.subconstructs)) + " subconstructs." + \
               ("\n\tWith Stabilization: "+str(self._stabilization) if len(self._stabilization.keys()) > 0 else "")

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
        self._stabilization = {}
        self.ident = subconstruct.origin.ident
        self._instance = subconstruct._instance
        self._transport_types = tuple()
        self.attributes = {}

    def stabilize(self, row: int, direction: Literal["Positive"] | Literal["Positive and Negative"] | Literal["Negative"]) -> None:
        """Applies stabilization on this ComplexConstruct

        Parameters
        ----------
        row : int
            Which row to stabilize
        direction : str
            Direction of stabilization. "Positive", "Positive and Negative", "Negative"
        """
        raise RuntimeError("Cannot stabilize a singular constuct.")

    def columns(self, args: ColumnSpecifier) -> ColumnTable:
        """Produces the best column possible given a pricing model

        Parameters
        ----------
        args : ColumnSpecifier
            ColumnSpecifier to make columns for

        Returns
        -------
        ColumnTable
            Table of columns for this construct
        """
        assert isinstance(self.subconstructs[0], CompiledConstruct)
        return self.subconstructs[0].columns(args)

class ManualConstruct:
    """Manual Constructs are hand crafted constructs. This should only be used when there is no other way to progress
    """
    ident: str
    """Unique identifier"""
    deltas: CompressedVector
    """Delta vector of this construct"""
    _column: np.ndarray
    """Column vector of this manual action"""
    limit: TechnologicalLimitation
    """Tech level required to complete this manual action"""

    def __init__(self, ident: str, deltas: CompressedVector, limit: TechnologicalLimitation, instance: FactorioInstance):
        """
        Parameters
        ----------
        ident : str
            Unique identifier
        deltas : CompressedVector
            Deltas from running construct
        limit : TechnologicalLimitation
            Tech level required to complete this manual action
        instance : FactorioInstance
            FactorioInstance in use
        """        
        self.ident = ident
        self.deltas = deltas
        self._column = np.zeros(len(instance.reference_list))
        for k, v in deltas.items():
            self._column[instance.reference_list.index(k)] = v
        self.limit = limit

    def column(self, known_technologies: TechnologicalLimitation) -> tuple[np.ndarray, float, str | None]:
        """Gets the column for this construct

        Parameters
        ----------
        known_technologies : TechnologicalLimitation
            Current tech level

        Returns
        -------
        tuple[np.ndarray, float, str | None]
            Effect column
            Cost
            Ident
        """        
        if self.limit >= known_technologies:
            return self._column, 1, None
        else:
            return np.zeros_like(self._column), 0, self.ident
        
    @staticmethod
    def columns(all_constructs: Collection[ManualConstruct], known_technologies: TechnologicalLimitation) -> ColumnTable:
        """Calculates the columns for all ManualConstructs

        Parameters
        ----------
        all_constructs : Collection[ManualConstruct]
            All ManualConstructs to calculate columns for
        known_technologies : TechnologicalLimitation
            Current tech level

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray[CompressedVector, Any]]
            Matrix of effect columns,
            Vector of costs,
            Matrix of exact costs,
            Ident columns
        """        
        construct_arr: np.ndarray[ManualConstruct, Any] = np.array(all_constructs, dtype=ManualConstruct)
        mask = np.array([known_technologies >= construct.limit for construct in construct_arr])
        construct_arr = construct_arr[mask]

        columns = np.vstack([construct.effect_vector for construct in construct_arr]).T
        costs = np.ones_like(construct_arr, dtype=np.float64)
        true_costs = np.vstack([np.zeros_like(construct.effect_vector) for construct in construct_arr]).T
        idents = np.concatenate([np.array([CompressedVector({construct.ident: 1})]) for construct in construct_arr])

        return ColumnTable(columns, costs, true_costs, idents)
    
    def __repr__(self) -> str:
        return str(self.ident)+\
                "\n\tWith a deltas of: "+str(self.deltas)+\
                "\n\tRequiring: "+str(self.limit)
                #"\n\tWith a vector of: "+str(self.effect_vector)+\

def create_reference_list(uncompiled_constructs: Collection[UncompiledConstruct]) -> tuple[str, ...]:
    """Creates a reference list given a collection of UncompiledConstructs

    Parameters
    ----------
    uncompiled_constructs : Collection[UncompiledConstruct]
        Collection of UncompiledConstructs to create reference from

    Returns
    -------
    tuple[str, ...]
        A reference list containg every value of CompressedVector within
    """
    logging.info("Creating a reference list from a total of %d constructs.", len(uncompiled_constructs))
    reference_list = set()
    for construct in uncompiled_constructs:
        reference_list.update(set(construct.drain.keys()))
        reference_list.update(set(construct.deltas.keys()))
        reference_list.update(set(construct.base_inputs.keys()))
        reference_list.update(set(construct.cost.keys()))
        for val, _, _ in construct.allowed_modules:
            reference_list.add(val)
    reference_list = list(reference_list)
    reference_list.sort()
    logging.info("A total of %d items/fluids were found for the reference list.", len(reference_list))
    logging.debug(reference_list)
    return tuple(reference_list)

def determine_catalysts(uncompiled_construct_list: Collection[UncompiledConstruct], reference_list: Sequence[str]) -> tuple[str, ...]:
    """Determines the catalysts of a collection of UncompiledConstructs
    TODO: Detecting do nothing loops
    
    Parameters
    ----------
    uncompiled_construct_list : Collection[UncompiledConstruct]
        UncompiledConstructs to create catalysts from
    reference_list : Sequence[str]
        The universal reference list

    Returns
    -------
    tuple[str, ...]
        A list of catalyst items and fluids
    """
    logging.debug("Determining the catalysts present in a total of %d constructs.", len(uncompiled_construct_list))
    
    graph = {}
    for item in reference_list:
        graph[item] = set()
    for ident in [construct.ident for construct in uncompiled_construct_list]:
        graph[ident+"=construct"] = set()
        
    for construct in uncompiled_construct_list:
        for k, v in list(construct.deltas.items()) + list(construct.base_inputs.items()):
            if v > 0:
                graph[construct.ident+"=construct"].add(k)
            if v < 0:
                graph[k].add(construct.ident+"=construct")
    
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

def calculate_actives(reference_list: Sequence[str],catalyst_list: Sequence[str], data: dict) -> tuple[str, ...]:
    """Calculates all items that should be actively produce in a material factory. 
    Includes catalysts and any item that can be placed on the ground

    Parameters
    ----------
    reference_list : Sequence[str]
        The universal reference list
    catalyst_list : Sequence[str]
        The catalyst list
    data : dict
        The whole of data.raw

    Returns
    -------
    tuple[str, ...]
        Items that need to be actively produced in a material factory
    """    
    actives = set(copy.deepcopy(catalyst_list))

    for item in data['item'].values():
        if 'place_result' in item.keys() and item['name'] in reference_list:
            actives.add(item['name'])
    
    for module in data['module'].values():
        if module['name'] in reference_list:
            actives.add(module['name'])
        
    actives = list(actives)
    actives.sort()

    return tuple(actives)
