from __future__ import annotations

from globalsandimports import *

from utils import *
from generators import *
#from linearconstructs import *
from constructs import *
from datarawparse import *
from lpproblems import *
from postprocessing import *
from lookuptables import *
from costfunctions import *


class FactorioInstance():
    """Holds the information in an instance (specific game mod setup) after completing premanagment steps.

    Members
    -------
    _data_raw : dict
        Whole data.raw dictonary post-premanagment
    _tech_tree : TechnologyTree
        Technology Tree of this instance
    _uncompiled_constructs : tuple[UncompiledConstruct, ...]
        All UncompiledConstructs for the game instance
    _manual_constructs : tuple[ManualConstruct, ...]
        ManualConstructs in the instance
    _complex_constructs : list[ComplexConstruct]
        ComplexConstructs in the instance
    _disabled_constructs : list[ComplexConstruct]
        ComplexConstructs that have been disabled
    _compiled : ComplexConstruct | None
        ComplexConstruct of the entire instance or None if hasn't been compiled since last change
    reference_list : tuple[str, ...]
        Every relevent item, fluid, and research identifier (sorted)
    catalyst_list : tuple[str, ...]
        All catalytic refrence_list elements
    active_list : tuple[str, ...]
        All items that can be used in building a factory
    spatial_pricing : np.ndarray
        Pricing model for ore space consumption
    raw_ore_pricing : np.ndarray
        Pricing model for raw ore usage
    COST_MODE : str
        What cost mode is being used. https://lua-api.factorio.com/latest/concepts.html#DifficultySettings
    RELEVENT_FLUID_TEMPERATURES : dict
        Dict with keys of fluid names and values of a dict mapping temperatures to energy densities
    research_modifiers : dict[str, ResearchTable]
        Research modifier technology tables
        Currently ModuleLookupTables use "laboratory-productivity", "mining-drill-productivity-bonus", and "laboratory-speed"
    post_analyses : dict[str, dict[int, float]]
        Post analysis calculations to run, construct name and target outputs
    """
    _data_raw: dict
    _tech_tree: TechnologyTree
    _uncompiled_constructs: tuple[UncompiledConstruct, ...]
    _manual_constructs: tuple[ManualConstruct, ...]
    _complex_constructs: list[ComplexConstruct]
    _disabled_constructs: list[ComplexConstruct]
    _compiled: ComplexConstruct | None
    reference_list: tuple[str, ...]
    catalyst_list: tuple[str, ...]
    active_list: tuple[str, ...]
    spatial_pricing: np.ndarray
    raw_ore_pricing: np.ndarray
    COST_MODE: str
    RELEVENT_FLUID_TEMPERATURES: dict
    research_modifiers: dict[str, ResearchTable]
    post_analyses: dict[str, dict[int, float]]

    def __init__(self, filename: str, COST_MODE: str = 'normal', nobuild: bool = False, raw_ore_pricing: dict[str, Real] | CompressedVector | None = None) -> None:
        """
        Parameters
        ----------
        filename : str
            Filename of data.raw to load for this instance
        COST_MODE : str, optional
            What cost mode should be used https://lua-api.factorio.com/latest/concepts.html#DifficultySettings, by default 'normal'
        nobuild : bool, optional
            Don't build complex constructs, only load in data raw and manage. Only use for debugging. By default False
        raw_ore_pricing: dict[str, Real] | CompressedVector | None
            A particular raw ore pricing model. If None its assumed to be standard
        """
        with open(filename) as f:
            self._data_raw = json.load(f)
        
        self.COST_MODE = COST_MODE
        self.RELEVENT_FLUID_TEMPERATURES = {}

        self._tech_tree = complete_premanagement(self._data_raw, self.RELEVENT_FLUID_TEMPERATURES, self.COST_MODE)
        self.research_modifiers = generate_research_effect_tables(self._data_raw, self._tech_tree)
        self._uncompiled_constructs = generate_all_constructs(self)
        self.reference_list = create_reference_list(self._uncompiled_constructs)
        self.catalyst_list = determine_catalysts(self._uncompiled_constructs, self.reference_list)
        self.active_list = calculate_actives(self.reference_list, self.catalyst_list, self._data_raw)
        self._manual_constructs = generate_manual_constructs(self)
        
        self._disabled_constructs = []

        self.spatial_pricing = np.zeros(len(self.reference_list))
        logging.debug("Spatial pricing info:")
        for mining_drill in self._data_raw['mining-drill'].values():
            self.spatial_pricing[self.reference_list.index(mining_drill['name'])] = mining_drill['tile_width'] * mining_drill['tile_height']
            logging.debug(mining_drill['name']+" point:"+str(self.reference_list.index(mining_drill['name']))+" area:"+str(mining_drill['tile_width'] * mining_drill['tile_height']))
        for beacon in self._data_raw['beacon'].values():
            self.spatial_pricing[self.reference_list.index(beacon['name'])] = beacon['tile_width'] * beacon['tile_height']
            logging.debug(beacon['name']+" point:"+str(self.reference_list.index(beacon['name']))+" area:"+str(beacon['tile_width'] * beacon['tile_height']))
        self.spatial_pricing = self.spatial_pricing / np.linalg.norm(self.spatial_pricing)

        self.raw_ore_pricing = np.zeros(len(self.reference_list))
        if not raw_ore_pricing is None:
            for k, v in raw_ore_pricing.items():
                self.raw_ore_pricing[self.reference_list.index(k)] = v
        else:
            for resource in self._data_raw['resource'].values():
                if resource['category']=='basic-solid':
                    self.raw_ore_pricing[self.reference_list.index(resource['name'])] = 10
                elif resource['category']=='basic-fluid':
                    self.raw_ore_pricing[self.reference_list.index(resource['name'])] = 1
                else:
                    raise ValueError(resource)
        self.raw_ore_pricing = self.raw_ore_pricing / np.linalg.norm(self.raw_ore_pricing)
                
        self.post_analyses = {}

        if not nobuild:
            logging.debug("Building complex constructs.")
            self._complex_constructs = [SingularConstruct(CompiledConstruct(uc, self)) for uc in self._uncompiled_constructs]
            self._compiled = None
            self.compiled
    
    @staticmethod
    def load(filename: str) -> FactorioInstance:
        """Loads a FactorioInstance from memory

        Parameters
        ----------
        filename : str
            File with FactorioInstance in it
        """
        with open(filename, 'rb') as file:
            return pickle.load(file)
        
    def save(self, filename: str) -> None:
        """Saves a FactorioInstance to memory

        Parameters
        ----------
        filename : str
            File to place FactorioInstance into
        """
        self._compiled = None
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @property
    def compiled(self) -> ComplexConstruct:
        if self._compiled is None:
            self._compiled = ComplexConstruct(tuple([cc for cc in self._complex_constructs if not cc in self._disabled_constructs]), "Whole Game Construct") # type: ignore
        return self._compiled

    def disable_complex_construct(self, target_name: str) -> None:
        """Disables the closest named complex construct

        Parameters
        ----------
        target_name : str
            Target name of complex construct to be disabled
        """
        self._disabled_constructs.append(self.search_complex_constructs(target_name))
        self._compiled = None
    
    def enable_complex_construct(self, target_name: str) -> None:
        """Enables the closest named complex construct

        Parameters
        ----------
        target_name : str
            Target name of complex construct to be disabled
        """
        construct = self.search_complex_constructs(target_name)
        try:
            self._disabled_constructs.remove(construct)
            self._compiled = None
        except:
            logging.warning(construct.ident+" was not disabled in the first place.")

    def search_complex_constructs(self, target_name: str) -> ComplexConstruct:
        """Finds the closest named complex construct, using Levenshtein distance

        Parameters
        ----------
        target_name : str
            Target name of complex construct to be disabled

        Returns
        -------
        ComplexConstruct
            Closest match
        """ 
        best_matches: list[ComplexConstruct] = []
        match_distance = Levenshtein.distance(target_name, self._complex_constructs[0].ident)
        logging.debug("Atempting to translate: "+"\""+target_name+"\" starting distance is: "+str(match_distance)+" there "+("is" if target_name in [c.ident for c in self._complex_constructs] else "is not")+" a 0 length translation.")
        for c in self._complex_constructs:
            dist = Levenshtein.distance(target_name, c.ident)
            if dist < match_distance:
                best_matches = [c]
                match_distance = dist
                logging.log(5, "\tFound a new best match in: \""+c.ident+"\" at distance: "+str(match_distance))
            elif dist == match_distance:
                best_matches.append(c)
                logging.log(5, "\t\tAdded the new possible match: \""+c.ident+"\"")
        assert len(best_matches)==1, "Unable to determine which construct an input phrase is associated with.\nPhrase is: "+target_name+"\nPossible constructs were:\n\t"+"\n\t".join([str(m) for m in best_matches])+"\n\tWith distance: "+str(match_distance)
        logging.debug("Translated: \""+target_name+"\" to mean the construct: \""+str(best_matches[0])+"\"")
        return best_matches[0]

    def bind_complex_constructs(self, target_names: list[str | tuple[str, bool]], new_name: str | None = None) -> str:   
        """Binds a list of complex constructs together, disables marked ones, and returns the new name they are under

        Parameters
        ----------
        target_names : list[str  |  tuple[str, bool]]
            List of names and if the original construct should be disabled. If no bool is provided "True" is assumed.
        new_name : str | None, optional
            Specific name to use for the new construct, by default makes up a sensible one

        Returns
        -------
        str
            New construct name
        """
        lookup_names: list[str] = []
        disable_list: list[bool] = []
        for pp in target_names:
            if isinstance(pp, tuple):
                lookup_names.append(pp[0])
                disable_list.append(pp[1])
            else:
                lookup_names.append(pp)
                disable_list.append(True)
        
        constructs: list[ComplexConstruct] = [self.search_complex_constructs(name) for name in lookup_names]
        
        if new_name is None:
            new_name = "Combined Construct["+" and ".join([construct.ident for construct in constructs])+"]"
        else:
            logging.warning("new_names past to bind_complex_construct aren't checked for duplication TODO")

        self._complex_constructs.append(ComplexConstruct(tuple(constructs), new_name)) # type: ignore
        for i, construct in enumerate(constructs):
            if disable_list[i]:
                self.disable_complex_construct(construct.ident)
        
        return new_name

    def solve_for_target(self, targets: CompressedVector, known_technologies: TechnologicalLimitation, reference_model: CompressedVector, uncompiled_cost_function: CostFunction, 
                         recovered_run: ColumnTable | None = None, use_manual: bool = False) -> tuple[CompressedVector, CompressedVector, CompressedVector, CompressedVector, list[int], float, CompressedVector, CompressedVector, ColumnTable]:
        """Solves for a target factory given a tech level and reference pricing model

        Parameters
        ----------
        targets : CompressedVector
            Target outputs for the factory
        known_technologies : TechnologicalLimitation
            Technology level to use
        reference_model : CompressedVector
            reference pricing model
        uncompiled_cost_function : CostFunction
            Uncompiled cost function to use for this solve
        recovered_run : ColumnTable | None, optional
            The ColumnTable from the last run
        use_manual : bool, optional (default: False)
            Should manual columns be added.
            Warning: its a waste of processing power to turn this on if it doesn't have to be

        Returns
        -------
        tuple[CompressedVector, CompressedVector, CompressedVector, CompressedVector, list[int], float, CompressedVector, CompressedVector, ColumnTable]
            Amount of each construct that should be used,
            Pricing model of resulting factory,
            Pricing model of resulting factory including items that weren't targeted,
            How good each construct is (1 should be maximum, less than 1 indicates loss when used),
            List of indicies of items and fluids of which none were produced,
            Scaling factor difference between input and output pricing model,
            Full item/fluid cost of the created factory,
            Evaluations of unmodded constructs
            The result table used in the last optimization step
        """
        n = len(self.reference_list)

        if DEBUG_BLOCK_MODULES:
            reference_model = CompressedVector({k: v for k, v in reference_model.items() if not '-module' in k})

        p0_j = np.zeros(n, dtype=np.longdouble)
        for k, v in reference_model.items():
            p0_j[self.reference_list.index(k)] = v

        inverse_priced_indices = np.ones(len(self.reference_list))
        inverse_priced_indices[np.array([self.reference_list.index(k) for k in reference_model.keys()], dtype=np.int32)] = 0

        u_j = np.zeros(n, dtype=np.longdouble)
        for k, v in targets.items():
            u_j[self.reference_list.index(k)] = v

        cost_function = lambda construct, point_evaluations: uncompiled_cost_function(p0_j, construct, point_evaluations)

        if use_manual:
            logging.info("Starting a manual program solving.")
            s_i, p_j, R_vector_table = solve_manual_factory_optimization_problem(self.compiled, u_j, cost_function, inverse_priced_indices, known_technologies, ManualConstruct.columns(self._manual_constructs, known_technologies), recovered_run)
        else:
            logging.info("Starting a program solving.")
            s_i, p_j, R_vector_table = solve_factory_optimization_problem(self.compiled, u_j, cost_function, inverse_priced_indices, known_technologies, recovered_run)
        
        logging.debug("Reconstructing factory.")
        s_i[np.where(s_i < 0)] = 0 #<0 is theoretically impossible, and indicates tiny tolerance errors, so we remove them to remove issues

        #scale = 100 / np.max(p_j) #normalization to prevent massive numbers.
        scale = float(1 / np.linalg.norm(p_j))

        p = CompressedVector({k: p_j[self.reference_list.index(k)] * scale for k in targets.keys()})
        p_full = CompressedVector({k: p_j[i] * scale for i, k in enumerate(self.reference_list)})
        
        s = CompressedVector({})
        positives = np.logical_not(np.isclose(s_i, 0, rtol=SOLVER_TOLERANCES['rtol'], atol=SOLVER_TOLERANCES['atol']))
        for i in range(s_i.shape[0]):
            if positives[i]:
                s = s + s_i[i] * R_vector_table.idents[i]
        
        k: CompressedVector = _efficiency_analysis(self.compiled, cost_function, inverse_priced_indices, p_j, known_technologies, R_vector_table.valid_rows, self.post_analyses)

        fc_i = R_vector_table.true_costs @ s_i
        assert np.logical_or(fc_i >= 0, np.isclose(fc_i, 0, rtol=SOLVER_TOLERANCES['rtol'], atol=SOLVER_TOLERANCES['atol'])).all(), fc_i
        full_material_cost = CompressedVector({self.reference_list[i]: fc_i[i] for i in range(fc_i.shape[0]) if fc_i[i]>0})
        for tempk, tempv in full_material_cost.items():
            assert tempk in self.active_list, str(tempk)+": "+str(tempv)+", "+str(R_vector_table.idents[np.where(R_vector_table.true_costs[self.reference_list.index(tempk)]!=0)])

        #nmv, nmc, nmtc, nmi = self.compiled.reduce(cost_function, inverse_priced_indices, None, known_technologies)
        nm_vector_table = self.compiled.columns(cost_function, inverse_priced_indices, None, known_technologies).reduced
        nme = (nm_vector_table.columns.T @ p_j) / nm_vector_table.costs
        nmv = CompressedVector({list(k.keys())[0]: nme[i] for i, k in enumerate(nm_vector_table.idents) if len(k.keys())==1})

        zs = R_vector_table.find_zeros(s_i)

        return s, p, p_full, k, zs, scale, full_material_cost, nmv, R_vector_table
    
    def technological_limitation_from_specification(self, fully_automated: list[str] = [], extra_technologies: list[str] = [], extra_recipes: list[str] = []) -> TechnologicalLimitation:
        """Generates a TechnologicalLimitation from a specification. Works as a more user friendly way of getting useful TechnologicalLimitations.

        Parameters
        ----------
        fully_automated : list[str], optional
            List of fully automated science packs, by default []
        extra_technologies : list[str], optional
            List of additional unlocked technologies, by default []
        extra_recipes : list[str], optional
            List of additionally unlocked recipes, by default []

        Returns
        -------
        TechnologicalLimitation
            Specified TechnologicalLimitation
        """
        return technological_limitation_from_specification(self, fully_automated=fully_automated, extra_technologies=extra_technologies, extra_recipes=extra_recipes)

    def add_post_analysis(self, target_name: str, target_outputs: dict[int, float]):
        """Adds a construct to the special post analysis list

        Parameters
        ----------
        target_name : str
            Construct ident
        target_outputs : dict[int, float]
            Output targets to optimize on for the target
        """        
        self.post_analyses.update({target_name: target_outputs})


def _efficiency_analysis(construct: ComplexConstruct, cost_function: CompiledCostFunction, inverse_priced_indices: np.ndarray, dual_vector: np.ndarray, 
                        known_technologies: TechnologicalLimitation, valid_rows: np.ndarray, post_analyses: dict[str, dict[int, float]]) -> CompressedVector:
    """Constructs an efficency analysis of a ComplexConstruct recursively

    Parameters
    ----------
    construct : ComplexConstruct
        Construct to determine efficencies of
    cost_function : Callable[[CompiledConstruct, np.ndarray]
        A compiled cost function
    inverse_priced_indices : np.ndarray
        What indices of the pricing vector aren't priced
    dual_vector : np.ndarray
        Dual vector to calculate with, if None is given, give the module-less beacon-less design
    known_technologies : TechnologicalLimitation
        Current tech level to calculate for
    valid_rows : np.ndarray
        Outputing rows of the dual

    Returns
    -------
    CompressedVector
        The efficiency analysis
    """        
    efficiencies: CompressedVector = CompressedVector({})
    for sc in construct._subconstructs:
        if isinstance(sc, ComplexConstruct):
            efficiencies.update({sc.ident: sc.efficiency_analysis(cost_function, inverse_priced_indices, dual_vector, known_technologies, valid_rows, post_analyses)})
            efficiencies = efficiencies + _efficiency_analysis(sc, cost_function, inverse_priced_indices, dual_vector, known_technologies, valid_rows, post_analyses)
    return efficiencies


class FactorioFactory():
    """Abstract Class. A factory in Factorio.

    Members
    -------
    _instance : FactorioInstance
        Instance associated with this factory
    _known_technologies : TechnologicalLimitation
        Tech level for this factory
    _targets : CompressedVector
        Target outputs and amounts
    optimal_factory : CompressedVector
        Calculated optimal factory from last calculate_optimal_factory run
    optimal_pricing_model : CompressedVector
        Calculated pricing model of self.optimal_factory
    full_optimal_pricing_model : CompressedVector
        Calculated full pricing model of self.optimal_factory
    _construct_efficiencies : CompressedVector
        Calculated efficiencies of available constructs
    _zero_throughputs : list[int]
        List of indicies of items and fluids of which none were produced in optimal factory
    optimized : bool
        If an optimal factory has been calculated since targets were last updated
    _intermediate_scale_factor : float
        Scaling factor difference between input and output pricing model
    true_cost : CompressedVector
        Full item/fluid cost of the optimal factory
    _no_module_value : CompressedVector
        Evaluations of unmodded constructs
    last_run_columns : ColumnTable | None
        Columns used in the last run to pass to solver when retargeting is done
    """
    _instance: FactorioInstance
    _known_technologies: TechnologicalLimitation
    _targets: CompressedVector
    optimal_factory: CompressedVector
    optimal_pricing_model: CompressedVector
    full_optimal_pricing_model: CompressedVector
    _construct_efficiencies: CompressedVector
    _zero_throughputs: list[int]
    optimized: bool
    _intermediate_scale_factor: float
    true_cost: CompressedVector
    _no_module_value: CompressedVector
    last_run_columns: ColumnTable | None

    def __init__(self, instance: FactorioInstance, known_technologies: TechnologicalLimitation, targets: CompressedVector) -> None:
        """
        Parameters
        ----------
        instance : FactorioInstance
            Instance associated with this factory
        known_technologies : TechnologicalLimitation
            Tech level for this factory
        targets : CompressedVector
            Target outputs and amounts
        """
        assert isinstance(instance, FactorioInstance)
        assert isinstance(known_technologies, TechnologicalLimitation)
        assert isinstance(targets, CompressedVector)
        self._instance = instance
        self._known_technologies = known_technologies
        self._targets = targets.norm()
        self.optimal_factory = CompressedVector()
        self.optimal_pricing_model = CompressedVector()
        self.full_optimal_pricing_model = CompressedVector()
        self.optimized = False
        self._intermediate_scale_factor = 0
        self.true_cost = CompressedVector()
        self._no_module_value = CompressedVector()
        self.last_run_columns = None

    def calculate_optimal_factory(self, reference_model: CompressedVector, uncompiled_cost_function: CostFunction, use_manual: bool = False) -> bool:
        """Calculates a optimal factory with the the reference model

        Parameters
        ----------
        reference_model : CompressedVector
            CompressedVector of reference pricing model
        uncompiled_cost_function : CostFunction
            Cost function to use for this factory optimization
        use_manual : bool, optional (default: False)
            Should manual construct be added in for calculating the optimal factory

        Returns
        -------
        bool
            If pricing model has changed since this factory was last optimized
        """
        s, p, pf, k, zs, scale, tc, nmv, recovery = self._instance.solve_for_target(self._targets, self._known_technologies, reference_model, uncompiled_cost_function, self.last_run_columns, use_manual=use_manual)
        same = p==self.optimal_pricing_model
        self.optimal_factory, self.optimal_pricing_model, self.full_optimal_pricing_model, self._construct_efficiencies, self._intermediate_scale_factor, self.true_cost, self._no_module_value = s, p, pf, k, scale, tc, nmv
        self.last_run_columns = recovery
        self._zero_throughputs = zs
        self.optimized = True

        for k, v in self.true_cost.items():
            assert k in reference_model.keys(), k+" w/ "+str(v)

        return not same

    def retarget(self, targets: CompressedVector, retainment: float = BASELINE_RETAINMENT_VALUE) -> None:
        """Rebuilds targets. This is useful if iteratively optimizing a built chain.
        After this, one should re-run calculate_optimal_factory and if it returns False then the pricing model didn't change even after retargeting.

        Parameters
        ----------
        targets : CompressedVector
            CompressedVector of the new target outputs
        retainment : float, optional
            How much building of targets that used to exist should be retained. 
            With 0 retainment this factory may start mispricing those targets. 
            By default global RETAINMENT_VALUE
        """
        for k in self._targets.keys():
            if not k in targets.keys():
                targets[k] = retainment
        self._targets = targets.norm()

        self.optimized = False
    
    def dump_to_excel(self, writer: pd.ExcelWriter, sheet_name: str) -> None:
        """Dumps the target, optimal factory, optimal pricing model, and inefficient constructs into an excel spreadsheet sheet

        Parameters
        ----------
        writer : pd.ExcelWriter
            Excel writer to use
        sheet_name : str
            What sheet to write to

        Raises
        ------
        RuntimeError
            If the factory isn't optimized
        """
        if not self.optimized:
            raise RuntimeError("Dump asked but factory isn't nessisarily correct.")
        targets_df = pd.DataFrame(list(self._targets.items()), columns=['target', 'count'])
        optimal_factory_df = pd.DataFrame(list(self.optimal_factory.items()), columns=['construct', 'count'])
        optimal_pricing_model_df = pd.DataFrame(list(self.optimal_pricing_model.items()), columns=['item', 'value'])
        inefficient_constructs_df = pd.DataFrame(list(self._construct_efficiencies.items()), columns=['construct', 'relative value'])
        transport_df = pd.DataFrame(list(compute_transportation_densities(self.full_optimal_pricing_model, self._instance._data_raw)), columns=['item', 'via', 'relative density'])
        no_module_df = pd.DataFrame(list(self._no_module_value.items()), columns=['construct', 'relative value'])
        merged_df = pd.concat([targets_df, pd.DataFrame({}, columns=['']), 
                            optimal_factory_df, pd.DataFrame({}, columns=['']),
                            optimal_pricing_model_df, pd.DataFrame({}, columns=['']),
                            inefficient_constructs_df, pd.DataFrame({}, columns=['']),
                            transport_df, pd.DataFrame({}, columns=['']),
                            no_module_df], axis=1)
        merged_df.to_excel(writer, sheet_name=sheet_name, index=False)


class FactorioMaterialFactory(FactorioFactory):
    """A factory in factorio that makes material parts for future factories.

    Added Members
    -------------
    _last_material_factory : FactorioMaterialFactory | InitialFactory
        Factory to base the pricing model of this factory on
    """
    _last_material_factory: FactorioMaterialFactory | InitialFactory

    def __init__(self, instance: FactorioInstance, previous_science: FactorioScienceFactory | InitialFactory | TechnologicalLimitation, 
                 last_material_factory: FactorioMaterialFactory | InitialFactory, material_targets: CompressedVector) -> None:
        """
        Parameters
        ----------
        instance : FactorioInstance
            Instance associated with this factory
        previous_science : FactorioScienceFactory | InitialFactory | TechnologicalLimitation
            Factory to base current science coverage on or tech limit of current science coverage
        last_material_factory : FactorioMaterialFactory | InitialFactory
            Factory to base the pricing model of this factory on
        material_targets : CompressedVector
            Outputs of the factory, doesn't contain research
        """
        assert isinstance(instance, FactorioInstance)
        assert isinstance(previous_science, FactorioScienceFactory) or isinstance(previous_science, InitialFactory) or isinstance(previous_science, TechnologicalLimitation)
        assert isinstance(last_material_factory, FactorioMaterialFactory) or isinstance(last_material_factory, InitialFactory)
        assert isinstance(material_targets, CompressedVector)
        if isinstance(previous_science, FactorioScienceFactory) or isinstance(previous_science, InitialFactory):
            super().__init__(instance, previous_science.tech_coverage, material_targets)
        else: #we probably got here via self.startup_base
            super().__init__(instance, previous_science, material_targets)
        self._last_material_factory = last_material_factory


class FactorioManualFactory(FactorioMaterialFactory):
    """A factory with manual crafting
    """

    def calculate_optimal_factory(self, reference_model: CompressedVector, uncompiled_cost_function: CostFunction, use_manual: bool = False) -> bool:
        return super().calculate_optimal_factory(reference_model, uncompiled_cost_function, True)


class FactorioScienceFactory(FactorioFactory):
    """A factory in factorio that completes research.

    Added Members
    -------------
    _last_material_factory : FactorioMaterialFactory
        Factory to base the pricing model of this factory on
    _previous_science : FactorioScienceFactory | InitialFactory | TechnologicalLimitation
        Last science factory or tech limit
    _clear : list[str]
        Set of tools that define which must be produced to clear all researches possible with just those tools.
        Example: If this was a set of red and green science pack that would indicate all research that only red and
                 green science packs are needed for MUST be completed within this factory, regardless of other calculations
    _extra : list[str]
        Set of extra technologies to research
    """
    _last_material_factory: FactorioMaterialFactory
    _previous_science: FactorioScienceFactory | InitialFactory | TechnologicalLimitation
    _clear: list[str]
    _extra: list[str]

    def __init__(self, instance: FactorioInstance, previous_science: FactorioScienceFactory | InitialFactory | TechnologicalLimitation, last_material_factory: FactorioMaterialFactory | InitialFactory, 
                 science_targets: CompressedVector) -> None:
        """
        Parameters
        ----------
        instance : FactorioInstance
            Instance associated with this factory
        previous_science : FactorioScienceFactory | InitialFactory | TechnologicalLimitation
            Last science factory or tech limit
        last_material_factory : FactorioMaterialFactory | InitialFactory
            Factory to base the pricing model of this factory on
        science_targets : CompressedVector
            Target tools defining a clear
        """        
        assert isinstance(instance, FactorioInstance)
        assert isinstance(previous_science, FactorioScienceFactory) or isinstance(previous_science, InitialFactory) or isinstance(previous_science, TechnologicalLimitation)
        assert isinstance(last_material_factory, FactorioMaterialFactory) or isinstance(last_material_factory, InitialFactory)
        assert isinstance(science_targets, CompressedVector)
        self._clear = [target for target in science_targets.keys() if target in instance._data_raw['tool'].keys()]
        self._extra = [target[:-1 * len(RESEARCH_SPECIAL_STRING)] for target in science_targets.keys() if RESEARCH_SPECIAL_STRING in target]
        self._previous_science = previous_science
        
        targets, covering_to, last_coverage  = _science_factory_parameters(instance, previous_science, self._clear, self._extra)

        super().__init__(instance, last_coverage, targets)

        self._last_material_factory = last_material_factory
    
    def _previous_coverage(self) -> TechnologicalLimitation:
        """
        Returns
        -------
        TechnologicalLimitation
            Last science coverage
        """        
        if isinstance(self._previous_science, TechnologicalLimitation):
            return self._previous_science
        else:
            return self._previous_science.tech_coverage
        
    @property
    def tech_coverage(self) -> TechnologicalLimitation:
        """
        Returns
        -------
        TechnologicalLimitation
            Tech level that will be unlocked when this factory is done
        """
        return self._previous_coverage() + TechnologicalLimitation(self._instance._tech_tree, [set([targ[:targ.rfind("=")] for targ in self._targets.keys()])])

    def retarget(self, targets: CompressedVector, retainment: float = BASELINE_RETAINMENT_VALUE) -> None:
        """Rebuilds targets. This is useful if iteratively optimizing a built chain.
        After this, one should re-run calculate_optimal_factory and if it returns False then the pricing model didn't change even after retargeting.
        Will make sure all technologies cleared by self.clear set are still in targets.

        Parameters
        ----------
        targets : CompressedVector
            CompressedVector of the new target outputs
        retainment : float, optional
            How much building of targets that used to exist should be retained. 
            With 0 retainment this factory may start mispricing those targets. 
            By default global RETAINMENT_VALUE
        """
        #assert not any([target in self.instance.data_raw['tool'].keys() for target in targets.keys()]), "retarget should NEVER be given a tool. Only researches."
        assert all([t in self._targets.keys() for t in targets]), "retarget should never add new targets... yet."
        covering_to: TechnologicalLimitation = self._instance.technological_limitation_from_specification(fully_automated=self._clear) + \
                      TechnologicalLimitation(self._instance._tech_tree, [set([target for target in targets.keys()])])
        last_coverage: TechnologicalLimitation = self._previous_coverage()

        raise NotImplementedError("actually fix this line this time.")
        targets = CompressedVector({k: 1 / self.time_target for k in next(iter(covering_to.canonical_form)) if k not in next(iter(last_coverage))}) # type: ignore

        super().retarget(targets, retainment=retainment)


class InitialFactory(FactorioMaterialFactory, FactorioScienceFactory):
    """A fake factory instance to hold an initial pricing model and tech level for full automation
    """

    def __init__(self, instance: FactorioInstance, pricing_model: CompressedVector, known_technologies: TechnologicalLimitation) -> None:
        """
        Parameters
        ----------
        instance : FactorioInstance
            Instance associated with this factory
        pricing_model : CompressedVector
            Baseline pricing model
        targets : CompressedVector
            TechnologicalLimitation baseline, usually enough to begin full automatization
        """
        assert isinstance(instance, FactorioInstance)
        assert isinstance(pricing_model, CompressedVector)
        assert isinstance(known_technologies, TechnologicalLimitation)
        self._instance = instance
        self._known_technologies = known_technologies
        self._targets = pricing_model
        self.optimal_factory = CompressedVector()
        self.optimal_pricing_model = pricing_model.norm()
        self._construct_efficiencies = CompressedVector()
        self._previous_science = known_technologies
    
    def calculate_optimal_factory(self, reference_model: CompressedVector, uncompiled_cost_function: CostFunction, use_manual: bool = False) -> bool:
        """Placeholder. Initial Factories cannot change.
        """
        return False
    
    @property
    def tech_coverage(self) -> TechnologicalLimitation:
        """
        Returns
        -------
        TechnologicalLimitation
            Tech level that will be unlocked when this factory is done
        """
        return self._previous_coverage()
    
    def retarget(self, targets: CompressedVector, retainment: float = BASELINE_RETAINMENT_VALUE) -> None:
        """Placeholder. Initial Factories cannot change.
        """
        return None
    

class FactorioFactoryChain():
    """A chain of optimal factory designs starting from the minimal science to a specified point. There are two avaiable types of factory in the chain:

        A science factory that produces a set of science packs. Having one of these will permit any recipe that is unlockable with those sciences to be used in factories after it.
        
        A material factory that produces a set of buildings and catalyst materials. Having one of these will set the pricing model for every factory in the chain up until the next component factory.
    
    Members
    -------
    _instance : FactorioInstance
        Instance for this chain
    _chain :list[FactorioFactory]
        List containing the chain elements
    _uncompiled_cost_function : CostFunction
        Cost function this chain uses
    """
    _instance: FactorioInstance
    _chain: list[FactorioFactory]
    _uncompiled_cost_function: CostFunction

    def __init__(self, instance: FactorioInstance, uncompiled_cost_function: CostFunction) -> None:
        """
        Parameters
        ----------
        instance : FactorioInstance
            Instance for this chain
        uncompiled_cost_function : CostFunction
            Cost function to use for this factory chain
        """
        assert isinstance(instance, FactorioInstance)
        self._instance = instance
        self._chain = []
        self._uncompiled_cost_function = uncompiled_cost_function
    
    def initial_pricing(self, pricing_model: CompressedVector, starting_techs: TechnologicalLimitation) -> None:
        """Initialize the first factory of the chain as an InitialFactory with a pricing model and tech level

        Parameters
        ----------
        pricing_model : CompressedVector
            CompressedVector that describes what the pricing model should be
        starting_techs : TechnologicalLimitation
            TechnologicalLimitation of techs unlocked before building a starting factory
        """
        self._chain.append(InitialFactory(self._instance, pricing_model, starting_techs))
    
    def add(self, targets: CompressedVector | str | None = None) -> bool:
        """Adds a factory to the chain. Won't add a factory if material targets are given and all materials are already priced
        
        Parameters
        ----------
        targets : CompressedVector | str | None
            Either:
                CompressedVector of target outputs for the factory. Must be either all science tools or building materials
            Or:
                String:
                    "all materials" = will autopopulate all possible buildings and active materials that can be made
                    "all tech" = will autopopulate all possible tech that can be made
                    "" = will autopopulate all tech if possible, otherwise autopopulate all possible buildings and active materials
            Or (default):
                None = String ""

        Raises
        ------
        ValueError
            Something very bad happened

        Returns
        -------
        If the addition actually added a factory
        """
        factory_type: str = ""
        previous_sciences = [fac for fac in self._chain if isinstance(fac, FactorioScienceFactory)]
        last_material = [fac for fac in self._chain if isinstance(fac, FactorioMaterialFactory)][-1]
        if len(previous_sciences)==0:
            known_technologies: TechnologicalLimitation = last_material._known_technologies #first science after a startup base.
        else:
            known_technologies: TechnologicalLimitation = previous_sciences[-1].tech_coverage

        if isinstance(targets, CompressedVector):
            if list(targets.keys())[0] in self._instance._data_raw['tool'].keys() or list(targets.keys())[0] in self._instance._data_raw['technology'].keys():
                factory_type = "science"
            elif list(targets.keys())[0] in self._instance._data_raw['item'].keys() or list(targets.keys())[0] in self._instance._data_raw['fluid'].keys():
                factory_type = "material"
        else:
            if targets is None:
                targets = ""
            targets, factory_type = _calculate_new_full_factory_targets(targets, last_material, known_technologies, self._instance)

        if factory_type == "science":
            for target in targets.keys():
                assert target in self._instance._data_raw['tool'].keys() or target in self._instance._data_raw['technology'].keys(), "If a factory does science stuff is only allowed to do science stuff."
            previous_science = previous_sciences[-1] if len(previous_sciences)>0 else known_technologies
            self._chain.append(FactorioScienceFactory(self._instance, previous_science, last_material, targets))
        elif factory_type == "material":
            difference: set[str] = set(targets.keys()).difference(set(last_material._targets.keys()))
            logging.info("Attempting to add the following targets to the new material factory:\n\t"+str(difference))
            last_science = previous_sciences[-1]
            self._chain.append(FactorioMaterialFactory(self._instance, last_science, last_material, targets))
        else: #factory_type == "manual"
            difference: set[str] = set(targets.keys()).difference(set(last_material._targets.keys()))
            if len(difference)==0:
                logging.info("Adding a manual factory without any actual additional prices. This would indicate that there is no more progress to be made.")
                return True
            logging.info("Attempting to add the following targets to the new manual factory:\n\t"+str(difference))
            last_science = previous_sciences[-1]
            self._chain.append(FactorioManualFactory(self._instance, last_science, last_material, targets))
        return False

    def complete(self) -> None:
        """Completes the chain, adding factories until no more progress can be made
        """
        i = 0
        while not self.add():
            i += 1
            if i>100:
                raise RuntimeError("Chain autocomplete seems to be infinite?")

    def compute_all(self) -> bool:
        """Computes all pricing models for chain iteratively and returns if the pricing models changed

        Returns
        -------
        bool
            If pricing model has been changed in any factory
        """
        changed = False

        last_reference_model = self._chain[0].optimal_pricing_model
        for i, factory in enumerate(self._chain[1:]):
            logging.info("Computing factory "+str(i))
            changed = factory.calculate_optimal_factory(last_reference_model, self._uncompiled_cost_function) or changed
            if isinstance(factory, FactorioMaterialFactory):
                last_reference_model = factory.optimal_pricing_model

        return changed

    def retarget_all(self) -> bool:
        """Retargets and recomputes all pricing models for chain iteratively and returns if the pricing models changed.

        Returns
        -------
        bool
            If pricing model has been changed in any factory
        """
        changed = False

        last_reference_model = self._chain[0].optimal_pricing_model
        for i, factory in enumerate(self._chain[1:]):
            logging.info("Retargeting factory "+str(i))
            if isinstance(factory, FactorioMaterialFactory):
                updated_targets = CompressedVector()
                for j in range(i+1, len(self._chain)):
                    for k, v in self._chain[j].true_cost.items():
                        assert k in self._instance.active_list, str(k)+": "+str(v)+", "+str(j)+" t: "+str(type(self._chain[j]))
                    updated_targets = updated_targets + self._chain[j].true_cost
                    if isinstance(self._chain[j], FactorioMaterialFactory):
                        break #done with this material factories outputs
                logging.debug(last_reference_model)
                logging.debug(updated_targets)
                logging.debug(factory.optimal_pricing_model)
                factory.retarget(updated_targets)
                for k, v in updated_targets.items():
                    assert k in factory.optimal_pricing_model.keys(), k

                changed = factory.calculate_optimal_factory(last_reference_model, self._uncompiled_cost_function) or changed
                last_reference_model = factory.optimal_pricing_model

            elif isinstance(factory, FactorioScienceFactory):
                changed = factory.calculate_optimal_factory(last_reference_model, self._uncompiled_cost_function) or changed
                pass #we don't update science factories at the moment
        
        return changed
            
    def dump_to_excel(self, file_name: str) -> None:
        """Dumps the entire chain into an excel spreadsheet

        Parameters
        ----------
        file_name : str
            Name of excel file to write to
        """
        writer = pd.ExcelWriter(file_name)
        manual_factory_ident = 1
        material_factory_ident = 1
        science_factory_ident = 1
        for factory in self._chain[1:]:
            if isinstance(factory, FactorioManualFactory):
                if not factory.optimized:
                    logging.error("Found Manual Factory "+str(manual_factory_ident)+" unoptimized.")
                else:
                    factory.dump_to_excel(writer, "Manual Factory "+str(manual_factory_ident))
                manual_factory_ident += 1
            elif isinstance(factory, FactorioMaterialFactory):
                if not factory.optimized:
                    logging.error("Found Material Factory "+str(material_factory_ident)+" unoptimized.")
                else:
                    factory.dump_to_excel(writer, "Material Factory "+str(material_factory_ident))
                material_factory_ident += 1
            elif isinstance(factory, FactorioScienceFactory):
                if not factory.optimized:
                    logging.error("Found Science Factory "+str(science_factory_ident)+" unoptimized.")
                else:
                    factory.dump_to_excel(writer, "Science Factory "+str(science_factory_ident))
                science_factory_ident += 1
            #else: #if isinstance(factory, InitialFactory):
        writer.close()

def _calculate_new_full_factory_targets(target_types: str, last_material: FactorioMaterialFactory,
                                        known_technologies: TechnologicalLimitation, instance: FactorioInstance) -> tuple[CompressedVector, str]:
    """Calculates the targets for a full factory type. 
    If no target is given it will create a science one if research can be done, otherwise it will crate a material one

    Parameters
    ----------
    target_types : str
        What factory targets to create.
            "all tech" for a science factory
            "all materials" for a material factory
            Anything else (usually "") means that the function should choose whatever gives progress
    last_material : FactorioMaterialFactory
        The last material factory for pricing
    known_technologies : TechnologicalLimitation
        The tech level of the factory
    instance : FactorioInstance
        The Factorio instance to use

    Returns
    -------
    CompressedVector
        Targets in a compressed vector
    str
        The factory type
    """
    output_items = list(last_material._targets.keys())
    #logging.info("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
    #logging.info(output_items)

    cost_function: CompiledCostFunction = lambda construct, point_evaluations: np.zeros(point_evaluations.beacon_cost.shape[0])

    inverse_priced_indices = np.ones(len(instance.reference_list))
    inverse_priced_indices[np.array([instance.reference_list.index(k) for k in output_items], dtype=np.int32)] = 0

    R_vector_table = instance.compiled.columns(cost_function, 
                                                            inverse_priced_indices, 
                                                            None, known_technologies).reduced
    #logging.info([k for k in N_i])

    if target_types=="all tech":
        factory_type = "science"
        target_names: list[str] = _new_science_factory_targets(instance, R_vector_table.valid_rows)
    elif target_types=="all materials":
        factory_type = "material"
        target_names: list[str] = _new_material_factory_targets(instance, R_vector_table.valid_rows)
    else:
        science_target_names: list[str] = _new_science_factory_targets(instance, R_vector_table.valid_rows)
        science_targets, covering_to, _ = _science_factory_parameters(instance, known_technologies, science_target_names, [])
        if len(science_targets.keys())>0:
            factory_type = "science"
            target_names = science_target_names
        else:
            material_target_names: list[str] = _new_material_factory_targets(instance, R_vector_table.valid_rows)
            #logging.info(material_target_names)
            difference: set[str] = set(material_target_names).difference(set(last_material._targets.keys()))
            #logging.info(difference)
            if len(difference)>0 and (not isinstance(last_material, FactorioManualFactory) or len(material_target_names)>len(last_material._targets.keys())): 
                #second check here is due to needing to keep using manual factories until material factories can actually take over
                factory_type = "material"
                target_names = material_target_names
            else:
                factory_type = "manual"
                manual_constructs = ManualConstruct.columns(instance._manual_constructs, known_technologies)
                R_vector_table = instance.compiled.columns(cost_function, 
                                                            inverse_priced_indices, 
                                                            None, known_technologies).shadow_attachment(manual_constructs).reduced
                target_names = _new_material_factory_targets(instance, R_vector_table.valid_rows)# + ['electric']
                #logging.info(target_names)

    logging.info("Auto-targets:\n\t"+str(target_names))

    return CompressedVector({k: Fraction(1) for k in target_names}), factory_type

def _new_science_factory_targets(instance: FactorioInstance, valid_rows: np.ndarray) -> list[str]:
    """Determines all possible tool targets for a science factory

    Parameters
    ----------
    instance : FactorioInstance
        The Factorio instance to use
    valid_rows : np.ndarray
        What values of the reference list can be produced

    Returns
    -------
    list[str]
        All possible tools in the valid_rows
    """    
    target_names: list[str] = []
    for tool in instance._data_raw['tool'].keys():
        if valid_rows[instance.reference_list.index(tool)]:
            target_names.append(tool)
    return target_names

def _science_factory_parameters(instance: FactorioInstance, previous_science: FactorioScienceFactory | InitialFactory | TechnologicalLimitation, 
                                clear: list[str], extra: list[str]) -> tuple[CompressedVector, TechnologicalLimitation, TechnologicalLimitation]:
    """Determines the targets, and coverages given some science factory parameters

    Parameters
    ----------
    instance : FactorioInstance
        The Factorio instance to use
    previous_science : FactorioScienceFactory | InitialFactory | TechnologicalLimitation
        Holder of previous science state
    clear : list[str]
        What tools are fully automated and all research requiring just them should be done
    extra : list[str]
        Extra science that must be done

    Returns
    -------
    CompressedVector
        The targets of the science factory
    TechnologicalLimitation
        The ending tech level
    TechnologicalLimitation
        The starting tech level
    """    
    covering_to: TechnologicalLimitation = instance.technological_limitation_from_specification(fully_automated=clear, extra_technologies=extra)

    last_coverage: TechnologicalLimitation = previous_science.tech_coverage

    targets = CompressedVector({instance._tech_tree._inverse_map[k]+RESEARCH_SPECIAL_STRING: 1 for k in next(iter(covering_to.canonical_form)) if k not in next(iter(last_coverage.canonical_form))}) #next(iter()) gives us the first (and theoretically only) set of nodes making up the tech limit

    return targets, covering_to, last_coverage

def _new_material_factory_targets(instance: FactorioInstance, valid_rows: np.ndarray) -> list[str]:
    """Determines all possible material targets for a material factory

    Parameters
    ----------
    instance : FactorioInstance
        The Factorio instance to use
    valid_rows : np.ndarray
        What values of the reference list can be produced

    Returns
    -------
    list[str]
        All possible active materials in the valid_rows

    Raises
    ------
    ValueError
        Various build issues
    """    
    target_names: list[str] = []
    for item_cata in ITEM_SUB_PROTOTYPES:
        if item_cata=='tool':
            continue #skip tools in material factories
        for item in instance._data_raw[item_cata].keys():
            if not item in instance.reference_list:
                if not item in OUTPUT_WARNING_LIST:
                    logging.warning("Detected some a weird "+item_cata+": "+item)
                    OUTPUT_WARNING_LIST.append(item)
            elif valid_rows[instance.reference_list.index(item)] and item in instance.active_list:
                target_names.append(item)
    for fluid in instance._data_raw['fluid'].keys():
        if fluid in instance.RELEVENT_FLUID_TEMPERATURES.keys():
            for temp in instance.RELEVENT_FLUID_TEMPERATURES[fluid].keys():
                if not fluid+'@'+str(temp) in instance.reference_list:
                    raise ValueError("Fluid \""+fluid+"\" found to have temperature "+str(temp)+" but said temperature wasn't found in the reference list.")
                if valid_rows[instance.reference_list.index(fluid+'@'+str(temp))] and fluid+'@'+str(temp) in instance.active_list:
                    target_names.append(fluid+'@'+str(temp))
        else:
            if not fluid in instance.reference_list:
                if not fluid in OUTPUT_WARNING_LIST:
                    logging.warning("Detected some a weird fluid: "+fluid)
                    OUTPUT_WARNING_LIST.append(fluid)
            elif valid_rows[instance.reference_list.index(fluid)] and fluid in instance.active_list:
                target_names.append(fluid)
    for other in ['electric', 'heat']:
        if not other in instance.reference_list:
            if not other in OUTPUT_WARNING_LIST:
                logging.warning("Was unable to find "+other+" in the reference list. While not nessisary wrong this is extreamly odd and should only happen on very strange mod setups.")
                OUTPUT_WARNING_LIST.append(other)
        if valid_rows[instance.reference_list.index(other)] and other in instance.active_list:
            target_names.append(other)
    return target_names


