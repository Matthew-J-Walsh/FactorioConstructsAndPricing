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


class FactorioInstance():
    """
    Holds the information in an instance (specific game mod setup) after completing premanagment steps.

    Members
    -------
    data_raw:
        Whole data.raw dictonary post-premanagment.
    uncompiled_constructs:
        List of all UncompiledConstructs for the game instance.
    complex_constructs:
        List of all SingularConstructs for the game instance.
    disabled_constructs:
        List of non-permitted SingularConstruct idents. Used to force the usage of a complex construct.
    compiled:
        ComplexConstruct of the entire instance or None if hasn't been compiled since last edit.
    reference_list:
        List containing every relevent value in any CompressedVector in the instance.
    catalyst_list:
        List of all catalytic item/fluids.
    COST_MODE:
        What cost mode is being used. https://lua-api.factorio.com/latest/concepts.html#DifficultySettings
    DEFAULT_TARGET_RESEARCH_TIME:
        Default time that should science factories should complete their science in.
    RELEVENT_FLUID_TEMPERATURES:
        Dict with keys of fluid names and values of a dict mapping temperatures to energy densities.
    """
    data_raw: dict
    tech_tree: TechnologyTree
    uncompiled_constructs: tuple[UncompiledConstruct, ...]
    complex_constructs: list[ComplexConstruct]
    disabled_constructs: list[ComplexConstruct]
    compiled: ComplexConstruct | None
    reference_list: tuple[str, ...]
    catalyst_list: tuple[str, ...]
    spatial_pricing: np.ndarray
    COST_MODE: str
    DEFAULT_TARGET_RESEARCH_TIME: Fraction
    RELEVENT_FLUID_TEMPERATURES: dict

    def __init__(self, filename: str, COST_MODE: str = 'normal', DEFAULT_TARGET_RESEARCH_TIME: Fraction = Fraction(10000), nobuild: bool = False) -> None:
        """
        Parameters
        ----------
        filename:
            Filename of data.raw to load for this instance.
        COST_MODE:
            What cost mode should be used. https://lua-api.factorio.com/latest/concepts.html#DifficultySettings
        DEFAULT_TARGET_RESEARCH_TIME:
            Default time that should science factories should complete their science in.
        nobuild:
            Don't build complex constructs, only load in data raw and manage.
        """
        with open(filename) as f:
            self.data_raw = json.load(f)
        
        self.COST_MODE = COST_MODE
        self.DEFAULT_TARGET_RESEARCH_TIME = DEFAULT_TARGET_RESEARCH_TIME
        self.RELEVENT_FLUID_TEMPERATURES = {}

        self.tech_tree = complete_premanagement(self.data_raw, self.RELEVENT_FLUID_TEMPERATURES, self.COST_MODE)
        logging.info("Building uncompiled constructs.")
        self.uncompiled_constructs = generate_all_constructs(self.data_raw, self.RELEVENT_FLUID_TEMPERATURES, self.COST_MODE)
        logging.info("Building reference and catalyst lists.")
        self.reference_list, self.catalyst_list = generate_references_and_catalysts(self.uncompiled_constructs)
        for k in self.reference_list:
            DEBUG_REFERENCE_LIST.append(k)
        
        self.disabled_constructs = []

        self.spatial_pricing = np.zeros(len(self.reference_list))
        for mining_drill in self.data_raw['mining-drill'].values():
            self.spatial_pricing[self.reference_list.index(mining_drill['name'])] = mining_drill['tile_width'] * mining_drill['tile_height']
        for beacon in self.data_raw['beacon'].values():
            self.spatial_pricing[self.reference_list.index(beacon['name'])] = beacon['tile_width'] * beacon['tile_height']

        if not nobuild:
            logging.info("Building complex constructs.")
            self.complex_constructs = [SingularConstruct(CompiledConstruct(uc, self)) for uc in self.uncompiled_constructs]
            self.compiled = None
            self.compile()
    
    def compile(self) -> ComplexConstruct:
        """
        Populates compiled and returns it.
        """
        if self.compiled is None:
            self.compiled = ComplexConstruct(tuple([cc for cc in self.complex_constructs if not cc in self.disabled_constructs]), "Whole Game Construct") # type: ignore
            #self.compiled_constructs = [CompiledConstruct(uc, self) for uc in self.uncompiled_constructs]
        return self.compiled

    def disable_complex_construct(self, target_name: str) -> None:
        """
        Disables the closest named complex construct. Searches using Levenshtein distance.

        Parameters
        ----------
        target_name:
            Target name of complex construct to be disabled.
        """
        self.disabled_constructs.append(self.search_complex_constructs(target_name))
        self.compiled = None
    
    def enable_complex_construct(self, target_name: str) -> None:
        """
        Disables the closest named complex construct. Searches using Levenshtein distance.

        Parameters
        ----------
        target_name:
            Target name of complex construct to be disabled.
        """
        construct = self.search_complex_constructs(target_name)
        try:
            self.disabled_constructs.remove(construct)
            self.compiled = None
        except:
            logging.warning(construct.ident+" was not disabled in the first place.")

    def search_complex_constructs(self, target_name: str) -> ComplexConstruct:
        """
        Finds the closest named complex construct

        Parameters
        ----------
        target_name:
            Target name of complex construct to be disabled.

        Returns
        -------
        ident of closest matching ComplexConstruct
        """
        best_matches: list[ComplexConstruct] = []
        match_distance = Levenshtein.distance(target_name, self.complex_constructs[0].ident)
        logging.debug("Atempting to translate: "+"\""+target_name+"\" starting distance is: "+str(match_distance)+" there "+("is" if target_name in [c.ident for c in self.complex_constructs] else "is not")+" a 0 length translation.")
        for c in self.complex_constructs:
            dist = Levenshtein.distance(target_name, c.ident)
            if dist < match_distance:
                best_matches = [c]
                match_distance = dist
                logging.debug("\tFound a new best match in: \""+c.ident+"\" at distance: "+str(match_distance))
            elif dist == match_distance:
                best_matches.append(c)
                logging.debug("\t\tAdded the new possible match: \""+c.ident+"\"")
            elif c.ident == target_name:
                logging.debug("Found the exact match but chose to ignore it because im a dumb program "+str(dist))
                raise ValueError
        assert len(best_matches)==1, "Unable to determine which construct an input phrase is associated with.\nPhrase is: "+target_name+"\nPossible constructs were:\n\t"+"\n\t".join([str(m) for m in best_matches])+"\n\tWith distance: "+str(match_distance)
        logging.debug("Translated: \""+target_name+"\" to mean the construct: \""+str(best_matches[0])+"\"")
        return best_matches[0]

    def bind_complex_constructs(self, target_names: list[str | tuple[str, bool]]) -> str:
        """
        Binds a list of complex constructs together, disables marked ones, and returns the new name they are under.

        Parameters
        ----------
        target_names:
            List of names and if the original construct should be disabled. If no bool is provided "True" is assumed.

        Returns
        -------
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
        
        new_name = "Combined Construct["+" and ".join([construct.ident for construct in constructs])+"]"
        self.complex_constructs.append(ComplexConstruct(tuple(constructs), new_name)) # type: ignore
        for i, construct in enumerate(constructs):
            if disable_list[i]:
                self.disable_complex_construct(construct.ident)
        
        return new_name

    def solve_for_target(self, targets: CompressedVector, known_technologies: TechnologicalLimitation, reference_model: CompressedVector, 
                         primal_guess: CompressedVector | None = None, dual_guess: CompressedVector | None = None, spatial_mode: bool = False) -> tuple[CompressedVector, CompressedVector, CompressedVector, CompressedVector, list[int], float, CompressedVector]:
        """
        Solves for a target output vector given a tech level and reference pricing model.

        Parameters
        ----------
        targets:
            CompressedVector representing what outputs the factory should have.
        known_technologies:
            TechnologicalLimitation representing what technologies are done.
        reference_model:
            CompressedVector of the reference to use to decide how much every component costs.
        
        Returns
        -------
        s:
            CompressedVector of amount of each construct that should be used.
        p:
            CompressedVector of pricing model of resulting factory.
        k:
            CompressedVector of how good each construct is. (1 should be maximum, less than 1 indicates loss when used)
        """
        n = len(self.reference_list)

        if DEBUG_BLOCK_MODULES:
            reference_model = CompressedVector({k: v for k, v in reference_model.items() if not '-module' in k})
        self.compile()
        assert not self.compiled is None

        p0_j = np.zeros(n, dtype=np.longdouble)
        for k, v in reference_model.items():
            p0_j[self.reference_list.index(k)] = v
        priced_indices =  np.array([self.reference_list.index(k) for k in reference_model.keys()])

        u_j = np.zeros(n, dtype=np.longdouble)
        for k, v in targets.items():
            u_j[self.reference_list.index(k)] = v

        if not primal_guess is None:
            logging.error("Primal guess not supported yet. We have to ignore it.")
        
        ginv_j = None
        if not dual_guess is None:
            ginv_j = np.zeros(len(self.reference_list))
            for k, v in dual_guess.items():
                ginv_j[self.reference_list.index(k)] = v

        if spatial_mode:
            logging.info("Starting a two-phase dual problem.")
            #s_i, p_j, R_j_i, C_j_i, N_i = solve_spatial_mode_factory_optimization_problem(self.compiled, u_j, self.spatial_pricing, p0_j, priced_indices, known_technologies, ginv_j)
            s_i, p_j, R_j_i, C_j_i, N_i = solve_factory_optimization_problem(self.compiled, u_j, self.spatial_pricing, priced_indices, known_technologies, ginv_j, spatial_mode=True)
            c_i = C_j_i.T @ self.spatial_pricing
        else:
            #raise ValueError("TESTING, REMOVE ME LATER")
            logging.info("Starting a dual problem.")
            s_i, p_j, R_j_i, C_j_i, N_i = solve_factory_optimization_problem(self.compiled, u_j, p0_j, priced_indices, known_technologies, ginv_j)
            c_i = C_j_i.T @ p0_j        
            assert linear_transform_is_gt(-1 * (R_j_i / c_i[None, :]).T, p_j, -1 * np.ones_like(c_i)).all(), [self.reference_list[i] for i in np.where((R_j_i / c_i[None, :]).T @ p_j > 0)[0]] # type: ignore
        logging.info("Reconstructing factory.")

        s_i[np.where(s_i < 0)] = 0 #0 is theoretically impossible, so we remove them to remove issues

        scale = 100 / np.max(p_j) #normalization to prevent massive numbers.
        p = CompressedVector({k: p_j[self.reference_list.index(k)] * scale for k in targets.keys()})
        p_full = CompressedVector({k: p_j[i] * scale for i, k in enumerate(self.reference_list)}) #normalization to prevent massive numbers.
        
        s = CompressedVector({})
        positives = np.logical_not(np.isclose(s_i, 0, rtol=SOLVER_TOLERANCES['rtol'], atol=SOLVER_TOLERANCES['atol']))
        for i in range(s_i.shape[0]):
            if positives[i]:
                s = s + s_i[i] * N_i[i]
        valid_rows = np.asarray((R_j_i > 0).sum(axis=1)).flatten() > 0
        k: CompressedVector = efficiency_analysis(self.compiled, p0_j, priced_indices, p_j, known_technologies, valid_rows, spatial_mode)

        fc_i = C_j_i @ s_i
        assert np.logical_or(fc_i >= 0, np.isclose(fc_i, 0, rtol=SOLVER_TOLERANCES['rtol'], atol=SOLVER_TOLERANCES['atol'])).all(), fc_i
        full_material_cost = CompressedVector({self.reference_list[i]: fc_i[i] for i in range(fc_i.shape[0]) if fc_i[i]>0})

        zs = find_zeros(R_j_i, s_i)

        return s, p, p_full, k, zs, scale, full_material_cost
    
    def solve_looped_pricing_model(self, starter_base: CompressedVector, known_technologies: TechnologicalLimitation) -> tuple[np.ndarray, sp.sparse.sparray]:
        """
        Currently unavailable as looped pricing models are nonlinear.
        Translate a starter base of just strings with values into something that makes a little more sense to the 
        program using Levenshtein distance. https://en.wikipedia.org/wiki/Levenshtein_distance
        Then calculates the pricing model from this factory.

        Parameters
        ----------
        starter_base:
            CompressedVector that describes what the factory should be.
        known_technologies:
            TechnologicalLimitation representing what technologies are done.
        
        Returns
        -------
        u_j:
            Output of the given factory.
        p_j:
            Pricing model of given factory.
        """
        raise NotImplementedError
        constructs = []
        for f in self.uncompiled_constructs:
            logging.debug("Working on construct: "+f.ident)
            for c in f.get_constructs(self.reference_list, self.catalyst_list, known_technologies, self.MODULE_REFERENCE):
                constructs.append(c)

        translated_base = CompressedVector()
        for k, v in starter_base.items():
            best_matches = []
            match_distance = Levenshtein.distance(k, constructs[0].ident)
            logging.debug("Atempting to translate: "+"\""+k+"\" starting distance is: "+str(match_distance)+" there "+("is" if k in [c.ident for c in constructs] else "is not")+" a 0 length translation.")
            for c in constructs:
                dist = Levenshtein.distance(k, c.ident)
                if dist < match_distance:
                    best_matches = [c.ident]
                    match_distance = dist
                    logging.debug("\tFound a new best match in: \""+c.ident+"\" at distance: "+str(match_distance))
                elif dist == match_distance:
                    best_matches.append(c.ident)
                    logging.debug("\t\tAdded the new possible match: \""+c.ident+"\"")
                elif c.ident == k:
                    logging.debug("Found the exact match but chose to ignore it because im a dumb program"+str(dist))
                    raise ValueError
            assert len(best_matches)==1, "Unable to determine which construct an input phrase is associated with.\nPhrase is: "+k+"\nPossible constructs were:\n\t"+"\n\t".join([m for m in best_matches])+"\n\tWith distance: "+str(match_distance)
            translated_base[best_matches[0]] = v
            logging.debug("Translated: \""+k+"\" to mean the construct: \""+best_matches[0]+"\"")
        
        n = len(self.reference_list)

        R_j_i = sp.hstack([construct.vector for construct in constructs])
        m = len(constructs)
        logging.debug("n="+str(n))
        logging.debug("m="+str(m))
        logging.debug("R_i_j shape: "+str(R_j_i.shape))

        s_i = np.zeros(m)
        for k, v in translated_base.items():
            s_i[constructs.index(next(c for c in constructs if c.ident==k))] = v
        logging.debug("s_i shape: "+str(s_i.shape))

        u_j = R_j_i @ s_i
        logging.debug("u_j shape: "+str(u_j.shape))
        assert (u_j>=0).all(), "Found negative values in looped factory. "+str([self.reference_list[i] for i in np.where(u_j<0)[0]])+" w/ "+str(u_j[u_j<0])

        C_j_i = sp.hstack([construct.cost for construct in constructs])
        logging.debug("C_i_j shape: "+str(C_j_i.shape))

        reference_index = np.argmax(u_j)
        
        p_j = calculate_pricing_model_via_prebuilt(R_j_i, C_j_i, s_i, u_j, reference_index)
        
        return u_j, p_j

    def technological_limitation_from_specification(self, fully_automated: list[str] = [], extra_technologies: list[str] = [], extra_recipes: list[str] = []) -> TechnologicalLimitation:
        """
        Wrapper for technological_limitation_from_specification that passes it the values it needs.
        Generates a TechnologicalLimitation from a specification. Works as a more user friendly way of getting useful TechnologicalLimitations.

        Parameters
        ----------
        data:
            Entire data.raw. https://wiki.factorio.com/Data.raw
        COST_MODE:
            What cost mode is being used. https://lua-api.factorio.com/latest/concepts.html#DifficultySettings
        fully_automated:
            List of fully automated science packs.
        extra_technologies:
            List of additional unlocked technologies.
        extra_recipes:
            List of additionally unlocked recipes.
        
        Returns
        -------
        Specified TechnologicalLimitations
        """
        return technological_limitation_from_specification(self, self.COST_MODE, fully_automated=fully_automated, extra_technologies=extra_technologies, extra_recipes=extra_recipes)

def efficiency_analysis(construct: ComplexConstruct, pricing_vector: np.ndarray, priced_indices: np.ndarray, dual_vector: np.ndarray, 
                        known_technologies: TechnologicalLimitation, valid_rows: np.ndarray, spatial_mode: bool) -> CompressedVector:
        efficiencies: CompressedVector = CompressedVector({})
        for sc in construct.subconstructs:
            if isinstance(sc, ComplexConstruct):
                efficiencies.update({sc.ident: sc.efficiency_analysis(pricing_vector, priced_indices, dual_vector, known_technologies, valid_rows, spatial_mode)})
                efficiencies = efficiencies + efficiency_analysis(sc, pricing_vector, priced_indices, dual_vector, known_technologies, valid_rows, spatial_mode)
        return efficiencies

def index_compiled_constructs(constructs: tuple[LinearConstruct, ...], ident: str) -> LinearConstruct:
    """
    Finds the LinearConstruct referenced by an ident.

    Parameters
    ----------
    constructs:
        List of LinearConstructs to search.
    ident:
        Identification string to look for.

    Returns
    -------
    LinearConstruct that's ident matches ident.
    """
    for construct in constructs:
        if construct.ident==ident:
            return construct
    raise ValueError(ident)


class FactorioFactory():
    """
    Abstract Class. A factory in factorio.

    Members
    -------
    instance:
        FactorioInstance associated with this factory.
    known_technologies:
        TechnologicalLimitation describing what recipes and buildings are avaiable.
    targets:
        CompressedVector of the target outputs.
    optimal_factory:
        Calculated optimal factory from last calculate_optimal_factory run.
    optimal_pricing_model:
        Calculated pricing model of the optimal_factory.
    inefficient_constructs:
        Calculated inefficencies of unused constructs.
    """
    instance: FactorioInstance
    known_technologies: TechnologicalLimitation
    targets: CompressedVector
    last_pricing_model: CompressedVector
    optimal_factory: CompressedVector
    optimal_pricing_model: CompressedVector
    full_optimal_pricing_model: CompressedVector
    inefficient_constructs: CompressedVector
    zero_throughputs: list[int]
    transport_densities: list[tuple[str, str, Fraction | float]]
    optimized: bool
    intermediate_scale_factor: float
    true_cost: CompressedVector

    def __init__(self, instance: FactorioInstance, known_technologies: TechnologicalLimitation, targets: CompressedVector) -> None:
        """
        Parameters
        ----------
        instance:
            Associated FactorioInstance.
        known_technologies:
            TechnologicalLimitation of what is avaiable for the factory.
        targets:
            Outputs of the factory.
        """
        assert isinstance(instance, FactorioInstance)
        assert isinstance(known_technologies, TechnologicalLimitation)
        assert isinstance(targets, CompressedVector)
        self.instance = instance
        self.known_technologies = known_technologies
        self.targets = targets
        self.optimal_factory = None # type: ignore
        self.optimal_pricing_model = CompressedVector()
        self.full_optimal_pricing_model = CompressedVector()
        self.last_pricing_model = CompressedVector()
        self.optimized = False
        self.intermediate_scale_factor = 0
        self.true_cost = CompressedVector()

    def calculate_optimal_factory(self, reference_model: CompressedVector, ore_area_optimized: bool = False) -> bool:
        """
        Calculates a optimal factory with the now avaiable reference model.

        Parameters
        ----------
        reference_model:
            CompressedVector of reference pricing model.

        Returns
        -------
        If pricing model has changed.
        """
        s, p, pf, k, zs, scale, tc = self.instance.solve_for_target(self.targets, self.known_technologies, reference_model, self.optimal_factory, self.optimal_pricing_model, spatial_mode=ore_area_optimized)
        self.last_pricing_model = reference_model
        same = p==self.optimal_pricing_model
        self.optimal_factory, self.optimal_pricing_model, self.full_optimal_pricing_model, self.inefficient_constructs, self.intermediate_scale_factor, self.true_cost = s, p, pf, k, scale, tc
        self.zero_throughputs = zs
        self.optimized = True

        return not same

    def retarget(self, targets: CompressedVector, retainment: float = RETAINMENT_VALUE) -> None:
        """
        Rebuilds targets. This is useful if iteratively optimizing a built chain.
        After this one should re-run calculate_optimal_factory and if it returns False then the pricing model didn't change even after retargeting.
        
        Parameters
        ----------
        targets:
            CompressedVector of the new target outputs.
        retainment:
            How much building of targets that used to exist should be retained. 
            With 0 retainment this factory may start mispricing those targets.
        """
        self_copy = copy.deepcopy(self.targets)
        self.targets = targets
        for k in self_copy.keys():
            if not k in self.targets.keys():
                self.targets[k] = retainment

        self.optimized = False

    def run_post_analysis(self) -> None:
        """
        Computes all the global post run analyses into post_analyses.
        """
        
        """for construct_name, targets in POST_ANALYSES:
            construct = self.instance.search_complex_constructs(construct_name)

            initial_pricing_vector = np.zeros(len(self.instance.reference_list))
            for k, v in self.last_pricing_model.items():
                initial_pricing_vector[self.instance.reference_list.index(k)] = v

            output_pricing_vector = np.zeros(len(self.instance.reference_list))
            for k, v in self.full_optimal_pricing_model.items():
                output_pricing_vector[self.instance.reference_list.index(k)] = v"""

        self.transport_densities = compute_transportation_densities(self.instance.data_raw, self.full_optimal_pricing_model)
    
    def dump_to_excel(self, writer: pd.ExcelWriter, sheet_name: str) -> None:
        """
        Dumps the target, optimal factory, optimal pricing model, and inefficient constructs into an excel spreadsheet sheet.

        Parameters
        ----------
        writer:
            Excel writer to use.
        sheet_name:
            What sheet to write to.
        """
        self.run_post_analysis()
        try:
            targets_df = pd.DataFrame(list(self.targets.items()), columns=['target', 'count'])
            optimal_factory_df = pd.DataFrame(list(self.optimal_factory.items()), columns=['construct', 'count'])
            optimal_pricing_model_df = pd.DataFrame(list(self.optimal_pricing_model.items()), columns=['item', 'value'])
            inefficient_constructs_df = pd.DataFrame(list(self.inefficient_constructs.items()), columns=['construct', 'relative value'])
            transport_df = pd.DataFrame(list(self.transport_densities), columns=['item', 'via', 'relative density'])
            merged_df = pd.concat([targets_df, pd.DataFrame({}, columns=['']), 
                                optimal_factory_df, pd.DataFrame({}, columns=['']),
                                optimal_pricing_model_df, pd.DataFrame({}, columns=['']),
                                inefficient_constructs_df, pd.DataFrame({}, columns=['']),
                                transport_df], axis=1)
            merged_df.to_excel(writer, sheet_name=sheet_name, index=False)
        except:
            if SUPRESS_EXCEL_ERRORS:
                logging.error("Was unable to dump factory to excel.")
                return
            else:
                raise RuntimeError("Unsuppressed Excel Error.")


class FactorioMaterialFactory(FactorioFactory):
    """
    A factory in factorio that makes materials (parts for future factories).

    Added Members
    -------------
    last_material_factory
        Factory to base the pricing model of this factory on
    """
    last_material_factory: FactorioMaterialFactory | InitialFactory

    def __init__(self, instance: FactorioInstance, previous_science: FactorioScienceFactory | InitialFactory | TechnologicalLimitation, 
                 last_material_factory: FactorioMaterialFactory | InitialFactory, material_targets: CompressedVector) -> None:
        """
        Parameters
        ----------
        instance:
            Associated FactorioInstance.
        previous_science:
            Factory to base current science coverage on or tech limit of current science coverage.
        last_material_factory
            Factory to base the pricing model of this factory on
        material_targets:
            Outputs of the factory, cannot contain research.
        """
        assert isinstance(instance, FactorioInstance)
        assert isinstance(previous_science, FactorioScienceFactory) or isinstance(previous_science, InitialFactory) or isinstance(previous_science, TechnologicalLimitation)
        assert isinstance(last_material_factory, FactorioMaterialFactory) or isinstance(last_material_factory, InitialFactory)
        assert isinstance(material_targets, CompressedVector)
        if isinstance(previous_science, FactorioScienceFactory) or isinstance(previous_science, InitialFactory):
            super().__init__(instance, previous_science.get_technological_coverage(), material_targets)
        else: #we probably got here via self.startup_base
            super().__init__(instance, previous_science, material_targets)
        self.last_material_factory = last_material_factory
    
    """
    @classmethod
    def startup_base(cls, instance: FactorioInstance, base_building_setup: CompressedVector, 
                     starting_techs: TechnologicalLimitation) -> FactorioMaterialFactory:

        Alternative initialization from a prebuilt base.

        Parameters
        ----------
        instance:
            FactorioInstance associated with this factory.
        base_building_setup:
            CompressedVector that describes what the factory should be.
        starting_techs:
            Starting tech limitations if needed.

        raise NotImplementedError("Current interface doesn't support prebuilt bases.")
        assert isinstance(instance, FactorioInstance)
        assert isinstance(base_building_setup, CompressedVector)
        assert isinstance(starting_techs, TechnologicalLimitation)
        inst = cls(instance, starting_techs, None, None)
        inst.previous_science = inst
        inst.optimal_pricing_model = inst.calculate_optimal_factory(base_building_setup, True)

        return inst
    """


class FactorioScienceFactory(FactorioFactory):
    """
    A factory in factorio that completes research.

    Added Members
    -------------
    last_material_factory:
        Factory to base the pricing model of this factory on.
    previous_science:
        Last science factory or tech limit.
    time_target:
        How long it should take to complete all research.
    clear:
        Set of tools that define which must be produced to clear all researches possible with just those tools.
        Example: If this was a set of red and green science pack that would indicate all research that only red and
                 green science packs are needed for MUST be completed within this factory.
    """
    last_material_factory: FactorioMaterialFactory
    previous_science: FactorioScienceFactory | InitialFactory | TechnologicalLimitation
    time_target: Fraction
    clear: list[str]

    def __init__(self, instance: FactorioInstance, previous_science: FactorioScienceFactory | InitialFactory | TechnologicalLimitation, last_material_factory: FactorioMaterialFactory | InitialFactory, 
                 science_targets: CompressedVector, time_target: Fraction | None = None) -> None:
        assert isinstance(instance, FactorioInstance)
        assert isinstance(previous_science, FactorioScienceFactory) or isinstance(previous_science, InitialFactory) or isinstance(previous_science, TechnologicalLimitation)
        assert isinstance(last_material_factory, FactorioMaterialFactory) or isinstance(last_material_factory, InitialFactory)
        assert isinstance(science_targets, CompressedVector)
        assert isinstance(time_target, Fraction) or time_target is None
        self.instance = instance
        self.clear = [target for target in science_targets.keys() if target in self.instance.data_raw['tool'].keys()]
        self.previous_science = previous_science
        covering_to = technological_limitation_from_specification(self.instance, self.instance.COST_MODE, fully_automated=self.clear) + \
                      TechnologicalLimitation(instance.tech_tree, [set([target for target in science_targets.keys() if target in self.instance.data_raw['technology'].keys()])])
        
        last_coverage = self._previous_coverage()

        if time_target is None:
            self.time_target = self.instance.DEFAULT_TARGET_RESEARCH_TIME
        else:
            self.time_target = time_target

        targets = CompressedVector({instance.tech_tree.inverse_map[k]+RESEARCH_SPECIAL_STRING: 1 / self.time_target for k in next(iter(covering_to.canonical_form)) if k not in next(iter(last_coverage.canonical_form))}) #next(iter()) gives us the first (and theoretically only) set of nodes making up the tech limit

        super().__init__(instance, last_coverage, targets)
        #super().__init__(instance, last_coverage, science_targets)

        self.last_material_factory = last_material_factory
    
    def _previous_coverage(self) -> TechnologicalLimitation:
        if isinstance(self.previous_science, TechnologicalLimitation):
            return self.previous_science
        else:
            return self.previous_science.get_technological_coverage()

    def get_technological_coverage(self) -> TechnologicalLimitation:
        """
        Determine what technologies will be unlocked when this factory is done.
        """
        return self._previous_coverage() + TechnologicalLimitation(self.instance.tech_tree, [set([targ[:targ.rfind("=")] for targ in self.targets.keys()])])

    def retarget(self, targets: CompressedVector, retainment: float = RETAINMENT_VALUE) -> None:
        """
        Rebuilds targets. This is useful if iteratively optimizing a built chain.
        After this one should re-run calculate_optimal_factory and if it returns False then the pricing model didn't change even after retargeting.
        Will make sure all technologies cleared by self.clear set are still in targets.
        
        Parameters
        ----------
        science_targets:
            CompressedVector of the new science target outputs.
        retainment:
            How much building of targets that used to exist should be retained. 
            With 0 retainment this factory may start mispricing those targets.
        """
        #assert not any([target in self.instance.data_raw['tool'].keys() for target in targets.keys()]), "retarget should NEVER be given a tool. Only researches."
        assert all([t in self.targets.keys() for t in targets]), "retarget should never add new targets... yet."
        covering_to: TechnologicalLimitation = technological_limitation_from_specification(self.instance, self.instance.COST_MODE, fully_automated=self.clear) + \
                      TechnologicalLimitation(self.instance.tech_tree, [set([target for target in targets.keys()])])
        last_coverage: TechnologicalLimitation = self._previous_coverage()
        
        targets = CompressedVector({k: 1 / self.time_target for k in next(iter(covering_to.canonical_form)) if k not in next(iter(last_coverage))}) # type: ignore

        super().retarget(targets, retainment=retainment)


class InitialFactory(FactorioMaterialFactory, FactorioScienceFactory):
    """
    A fake factory instance to hold a pricing model and tech level.
    """

    def __init__(self, instance: FactorioInstance, pricing_model: CompressedVector, known_technologies: TechnologicalLimitation) -> None:
        """
        Parameters
        ----------
        instance:
            Associated FactorioInstance.
        pricing_model:
            Baseline pricing model.
        known_technologies:
            TechnologicalLimitation baseline, usually enough to begin full automatization.
        """
        assert isinstance(instance, FactorioInstance)
        assert isinstance(pricing_model, CompressedVector)
        assert isinstance(known_technologies, TechnologicalLimitation)
        self.instance = instance
        self.known_technologies = known_technologies
        self.targets = CompressedVector()
        self.optimal_factory = CompressedVector()
        self.optimal_pricing_model = pricing_model
        self.inefficient_constructs = CompressedVector()
    
    def calculate_optimal_factory(self, reference_model: CompressedVector, ore_area_optimized: bool = False) -> bool:
        """
        Placeholder. Initial Factories cannot change.
        """
        return False
    
    def get_technological_coverage(self) -> TechnologicalLimitation:
        """
        Get the tech level unlocked by default.
        """
        return self.known_technologies
    
    def retarget(self, targets: CompressedVector, retainment: float = RETAINMENT_VALUE) -> None:
        """
        Placeholder. Initial Factories cannot change.
        """
        return None
    

class FactorioFactoryChain():
    """
    A chain of optimal factory designs starting from the minimal science to a specified point. There are two avaiable types of factory in the chain:
        A science factory that produces a set of science packs. Having one of these will permit any recipe that is unlockable with those sciences to be used in factories after it.
        A material factory that produces a set of buildings and catalyst materials. Having one of these will set the pricing model for every factory in the chain up until the next component factory.
    
    Members
    -------
    instance:
        FactorioInstance for this chain.
    chain:
        List containing the chain.
    ore_area_optimized:
        Should factories in this chain be forced to minimize ore area before optimizing cost?
        This in general will FORCE the program to use the most 'productive' choice, so it will stack max level productivities where possible and usually speeds in mining drills.
    """
    instance: FactorioInstance
    chain: list[FactorioFactory]
    ore_area_optimized: bool

    def __init__(self, instance: FactorioInstance, ore_area_optimized: bool = False) -> None:
        assert isinstance(instance, FactorioInstance)
        self.instance = instance
        self.chain = []
        self.ore_area_optimized = ore_area_optimized

    def startup(self, base_building_setup: CompressedVector, starting_techs: TechnologicalLimitation) -> None:
        """
        Initialize the first factory of the chain from a prebuilt design.

        Parameters
        ----------
        base_building_setup:
            CompressedVector that describes what the starting factory should be.
        starting_techs:
            TechnologicalLimitation of techs unlocked before building a starting factory.
        """
        raise NotImplementedError("Current implementation doesn't support startup bases.")
        self.chain.append(FactorioMaterialFactory.startup_base(self.instance, base_building_setup, starting_techs))
    
    def initial_pricing(self, pricing_model: CompressedVector, starting_techs: TechnologicalLimitation) -> None:
        """
        Initialize the first factory of the chain with a pricing model and techs.

        Parameters
        ----------
        pricing_model:
            CompressedVector that describes what the pricing model should be.
        starting_techs:
            TechnologicalLimitation of techs unlocked before building a starting factory.
        """
        self.chain.append(InitialFactory(self.instance, pricing_model, starting_techs))
    
    def add(self, targets: CompressedVector | str) -> None:
        """
        Adds a factory to the chain.

        Parameters
        ----------
        targets:
            Either:
                CompressedVector of target outputs for the factory. Must be either all science tools or building materials
            Or:
                String:
                    "all materials" = will autopopulate all possible buildings and materials that can be made (TODO: restrict to catalysts)
                    "all tech" = will populate all possible tech that can be made
        """
        factory_type = ""
        previous_sciences = [fac for fac in self.chain if isinstance(fac, FactorioScienceFactory)]
        last_material = [fac for fac in self.chain if isinstance(fac, FactorioMaterialFactory)][-1]
        if len(previous_sciences)==0:
            known_technologies = last_material.known_technologies #first science after a startup base.
        else:
            known_technologies = previous_sciences[-1].get_technological_coverage()

        if isinstance(targets, CompressedVector):
            if list(targets.keys())[0] in self.instance.data_raw['tool'].keys() or list(targets.keys())[0] in self.instance.data_raw['technology'].keys():
                factory_type = "science"
            elif list(targets.keys())[0] in self.instance.data_raw['item'].keys() or list(targets.keys())[0] in self.instance.data_raw['fluid'].keys():
                factory_type = "material"
        else:
            last_pricing_model = last_material.optimal_pricing_model 
            if DEBUG_BLOCK_MODULES:
                last_pricing_model = CompressedVector({k: v for k, v in last_pricing_model.items() if not '-module' in k})

            p0_j = np.zeros(len(self.instance.reference_list), dtype=np.longdouble)
            for k, v in last_pricing_model.items():
                p0_j[self.instance.reference_list.index(k)] = v
            
            self.instance.compile()
            assert not self.instance.compiled is None
            R_j_i, C_j_i, N_i = self.instance.compiled.reduce(p0_j, 
                                                              np.array([self.instance.reference_list.index(k) for k in last_pricing_model.keys()]), 
                                                              None, known_technologies)
            R_j_i = sparse.coo_matrix(R_j_i)

            targets_dict = CompressedVector()
            
            if targets=="all tech":
                factory_type = "science"
                for tool in self.instance.data_raw['tool'].keys():
                    if self.instance.reference_list.index(tool) in R_j_i.row:
                        targets_dict[tool] = Fraction(1)

            elif targets=="all materials":
                factory_type = "material"
                #only actually craftable materials
                for item_cata in ITEM_SUB_PROTOTYPES:
                    if item_cata=='tool':
                        continue #skip tools in material factories
                    for item in self.instance.data_raw[item_cata].keys():
                        if not item in self.instance.reference_list:
                            if not item in WARNING_LIST:
                                logging.warning("Detected some a weird "+item_cata+": "+item)
                                WARNING_LIST.append(item)
                        elif self.instance.reference_list.index(item) in R_j_i.row:
                            targets_dict[item] = Fraction(1)
                for fluid in self.instance.data_raw['fluid'].keys():
                    if fluid in self.instance.RELEVENT_FLUID_TEMPERATURES.keys():
                        for temp in self.instance.RELEVENT_FLUID_TEMPERATURES[fluid].keys():
                            if not fluid+'@'+str(temp) in self.instance.reference_list:
                                raise ValueError("Fluid \""+fluid+"\" found to have temperature "+str(temp)+" but said temperature wasn't found in the reference list.")
                            if self.instance.reference_list.index(fluid+'@'+str(temp)) in R_j_i.row:
                                targets_dict[fluid+'@'+str(temp)] = Fraction(1)
                    else:
                        if not fluid in self.instance.reference_list:
                            if not fluid in WARNING_LIST:
                                logging.warning("Detected some a weird fluid: "+fluid)
                                WARNING_LIST.append(fluid)
                        elif self.instance.reference_list.index(fluid) in R_j_i.row:
                            targets_dict[fluid] = Fraction(1)
                for other in ['electric', 'heat']:
                    if not other in self.instance.reference_list:
                        if not other in WARNING_LIST:
                            logging.warning("Was unable to find "+other+" in the reference list. While not nessisary wrong this is extreamly odd and should only happen on very strange mod setups.")
                            WARNING_LIST.append(other)
                    if self.instance.reference_list.index(other) in R_j_i.row:
                        targets_dict[other] = Fraction(1)
            logging.info("Auto-targets:\n\t"+str(list(targets_dict.keys())))
            targets = targets_dict

        if factory_type == "science":
            for target in targets.keys():
                assert target in self.instance.data_raw['tool'].keys() or target in self.instance.data_raw['technology'].keys(), "If a factory does science stuff is only allowed to do science stuff."
            previous_science = previous_sciences[-1] if len(previous_sciences)>0 else known_technologies
            self.chain.append(FactorioScienceFactory(self.instance, previous_science, last_material, targets))
        else: #factory_type == "material":
            logging.info("Attempting to add the following targets to the new material factory:\n\t"+str(set(targets.keys()).difference(set(last_material.targets.keys()))))
            last_science = previous_sciences[-1]
            self.chain.append(FactorioMaterialFactory(self.instance, last_science, last_material, targets))
        
        self.chain[-1].calculate_optimal_factory(last_material.optimal_pricing_model, ore_area_optimized=self.ore_area_optimized)
    
    def retarget_all(self) -> None:
        """
        Retargets all factories in the chain.
        """
        last_material = None
        for i in range(len(self.chain)):
            factory = self.chain[i]
            if isinstance(factory, InitialFactory):
                last_material = factory
                pass #cannot be updated
            elif isinstance(factory, FactorioMaterialFactory):
                updated_targets = CompressedVector()
                for j in range(i+1, len(self.chain)):
                    updated_targets = updated_targets + self.chain[j].true_cost
                    if isinstance(self.chain[j], FactorioMaterialFactory):
                        break #done with this material factories outputs
                logging.info(updated_targets)
                factory.retarget(updated_targets)
                assert not last_material is None
                logging.info(last_material.optimal_pricing_model)
                factory.calculate_optimal_factory(last_material.optimal_pricing_model, ore_area_optimized=self.ore_area_optimized)
                last_material = factory
            elif isinstance(factory, FactorioScienceFactory):
                assert not last_material is None
                factory.calculate_optimal_factory(last_material.optimal_pricing_model, ore_area_optimized=self.ore_area_optimized)
                pass #we don't update science factories at the moment

    def compute_all(self) -> bool:
        """
        Computes all pricing models for chain iteratively and returns if the pricing models changed.

        Returns
        -------
        True if pricing model has been changed in any factory.
        """
        self.retarget_all()
        changed = False

        last_reference_model = self.chain[0].optimal_pricing_model
        for i, factory in enumerate(self.chain[1:]):
            changed: bool = factory.calculate_optimal_factory(last_reference_model, ore_area_optimized=self.ore_area_optimized) or changed
            if isinstance(factory, FactorioMaterialFactory):
                updated_targets = CompressedVector()
                for j in range(i+1, len(self.chain)):
                    updated_targets = updated_targets + self.chain[j].true_cost
                    if isinstance(self.chain[j], FactorioMaterialFactory):
                        break #done with this material factories outputs
                logging.info(updated_targets)
                factory.retarget(updated_targets)
                logging.info(last_reference_model)
                factory_old_pm = factory.optimal_pricing_model
                try:
                    changed = factory.calculate_optimal_factory(last_reference_model, ore_area_optimized=self.ore_area_optimized) or changed
                except:
                    logging.info(i)
                    logging.info(last_reference_model)
                    logging.info(updated_targets)
                    logging.info(factory_old_pm)
                    raise RuntimeError("?")
                last_reference_model = factory.optimal_pricing_model
                for k in factory_old_pm.keys():
                    assert k in last_reference_model.keys()
            elif isinstance(factory, FactorioScienceFactory):
                changed = factory.calculate_optimal_factory(last_reference_model, ore_area_optimized=self.ore_area_optimized) or changed
                pass #we don't update science factories at the moment
        
        return changed
            
    def dump_to_excel(self, file_name: str) -> None:
        """
        Dumps the entire chain into an excel spreadsheet.

        Parameters
        ----------
        file_name:
            Name of excel file to write to.
        """
        writer = pd.ExcelWriter(file_name)
        material_factory_ident = 1
        science_factory_ident = 1
        for factory in self.chain[1:]:
            if isinstance(factory, FactorioMaterialFactory):
                if not factory.optimized:
                    logging.error("Found Material Factory "+str(material_factory_ident))
                else:
                    factory.dump_to_excel(writer, "Material Factory "+str(material_factory_ident))
                material_factory_ident += 1
            elif isinstance(factory, FactorioScienceFactory):
                if not factory.optimized:
                    logging.error("Found Science Factory "+str(science_factory_ident))
                else:
                    factory.dump_to_excel(writer, "Science Factory "+str(science_factory_ident))
                science_factory_ident += 1
            #else: #if isinstance(factory, InitialFactory):
        writer.close()