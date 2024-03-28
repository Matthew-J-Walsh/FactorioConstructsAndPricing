from __future__ import annotations
from globalsandimports import *

from utils import *
from generators import *
#from linearconstructs import *
from constructs import *
from datarawparse import *
from lpproblems import *


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
    uncompiled_constructs: list[UncompiledConstruct]
    complex_constructs: list[SingularConstruct]
    disabled_constructs: list[str]
    compiled: ComplexConstruct | None
    reference_list: list[str]
    catalyst_list: list[str]
    COST_MODE: str
    DEFAULT_TARGET_RESEARCH_TIME: Fraction
    RELEVENT_FLUID_TEMPERATURES: dict

    def __init__(self, filename: str, COST_MODE: str = 'normal', DEFAULT_TARGET_RESEARCH_TIME: Real = 10000) -> None:
        """
        Parameters
        ----------
        filename:
            Filename of data.raw to load for this instance.
        COST_MODE:
            What cost mode should be used. https://lua-api.factorio.com/latest/concepts.html#DifficultySettings
        DEFAULT_TARGET_RESEARCH_TIME:
            Default time that should science factories should complete their science in.
        """
        with open(filename) as f:
            self.data_raw = json.load(f)
        
        self.COST_MODE = COST_MODE
        self.DEFAULT_TARGET_RESEARCH_TIME = DEFAULT_TARGET_RESEARCH_TIME
        self.RELEVENT_FLUID_TEMPERATURES = {}

        complete_premanagement(self.data_raw, self.RELEVENT_FLUID_TEMPERATURES, self.COST_MODE)
        logging.info("Building uncompiled constructs.")
        self.uncompiled_constructs = generate_all_constructs(self.data_raw, self.RELEVENT_FLUID_TEMPERATURES, self.COST_MODE)
        logging.info("Building reference and catalyst lists.")
        self.reference_list, self.catalyst_list = generate_references_and_catalysts(self.uncompiled_constructs)
        for k in self.reference_list:
            DEBUG_REFERENCE_LIST.append(k)
        logging.info("Building complex constructs.")
        self.complex_constructs = [uc.compile(self.catalyst_list, self.data_raw['module'], self.reference_list, list(self.data_raw['beacon'].values())) for uc in self.uncompiled_constructs]
        self.compiled = None
        self.compile()
    
    def compile(self) -> ComplexConstruct:
        """
        Populates compiled and returns it.
        """
        if self.compiled is None:
            self.compiled = ComplexConstruct(self.complex_constructs, "Whole Game Construct")
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
        ident = self.search_complex_constructs(target_name)
        try:
            self.disabled_constructs.remove(ident)
            self.compiled = None
        except:
            logging.warning(ident+" was not disabled in the first place.")

    def search_complex_constructs(self, target_name: str) -> str:
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
                best_matches = [c.ident]
                match_distance = dist
                logging.debug("\tFound a new best match in: \""+c.ident+"\" at distance: "+str(match_distance))
            elif dist == match_distance:
                best_matches.append(c.ident)
                logging.debug("\t\tAdded the new possible match: \""+c.ident+"\"")
            elif c.ident == target_name:
                logging.debug("Found the exact match but chose to ignore it because im a dumb program "+str(dist))
                raise ValueError
        assert len(best_matches)==1, "Unable to determine which construct an input phrase is associated with.\nPhrase is: "+target_name+"\nPossible constructs were:\n\t"+"\n\t".join([m for m in best_matches])+"\n\tWith distance: "+str(match_distance)
        logging.debug("Translated: \""+target_name+"\" to mean the construct: \""+best_matches[0]+"\"")
        return best_matches[0]

    def solve_for_target(self, targets: CompressedVector, known_technologies: TechnologicalLimitation, reference_model: CompressedVector) -> tuple[CompressedVector, CompressedVector, CompressedVector]:
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
        #reference_model = CompressedVector({k:v for k, v in reference_model.items() if not '-module' in k}) #module stripping
        #construct_transform = ConstructTransform(self.compiled_constructs, self.reference_list).validate(known_technologies, reference_model)
        #assert not any(['with module setup' in construct.ident for construct in constructs]) #module stripping check
        self.compile()

        p0_j = np.zeros(n, dtype=np.longdouble)
        for k, v in reference_model.items():
            p0_j[self.reference_list.index(k)] = v

        R_j_i, c_i, N1, N0, FactoryRecovery = self.compiled.reduce(p0_j, [self.reference_list.index(k) for k in reference_model.keys()], known_technologies)

        assert (c_i!=0).all(), "Some suspiciously priced things. In general nothing creatable should have no price."
        assert (c_i < 1e50).all(), "Some error has occured where an unpricable construct has been allowed through"
        assert N1.shape[0]==0, "Stabilization unsupported "+str(N1.shape)
        assert N0.shape[0]==0, "Stabilization unsupported "+str(N0.shape)

        m = R_j_i.shape[1]
        logging.info("Construct count: "+str(m))

        u_j = np.zeros(n, dtype=np.longdouble)
        for k, v in targets.items():
            u_j[self.reference_list.index(k)] = v

        
        #s_i = solve_factory_optimization_problem(R_j_i, u_j, c_i)

        #try:
        #    assert linear_transform_is_gt(R_j_i.astype(np.longdouble), s_i.astype(np.longdouble), u_j.astype(np.longdouble)).all(), "Somehow solution infeasible? This should never happen but we check anyway."
        #except:
        #    nonzero = np.nonzero(1-linear_transform_is_gt(R_j_i.astype(np.longdouble), s_i.astype(np.longdouble), u_j.astype(np.longdouble)))[0]
        #    for i in nonzero:
        #        print("assertion failure on: "+self.reference_list[i]+" values are "+str((R_j_i @ s_i)[i])+" vs "+str(u_j[i]))
        #    raise AssertionError

        #logging.info("Starting solve for pricing model.")
        #p_j = solve_pricing_model_calculation_problem(R_j_i, s_i, u_j, c_i)

        logging.info("Starting a dual problem.")
        #s_i, p_j = solve_factory_optimization_problem_dual(R_j_i, u_j, c_i)
        s_i, p_j = solve_factory_optimization_problem_dual_iteratively(R_j_i, u_j, c_i, p0_j)
        #assert s_i.shape[0]==primal.shape[0]
        #assert p_j.shape[0]==dual.shape[0]
        #assert np.isclose(s_i, primal).all(), np.max(np.abs(s_i - primal))
        #targeting_mask = np.array([name in targets.keys() for name in self.reference_list])
        #assert targeting_mask.shape[0]==p_j.shape[0]
        #try:
        #    assert np.isclose(p_j[np.where(targeting_mask)], dual[np.where(targeting_mask)]).all()
        #except:
        #    tempers = np.argmax(np.abs(p_j[np.where(targeting_mask)] - dual[np.where(targeting_mask)]))
        #    print(str(p_j[np.where(targeting_mask)][tempers])+" "+str(dual[np.where(targeting_mask)][tempers]))
        #    dumpers = np.vstack([p_j[np.where(targeting_mask)], dual[np.where(targeting_mask)]]).T
        #    print(dumpers)
        #    raise AssertionError()
        
        p = CompressedVector({k: p_j[self.reference_list.index(k)] / np.max(p_j) * 100 for k in targets.keys()}) #normalization to prevent massive numbers.
        logging.info("Reconstructing factory.")
        s, k = FactoryRecovery(s_i, p_j)

        return s, p, k
    
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
        return technological_limitation_from_specification(self.data_raw, self.COST_MODE, fully_automated=fully_automated, extra_technologies=extra_technologies, extra_recipes=extra_recipes)


def index_compiled_constructs(constructs: list[LinearConstruct], ident: str) -> LinearConstruct:
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
    optimal_factory: CompressedVector
    optimal_pricing_model: CompressedVector
    inefficient_constructs: CompressedVector

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
        self.optimal_pricing_model = CompressedVector()

    def calculate_optimal_factory(self, reference_model: CompressedVector) -> bool:
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
        s, p, k = self.instance.solve_for_target(self.targets, self.known_technologies, reference_model)
        same = p==self.optimal_pricing_model
        self.optimal_factory, self.optimal_pricing_model, self.inefficient_constructs = s, p, k

        return not same
        #u_j, p_j = self.instance.solve_looped_pricing_model(input, self.known_technologies)
        #pricing_model = CompressedVector({self.instance.reference_list[i]: p_j[i] for i in np.nonzero(p_j)[0]})
        #same = pricing_model==self.optimal_pricing_model
        #self.targets = CompressedVector({self.instance.reference_list[i]: u_j[i] for i in np.nonzero(u_j)[0]})
        #self.optimal_factory, self.optimal_pricing_model = input, pricing_model

    def retarget(self, targets: CompressedVector, retainment: Fraction = Fraction(1/1e6)) -> None:
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
        try:
            targets_df = pd.DataFrame(list(self.targets.items()), columns=['target', 'count'])
            optimal_factory_df = pd.DataFrame(list(self.optimal_factory.items()), columns=['construct', 'count'])
            optimal_pricing_model_df = pd.DataFrame(list(self.optimal_pricing_model.items()), columns=['item', 'value'])
            inefficient_constructs_df = pd.DataFrame(list(self.inefficient_constructs.items()), columns=['construct', 'relative value'])
            merged_df = pd.concat([targets_df, pd.DataFrame({}, columns=['']), 
                                optimal_factory_df, pd.DataFrame({}, columns=['']),
                                optimal_pricing_model_df, pd.DataFrame({}, columns=['']),
                                inefficient_constructs_df], axis=1)
            merged_df.to_excel(writer, sheet_name=sheet_name, index=False)
        except:
            return


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
        covering_to = technological_limitation_from_specification(self.instance.data_raw, self.instance.COST_MODE, fully_automated=self.clear) + \
                      TechnologicalLimitation([set([target for target in science_targets.keys() if target in self.instance.data_raw['technology'].keys()])])
        
        last_coverage = self._previous_coverage()

        if time_target is None:
            self.time_target = self.instance.DEFAULT_TARGET_RESEARCH_TIME
        else:
            self.time_target = time_target

        targets = CompressedVector({k+"=research": 1 / self.time_target for k in next(iter(covering_to.canonical_form)) if k not in next(iter(last_coverage.canonical_form))}) #next(iter()) gives us the first (and theoretically only) set of nodes making up the tech limit

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
        return self._previous_coverage() + TechnologicalLimitation([set([targ[:targ.rfind("=")] for targ in self.targets.keys()])])

    def retarget(self, science_targets: CompressedVector, retainment: Fraction = Fraction(1/1e6)) -> None:
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
        assert not any([target in self.instance.data_raw['tool'].keys() for target in science_targets.keys()]), "retarget should NEVER be given a tool. Only researches."
        covering_to = technological_limitation_from_specification(self.instance.data_raw, self.instance.COST_MODE, fully_automated=self.clear) + \
                      TechnologicalLimitation([set([target for target in science_targets.keys()])])
        last_coverage = self._previous_coverage()
        
        targets = CompressedVector({k: 1 / self.time_target for k in next(iter(covering_to.canonical_form)) if k not in next(iter(last_coverage))})

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
    
    def calculate_optimal_factory(self, reference_model: CompressedVector) -> bool:
        """
        Placeholder. Initial Factories cannot change.
        """
        return False
    
    def get_technological_coverage(self) -> TechnologicalLimitation:
        """
        Get the tech level unlocked by default.
        """
        return self.known_technologies
    
    def retarget(self, targets: CompressedVector, retainment: Fraction = Fraction(1/1e6)) -> None:
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
    """
    instance: FactorioInstance
    chain: list[FactorioFactory]

    def __init__(self, instance: FactorioInstance) -> None:
        assert isinstance(instance, FactorioInstance)
        self.instance = instance
        self.chain = []

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
            R_j_i, c_i, N1, N0, FactoryRecovery = self.instance.compiled.reduce(p0_j, [self.instance.reference_list.index(k) for k in last_pricing_model.keys()], known_technologies)

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
            logging.info("Attempting to add the following targets to the new material factory:\n\t"+str(set(targets_dict.keys()).difference(set(last_material.targets.keys()))))
            targets = targets_dict

        if factory_type == "science":
            for target in targets.keys():
                assert target in self.instance.data_raw['tool'].keys() or target in self.instance.data_raw['technology'].keys(), "If a factory does science stuff is only allowed to do science stuff."
            previous_science = previous_sciences[-1] if len(previous_sciences)>0 else known_technologies
            self.chain.append(FactorioScienceFactory(self.instance, previous_science, last_material, targets))
        else: #factory_type == "material":
            last_science = previous_sciences[-1]
            self.chain.append(FactorioMaterialFactory(self.instance, last_science, last_material, targets))
        
        self.chain[-1].calculate_optimal_factory(last_material.optimal_pricing_model)
    
    def retarget_all(self) -> None:
        """
        Retargets all factories in the chain.
        """
        for i in range(len(self.chain)):
            factory = self.chain[i]
            if isinstance(factory, InitialFactory):
                pass #cannot be updated
            elif isinstance(factory, FactorioMaterialFactory):
                updated_targets = CompressedVector()
                for j in range(i+1, len(self.chain)):
                    for construct_name, count in self.chain[j].optimal_factory.items():
                        for k, v in index_compiled_constructs(self.instance.compiled_constructs, construct_name).cost.items():
                            updated_targets.key_addition(k, count * v)
                    if isinstance(self.chain[j], FactorioMaterialFactory):
                        break #done with this material factories outputs
                factory.retarget(updated_targets)
            elif isinstance(factory, FactorioScienceFactory):
                pass #we don't update science factories at the moment

    def compute(self) -> bool:
        """
        Computes all pricing models for chain iteratively and returns if the pricing models changed.

        Returns
        -------
        True if pricing model has been changed in any factory.
        """
        self.retarget_all()
        changed = False

        last_reference_model = self.chain[0].optimal_pricing_model
        for fac in self.chain[1:]:
            changed = fac.calculate_optimal_factory(last_reference_model) or changed
            if isinstance(fac, FactorioMaterialFactory):
                last_reference_model = fac.optimal_pricing_model
        
        return changed

    def compute_iteratively(self) -> None:
        """
        Computes the chain then iterates on it until pricing models no longer change. Useful for refining a chain.
        """
        while self.compute():
            self.retarget_all()
            
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
                factory.dump_to_excel(writer, "Material Factory "+str(material_factory_ident))
                material_factory_ident += 1
            elif isinstance(factory, FactorioScienceFactory):
                factory.dump_to_excel(writer, "Science Factory "+str(science_factory_ident))
                science_factory_ident += 1
            #else: #if isinstance(factory, InitialFactory):
        writer.close()