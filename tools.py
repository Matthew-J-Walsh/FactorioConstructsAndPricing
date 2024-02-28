from __future__ import annotations
import json
import logging
logging.basicConfig(level=logging.INFO)
from utils import *
from generators import *
from linearconstructs import *
from scipysolvers import *
from datarawparse import *
import Levenshtein
from numbers import Real


class FactorioInstance():
    """
    Holds the information in an instance (specific game mod setup) after completing premanagment steps.

    Members
    -------
    data_raw:
        Whole data.raw dictonary post-premanagment.
    families:
        LinearConstructFamilys that exist in this Factorio instance.
    reference_list:
        The universal reference list for this Factorio instance.
    catalyst_list:
        List of all catalytic item/fluids.
    """
    data_raw: dict
    uncompiled_constructs: list[UncompiledConstruct]
    reference_list: list[str]
    catalyst_list: list[str]
    COST_MODE: str
    DEFAULT_TARGET_RESEARCH_TIME: Fraction
    RELEVENT_FLUID_TEMPERATURES: dict
    MODULE_REFERENCE: dict

    def __init__(self, filename: str, COST_MODE: str = 'normal', DEFAULT_TARGET_RESEARCH_TIME: Real = 10000) -> None:
        with open(filename) as f:
            self.data_raw = json.load(f)
        
        self.COST_MODE = COST_MODE
        self.DEFAULT_TARGET_RESEARCH_TIME = DEFAULT_TARGET_RESEARCH_TIME
        self.RELEVENT_FLUID_TEMPERATURES = {} #keys are fluid names, values are a dict with keys of temperature and values of energy density
        self.MODULE_REFERENCE = {} #Keys are names and values are Module class instances form linearconstructs.py

        complete_premanagement(self.data_raw, self.RELEVENT_FLUID_TEMPERATURES, self.MODULE_REFERENCE, COST_MODE=self.COST_MODE)
        constructs = generate_all_constructs(self.data_raw, self.RELEVENT_FLUID_TEMPERATURES, COST_MODE=self.COST_MODE)
        self.uncompiled_constructs, self.reference_list, self.catalyst_list = generate_all_construct_families(constructs)
    
    def solve_for_target(self, targets: CompressedVector, known_technologies: TechnologicalLimitation, reference_model: CompressedVector) -> tuple[CompressedVector, np.ndarray]:
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
        p_j:
            Pricing model of resulting factory.
        """
        n = len(self.reference_list)

        constructs = sum([f.get_constructs(self.reference_list, self.catalyst_list, known_technologies, self.MODULE_REFERENCE) for f in self.uncompiled_constructs], [])
        m = len(constructs)
        R_i_j = sp.sparse.vstack([construct.vector for construct in constructs]).T

        u_j = np.zeros(n)
        for k, v in targets.items():
            u_j[self.reference_list.index(k)] = v

        p0_i = np.full(n, np.inf) #TODO: Fractional infinity?
        for k, v in reference_model.items():
            p0_i[self.reference_list.index(k)] = v

        c_i = np.array([np.dot(p0_i, construct.cost) for construct in constructs])

        s_i = solve_optimization_problem(R_i_j, u_j, c_i)
        assert np.logical_or(np.isclose(R_i_j @ s_i, u_j), R_i_j @ s_i >= u_j).all(), "Somehow solution infeasible?"

        p_j = calculate_pricing_model_via_optimal(R_i_j, s_i, u_j, c_i)

        s = CompressedVector()
        for i in range(m):
            if s_i[i] != 0:
                s[constructs[i].ident] = s_i[i]
        
        return s, p_j
    
    def solve_looped_pricing_model(self, starter_base: CompressedVector, known_technologies: TechnologicalLimitation) -> sp.sparse.sparray:
        """
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
        Pricing model of given factory.
        """
        constructs = sum([f.get_constructs(self.reference_list, self.catalyst_list, known_technologies, self.MODULE_REFERENCE) for f in self.uncompiled_constructs], [])

        translated_base = CompressedVector()
        for k, v in starter_base.items():
            best_matches = []
            match_distance = Levenshtein.distance(k, constructs[0].ident)
            for c in constructs:
                dist = Levenshtein.distance(k, c.ident)
                if dist < match_distance:
                    best_matches = [c.ident]
                elif dist == match_distance:
                    best_matches.append(c.ident)
            assert len(best_matches)==1, "Unable to determine which construct an input phrase is associated with.\nPhrase is: "+k+"\nPossible constructs were:\n\t"+"\n\t".join([m for m in best_matches])
            translated_base[best_matches[0]] = v
            logging.debug("Translated: \""+k+"\" to mean the construct: "+best_matches[0])
        
        n = len(self.reference_list)

        R_i_j = sp.sparse.vstack([construct.vector for construct in constructs]).T
        m = len(constructs)

        s_i = np.zeros(m)
        for k, v in translated_base:
            s_i[next(c for c in constructs if c.ident==k)] = v

        u_j = R_i_j @ s_i

        C_i_j = sp.sparse.vstack([construct.cost for construct in constructs])

        reference_index = np.argmax(u_j)[0]

        p_j = calculate_pricing_model_via_prebuilt(R_i_j, C_i_j, s_i, u_j, reference_index)
        
        return p_j


class FactorioFactory():
    """
    A factory in factorio.

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
    """
    instance: FactorioInstance
    known_technologies: TechnologicalLimitation
    targets: CompressedVector
    optimal_factory: list[tuple[LinearConstruct, int]]
    optimal_pricing_model: CompressedVector

    def __init__(self, instance: FactorioInstance, known_technologies: TechnologicalLimitation, targets: CompressedVector) -> None:
        self.instance = instance
        self.known_technologies = known_technologies
        self.targets = targets
        self.optimal_pricing_model = CompressedVector()

    def calculate_optimal_factory(self, input: CompressedVector, looped: bool = False) -> bool:
        """
        Calculates a optimal factory with the now avaiable reference model.

        Parameters
        ----------
        input:
            If looped is false:
                CompressedVector of the reference to use to decide how much every component costs.
            If looped is true:
                CompressedVector that describes what the factory should be.
        looped:
            Decider for input.

        Returns
        -------
        If pricing model has changed.
        """
        if not looped: #build it based on some other reference
            s, p_j = self.instance.solve_for_target(self.targets, self.known_technologies, input)
            same = p_j==self.optimal_pricing_model
        else: #isinstance(reference_model, FactorioFactory): #build it based on itself
            p_j = self.instance.solve_looped_pricing_model(input, self.known_technologies)
            same = False
        self.optimal_factory, self.optimal_pricing_model = input, p_j

        return same
    
    def time_estimate(self) -> Fraction:
        """
        Estimate how long it will take this factory to complete all the work assigned to it. Optimal factory must already be calculated.
        TODO: Time estimate is going to be disaster
        """
        raise NotImplementedError

    def retarget(self, targets: CompressedVector, retainment: Fraction = Fraction(1/1e10)) -> None:
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


class FactorioMaterialFactory(FactorioFactory):
    """
    A factory in factorio that makes materials (parts for future factories).

    Added Members
    -------------
    last_material_factory
        Factory to base the pricing model of this factory on
    """
    last_material_factory: FactorioMaterialFactory

    def __init__(self, instance: FactorioInstance, last_science_factory: FactorioScienceFactory, 
                 last_material_factory: FactorioMaterialFactory, material_targets: CompressedVector) -> None:
        if not last_science_factory is None:
            super().__init__(instance, last_science_factory.get_technological_coverage(), material_targets)
        else: #we probably got here via self.startup_base
            super().__init__(instance, last_science_factory, material_targets)
        self.last_material_factory = last_material_factory
    
    @classmethod
    def startup_base(cls, instance: FactorioInstance, base_building_setup: CompressedVector, 
                     starting_techs: TechnologicalLimitation, material_targets: CompressedVector) -> FactorioMaterialFactory:
        """
        Alternative initialization from a prebuilt base.

        Parameters
        ----------
        instance:
            FactorioInstance associated with this factory.
        base_building_setup:
            CompressedVector that describes what the factory should be.
        starting_techs:
            Starting tech limitations if needed.
        material_targets:
            Target output of the factory.
        """
        inst = cls(instance, starting_techs, None, material_targets)
        inst.last_material_factory = inst
        inst.optimal_pricing_model = inst.calculate_optimal_factory(base_building_setup, True)

        return inst


class FactorioScienceFactory(FactorioFactory):
    """
    A factory in factorio that completes research.

    Added Members
    -------------
    last_material_factory:
        Factory to base the pricing model of this factory on.
    previous_science_factories:
        List of previous science factories.
    time_target:
        How long it should take to complete all research.
    """
    last_material_factory: FactorioMaterialFactory
    previous_science_factories: list[FactorioScienceFactory]
    time_target: Fraction

    def __init__(self, instance: FactorioInstance, previous_science: list[FactorioScienceFactory] | TechnologicalLimitation, last_material_factory: FactorioMaterialFactory, 
                 science_target: CompressedVector, time_target: Fraction | None = None) -> None:
        if isinstance(previous_science, TechnologicalLimitation):
            super().__init__(instance, previous_science, science_target)
        else:
            super().__init__(instance, previous_science[-1].get_technological_coverage(), science_target)
        self.last_material_factory = last_material_factory
        self.previous_science_factories = previous_science
        if time_target is None:
            self.time_target = self.instance.DEFAULT_TARGET_RESEARCH_TIME
        else:
            self.time_target = time_target
    
    def get_technological_coverage(self) -> TechnologicalLimitation:
        """
        Determine what technologies will be unlocked when this factory is done.
        """
        fully_automated_units = set()
        for targets in [self.targets] + [fac.targets for fac in self.previous_science_factories]:
            for target in targets.keys():
                if target in self.instance.data_raw['tool'].keys():
                    fully_automated_units.add(target)

        return technological_limitation_from_specification(self.instance.data_raw, fully_automated=list(fully_automated_units), COST_MODE=self.instance.COST_MODE)


class FactorioFactoryChain():
    """
    A chain of optimal factory designs starting from the minimal science to a specified point. There are two avaiable types of factory in the chain:
        A science factory that produces a set of science packs. Having one of these will permit any recipe that is unlockable with those sciences to be used in factories after it.
        A component factory that produces a set of buildings. Having one of these will set the pricing model for every factory in the chain up until the next component factory.
    """
    instance: FactorioInstance
    chain: list[FactorioMaterialFactory | FactorioScienceFactory]

    def __init__(self, instance: FactorioInstance) -> None:
        self.instance = instance
        self.chain = []

    def startup(self, base_building_setup: CompressedVector, starting_techs: TechnologicalLimitation, material_targets: CompressedVector) -> None:
        """
        Initialize the first factory of the chain from a prebuilt design.

        Parameters
        ----------
        base_building_setup:
            CompressedVector that describes what the starting factory should be.
        starting_techs:
            TechnologicalLimitation of techs unlocked before building a starting factory.
        material_targets:
            Target output of the starting factory.
        """
        self.chain.append(FactorioMaterialFactory.startup_base(self.instance, base_building_setup, starting_techs, material_targets))
    
    def add(self, targets: CompressedVector) -> None:
        """
        Adds a factory to the chain.

        Parameters
        ----------
        targets:
            CompressedVector of target outputs for the factory. Must be either all science tools or building materials
        """
        factory_type = targets.keys()[0] in self.instance.data_raw['tool'].keys() #True if science, False if non-science
        if factory_type:
            for target in targets.keys():
                assert target in self.instance.data_raw['tool'].keys(), "If a factory makes science tools it must only make science tools"
            previous_sciences = [fac for fac in self.chain if isinstance(fac, FactorioScienceFactory)]
            last_material = [fac for fac in self.chain if isinstance(fac, FactorioMaterialFactory)][0]
            self.chain.append(FactorioScienceFactory(self.instance, previous_sciences, last_material, targets))
        else:
            for target in targets.keys():
                assert not target in self.instance.data_raw['tool'].keys(), "If a factory makes building materials it must only make building materials"
            last_science = [fac for fac in self.chain if isinstance(fac, FactorioScienceFactory)][0]
            last_material = [fac for fac in self.chain if isinstance(fac, FactorioMaterialFactory)][0]
            self.chain.append(FactorioMaterialFactory(self.instance, last_science, last_material, targets))

    def compute(self) -> bool:
        """
        Computes all pricing models for chain iteratively and returns if the pricing models changed.
        """
        changed = False

        last_reference_model = self.chain[0].optimal_pricing_model
        for fac in self.chain[1:]:
            changed = changed and fac.calculate_optimal_factory(last_reference_model)
            if isinstance(fac, FactorioMaterialFactory):
                last_reference_model = fac.optimal_pricing_model
        
        return changed
    
    def compute_iteratively(self) -> None:
        """
        Computes the chain then iterates on it until pricing models no longer change.
        Completing this should generally reduce the time_estimate
        """
        while not self.compute():
            for i, fac in enumerate(self.chain):
                if isinstance(fac, FactorioScienceFactory):
                    target_factory = [sci_fac for sci_fac in self.chain[i+1:] if isinstance(sci_fac, FactorioScienceFactory)]
                    if len(target_factory)>0:
                        target_factory = target_factory[0]
                    else:
                        target_factory = self.chain[-1] #no more science factories to make, therefor we will target the final factory.
                    techs_to_research = target_factory.known_technologies - fac.known_technologies
                    new_target = CompressedVector({tech: 1 for tech in techs_to_research})

                else: #if isinstance(fac, FactorioMaterialFactory):
                    new_target = CompressedVector()
                    for j in range(i+1, len(self.chain)):
                        if isinstance(self.chain[j], FactorioMaterialFactory):
                            break
                        for const, count in self.chain[j].optimal_factory:
                            new_target = new_target + count * const.cost

                fac.retarget(new_target)
            
    
    def time_estimate(self) -> Fraction:
        """
        Estimate how long it takes for this chain to complete.
        G_n(x) = min(1, integral from 0 to x of rate * G_{n-1}(t) dt)
        T = argmin(G_n(x)==1)
        """
        raise NotImplementedError
    
    

