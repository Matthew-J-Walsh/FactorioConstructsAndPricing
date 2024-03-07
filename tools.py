from __future__ import annotations

import json
import Levenshtein
import logging
logging.basicConfig(level=logging.INFO)
import scipy as sp
import scipy.sparse

from utils import *
from generators import *
from linearconstructs import *
from datarawparse import *
from lpproblems import *

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
    compiled_constructs: list[LinearConstruct]
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

        complete_premanagement(self.data_raw, self.RELEVENT_FLUID_TEMPERATURES, self.MODULE_REFERENCE, self.COST_MODE)
        constructs = generate_all_constructs(self.data_raw, self.RELEVENT_FLUID_TEMPERATURES, self.MODULE_REFERENCE, COST_MODE=self.COST_MODE)
        self.uncompiled_constructs, self.reference_list, self.catalyst_list = generate_all_construct_families(constructs)
        self.compiled_constructs = sum([uc.get_constructs(self.reference_list, self.catalyst_list, self.MODULE_REFERENCE) for uc in self.uncompiled_constructs], [])
    
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

        #print(targets.keys())
        constructs = valid_construct_iterative(self.compiled_constructs, known_technologies, reference_model)
        logging.info("valid_construct len: "+str(len(constructs)))
        removals = set()
        #uncomment if highs throws errors:
        for k in targets.keys():
            if not any([k in construct.vector.keys() and construct.vector[k] > 0 for construct in constructs]):
                logging.warning("Cannot find a valid way of crafting "+str(k)+" removing it from the target.")
                removals.add(k)
            for c1 in constructs:
                if k in c1.vector.keys() and c1.vector[k] > 0:
                    for inp in [k2 for k2, v in c1.vector.items() if v < 0]:
                        if not any([inp in construct.vector.keys() and construct.vector[inp] > 0 for construct in constructs]):
                            logging.warning("Cannot find a valid way of crafting "+str(inp)+" for "+str(k)+" removing "+str(k)+" from the target.")
                            removals.add(k)
        for k in removals:
            del targets[k]

        m = len(constructs)
        R_j_i, C_i_j = ConstructTransform(constructs, self.reference_list).to_dense()
        R_j_i = R_j_i.T

        p0_j = np.full(n, Fraction(1e100), dtype=Fraction)
        for k, v in reference_model.items():
            p0_j[self.reference_list.index(k)] = v

        c_i = C_i_j @ p0_j
        assert (c_i!=0).all(), "Some suspiciously priced things"
        #print("Out")
        #print(R_i_j.shape)
        #print(u_j.shape)
        #print(c_i.shape)
        #print(R_i_j.dtype)
        #print(u_j.dtype)
        #print(c_i.dtype)
        R_j_i = R_j_i.astype(np.longdouble)
        c_i = c_i.astype(np.longdouble)
        assert not (c_i >= 1e50).any(), (c_i >= 1e50)

        u_j = np.zeros(n, dtype=Fraction)
        for k, v in targets.items():
            u_j[self.reference_list.index(k)] = v
        u_j = u_j.astype(np.longdouble)

        s_i = solve_factory_optimization_problem(R_j_i, u_j, c_i)

        #try:
        #    u_j = np.zeros(n, dtype=Fraction)
        #    for k, v in targets.items():
        #        u_j[self.reference_list.index(k)] = v
        #    u_j = u_j.astype(np.double)
        #
        #    s_i = solve_optimization_problem(R_j_i, u_j, c_i)
        #except:
        #    logging.warning("Solve optimization failed. Trying to figure out why.")
        #    s_i = np.zeros_like(c_i)
        #    for k, v in targets.items():
        #        try:
        #            logging.warning("Trying "+k)
        #            u_j = np.zeros(n, dtype=Fraction)
        #            u_j[self.reference_list.index(k)] = v
        #            u_j = u_j.astype(np.double)
        #            partial_s_i = solve_optimization_problem(R_j_i, u_j, c_i)
        #            s_i = s_i + partial_s_i
        #        except:
        #            logging.warning(k+" failed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            #np.savetxt("A.txt", R_j_i)
            #np.savetxt("b.txt", u_j)
            #np.savetxt("c.txt", c_i)
            #np.savetxt("x.txt", s_i)
            #assert linear_transform_is_gt(R_j_i, subs, u_j).all(), "Kekw. Nice program."
            #raise ValueError()
        #print(s_i)
        #print(R_i_j @ s_i)
        try:
            assert linear_transform_is_gt(R_j_i, s_i, u_j).all(), "Somehow solution infeasible?"
        except:
            nonzero = np.nonzero(1-linear_transform_is_gt(R_j_i, s_i, u_j))[0]
            for i in nonzero:
                print("assertion failure on: "+self.reference_list[i]+" values are "+str((R_j_i @ s_i)[i])+" vs "+str(u_j[i]))
            raise AssertionError

        p_j = solve_pricing_model_calculation_problem(R_j_i, s_i, u_j, c_i)
        p_j = p_j / np.max(p_j) * 100

        s = CompressedVector()
        for i in range(m):
            if s_i[i] != 0:
                s[constructs[i].ident] = s_i[i]

        #logging.info("Shapes")
        #logging.info(R_j_i.shape)
        #logging.info(s_i.shape)
        #logging.info(R_j_i.T.shape)
        #logging.info(p_j.shape)

        #logging.info("Used "+str(np.sum(s_i!=0))+" total constructs")
        #value = R_j_i.T @ p_j
        #value_to_cost = (value / c_i)[np.nonzero(s_i)[0]]
        #logging.info(value_to_cost)
        #logging.info("Value to cost spread: "+str((np.max(value_to_cost)-np.min(value_to_cost))/np.min(value_to_cost)))

        return s, p_j
    
    def solve_looped_pricing_model(self, starter_base: CompressedVector, known_technologies: TechnologicalLimitation) -> tuple[np.ndarray, sp.sparse.sparray]:
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
        """
        return technological_limitation_from_specification(self.data_raw, fully_automated=fully_automated, extra_technologies=extra_technologies, extra_recipes=extra_recipes, COST_MODE=self.COST_MODE)

    def all_buildings_at_tech(self, limit: TechnologicalLimitation) -> list[str]:
        """
        Returns a list of all buildings that can be built while at a specific tech level
        """
        raise NotImplementedError


def valid_construct(construct: LinearConstruct, known_technologies, reference_model):
    """
    Is this construct formable?
    """
    if known_technologies >= construct.limit:
        if all([c in reference_model.keys() for c in construct.cost.keys()]):
            return True
    return False

def valid_construct_logic_helper(constructs, k):
    """
    not any([any([ok==k and ov > 0 for ok, ov in oc.vector.items()]) for oc in constructs])
    """
    for oc in constructs:
        for ok, ov in oc.vector.items():
            if ok==k and ov > 0:
                return False
    return True

def valid_construct_iterative(constructs: list[LinearConstruct], known_technologies, reference_model):
    """
    Iteratively removes constructs that cannot be completed due to lack of inputs
    """
    constructs = list(filter(lambda construct: valid_construct(construct, known_technologies, reference_model), constructs))
    i = 0
    while i < len(constructs):
        for k, v in constructs[i].vector.items():
            if v < 0 and valid_construct_logic_helper(constructs, k):
                logging.debug("Deleting construct "+constructs[i].ident+" because we couldn't find a way to make "+k+".")
                constructs.pop(i)
                i = 0
                break
        i += 1
    return constructs
def valid_construct_pass_three(construct: LinearConstruct, others: list[LinearConstruct]):
    """
    Is this research construct actualy completable?
    """
    if not any(["=research" in k for k in construct.vector.keys()]): #true iff not a research construct
        return True
    for k, v in construct.vector.items():
        if v < 0 and not any([any([ok==k and ov > 0 for ok, ov in oc.vector.items()]) for oc in others]):
            return False
    return True


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
        assert isinstance(instance, FactorioInstance)
        assert isinstance(known_technologies, TechnologicalLimitation)
        assert isinstance(targets, CompressedVector)
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
            #pricing_model = CompressedVector({self.instance.reference_list[i]: p_j[i] for i in np.nonzero(p_j)[0]})
            pricing_model = CompressedVector({k: p_j[self.instance.reference_list.index(k)] for k in self.targets.keys()}) + CompressedVector({self.instance.reference_list[i]: p_j[i] for i in np.nonzero(p_j)[0]})
            same = pricing_model==self.optimal_pricing_model
            pricing_model = CompressedVector({k:v for k, v in pricing_model.items() if not '-module' in k})
            self.optimal_factory, self.optimal_pricing_model = s, pricing_model
        else: #isinstance(reference_model, FactorioFactory): #build it based on itself
            raise NotImplementedError
            u_j, p_j = self.instance.solve_looped_pricing_model(input, self.known_technologies)
            pricing_model = CompressedVector({self.instance.reference_list[i]: p_j[i] for i in np.nonzero(p_j)[0]})
            same = pricing_model==self.optimal_pricing_model
            self.targets = CompressedVector({self.instance.reference_list[i]: u_j[i] for i in np.nonzero(u_j)[0]})
            self.optimal_factory, self.optimal_pricing_model = input, pricing_model

        return same
    
    def time_estimate(self) -> Fraction:
        """
        Estimate how long it will take this factory to complete all the work assigned to it. Optimal factory must already be calculated.
        TODO: Time estimate is going to be disaster
        """
        raise NotImplementedError

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


class InitialFactory(FactorioFactory):
    """
    A fake factory instance to hold a pricing model and tech level.
    """

    def __init__(self, instance: FactorioInstance, pricing_model: CompressedVector, known_technologies: TechnologicalLimitation) -> None:
        assert isinstance(instance, FactorioInstance)
        assert isinstance(pricing_model, CompressedVector)
        assert isinstance(known_technologies, TechnologicalLimitation)
        self.instance = instance
        self.known_technologies = known_technologies
        self.targets = CompressedVector()
        self.optimal_pricing_model = pricing_model
    
    def calculate_optimal_factory(self, input: CompressedVector, looped: bool = False) -> bool:
        return False
    
    def get_technological_coverage(self) -> TechnologicalLimitation:
        return self.known_technologies
    
    def retarget(self, targets: CompressedVector, retainment: Fraction = Fraction(1/1e6)) -> None:
        return None


class FactorioMaterialFactory(FactorioFactory):
    """
    A factory in factorio that makes materials (parts for future factories).

    Added Members
    -------------
    last_material_factory
        Factory to base the pricing model of this factory on
    """
    last_material_factory: FactorioMaterialFactory

    def __init__(self, instance: FactorioInstance, last_science_factory: FactorioScienceFactory | InitialFactory | TechnologicalLimitation, 
                 last_material_factory: FactorioMaterialFactory, material_targets: CompressedVector) -> None:
        assert isinstance(instance, FactorioInstance)
        assert isinstance(last_science_factory, FactorioScienceFactory) or isinstance(last_science_factory, InitialFactory) or isinstance(last_science_factory, TechnologicalLimitation)
        assert isinstance(last_material_factory, FactorioMaterialFactory) or isinstance(last_material_factory, InitialFactory)
        assert isinstance(material_targets, CompressedVector)
        if isinstance(last_science_factory, FactorioScienceFactory) or isinstance(last_science_factory, InitialFactory):
            super().__init__(instance, last_science_factory.get_technological_coverage(), material_targets)
        else: #we probably got here via self.startup_base
            super().__init__(instance, last_science_factory, material_targets)
        self.last_material_factory = last_material_factory
    
    @classmethod
    def startup_base(cls, instance: FactorioInstance, base_building_setup: CompressedVector, 
                     starting_techs: TechnologicalLimitation) -> FactorioMaterialFactory:
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
        """
        assert isinstance(instance, FactorioInstance)
        assert isinstance(base_building_setup, CompressedVector)
        assert isinstance(starting_techs, TechnologicalLimitation)
        inst = cls(instance, starting_techs, None, None)
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
        covering_to = technological_limitation_from_specification(self.instance.data_raw, fully_automated=self.clear, COST_MODE=self.instance.COST_MODE) + \
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
        covering_to = technological_limitation_from_specification(self.instance.data_raw, fully_automated=self.clear, COST_MODE=self.instance.COST_MODE) + \
                      TechnologicalLimitation([set([target for target in science_targets.keys()])])
        last_coverage = self._previous_coverage()
        
        targets = CompressedVector({k: 1 / self.time_target for k in next(iter(covering_to.canonical_form)) if k not in next(iter(last_coverage))})

        super().retarget(targets, retainment=retainment)


class FactorioFactoryChain():
    """
    A chain of optimal factory designs starting from the minimal science to a specified point. There are two avaiable types of factory in the chain:
        A science factory that produces a set of science packs. Having one of these will permit any recipe that is unlockable with those sciences to be used in factories after it.
        A component factory that produces a set of buildings. Having one of these will set the pricing model for every factory in the chain up until the next component factory.
    """
    instance: FactorioInstance
    chain: list[FactorioMaterialFactory | FactorioScienceFactory | InitialFactory]

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
                    "all materials" = will autopopulate all possible buildings and materials that can be made (not including tools TODO: may cause issues if tools are used in recipes as catalysts)
                    "all tech" = will populate all possible tech that can be made
        """
        if isinstance(targets, CompressedVector):
            if list(targets.keys())[0] in self.instance.data_raw['tool'].keys() or list(targets.keys())[0] in self.instance.data_raw['technology'].keys():
                factory_type = "science"
            elif list(targets.keys())[0] in self.instance.data_raw['item'].keys() or list(targets.keys())[0] in self.instance.data_raw['fluid'].keys():
                factory_type = "material"
        else:
            previous_sciences = [fac for fac in self.chain if isinstance(fac, FactorioScienceFactory) or isinstance(fac, InitialFactory)]
            last_material = [fac for fac in self.chain if isinstance(fac, FactorioMaterialFactory) or isinstance(fac, InitialFactory)][-1]
            if len(previous_sciences)==0:
                known_technologies = last_material.known_technologies
            else:
                known_technologies = previous_sciences[-1].get_technological_coverage()
            craftable_constructs = valid_construct_iterative(self.instance.compiled_constructs, known_technologies, last_material.optimal_pricing_model)

            if targets=="all tech":
                #print(len(self.instance.uncompiled_constructs))
                #print(len(unlocked_constructs))
                #for tool in self.instance.data_raw['tool'].keys():
                #    for uc in unlocked_constructs:
                #        if tool in uc.deltas.keys() and uc.deltas[tool] > 0:
                #            print(tool)
                #            print(uc)
                #            break
                factory_type = "science"
                targets = CompressedVector({tool: Fraction(1) for tool in self.instance.data_raw['tool'].keys() if any([tool in construct.vector.keys() and construct.vector[tool] > 0 for construct in craftable_constructs])})
            elif targets=="all materials":
                factory_type = "material"
                #only actually craftable materials
                targets = CompressedVector({item: Fraction(1) for item in self.instance.data_raw['item'].keys() if any([item in construct.vector.keys() and construct.vector[item] > 0 for construct in craftable_constructs])}) + \
                          CompressedVector({fluid: Fraction(1) for fluid in self.instance.data_raw['fluid'].keys() if any([fluid in construct.vector.keys() and construct.vector[fluid] > 0 for construct in craftable_constructs])}) + \
                          CompressedVector({module: Fraction(1) for module in self.instance.data_raw['module'].keys() if any([module in construct.vector.keys() and construct.vector[module] > 0 for construct in craftable_constructs])}) + \
                          CompressedVector({'electric': Fraction(1)} if any(['electric' in construct.vector.keys() and construct.vector['electric'] > 0 for construct in craftable_constructs]) else {})
            logging.info(targets.keys())

        if factory_type == "science":
            for target in targets.keys():
                assert target in self.instance.data_raw['tool'].keys() or target in self.instance.data_raw['technology'].keys(), "If a factory does science stuff is only allowed to do science stuff."
            previous_sciences = [fac for fac in self.chain if isinstance(fac, FactorioScienceFactory) or isinstance(fac, InitialFactory)]
            last_material = [fac for fac in self.chain if isinstance(fac, FactorioMaterialFactory) or isinstance(fac, InitialFactory)][-1]
            if len(previous_sciences)==0:
                previous_science = last_material.known_technologies
            else:
                previous_science = previous_sciences[-1]
            self.chain.append(FactorioScienceFactory(self.instance, previous_science, last_material, targets))
        else: #factory_type == "material":
            #for target in targets.keys():
                #assert target in self.instance.data_raw['item'].keys() or target in self.instance.data_raw['fluid'].keys() or target in self.instance.data_raw['module'].keys(), "If a factory makes building materials it must only make building materials"
            last_science = [fac for fac in self.chain if isinstance(fac, FactorioScienceFactory) or isinstance(fac, InitialFactory)][-1]
            last_material = [fac for fac in self.chain if isinstance(fac, FactorioMaterialFactory) or isinstance(fac, InitialFactory)][-1]
            self.chain.append(FactorioMaterialFactory(self.instance, last_science, last_material, targets))

    def compute(self) -> bool:
        """
        Computes all pricing models for chain iteratively and returns if the pricing models changed.
        """
        changed = False

        last_reference_model = self.chain[0].optimal_pricing_model
        print(last_reference_model)
        for fac in self.chain[1:]:
            if isinstance(fac, FactorioMaterialFactory):
                changed = fac.calculate_optimal_factory(last_reference_model) or changed
                last_reference_model = fac.optimal_pricing_model
                print(last_reference_model)
        
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
            
