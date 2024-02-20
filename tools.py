from __future__ import annotations
import json
import logging
logging.basicConfig(level=logging.INFO)
from utils import *
from generators import *
from linearconstructs import *
from linearsolvers import *
from datarawparse import *



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
    families: list[LinearConstructFamily]
    reference_list: list[str]
    catalyst_list: list[str]

    def __init__(self, filename: str) -> None:
        with open(filename) as f:
            self.data_raw = json.load(f)

        complete_premanagement(self.data_raw)
        constructs = generate_all_constructs(self.data_raw)
        self.families, self.reference_list = generate_all_construct_families(constructs)
    
    def solve_for_target(self, targets: CompressedVector, known_technologies: TechnologicalLimitation, reference_model: CompressedVector) -> tuple[sp.sparray, sp.sparray]:
        """
        Wrapper for optimize_for_outputs_via_reference_model that feeds info about the Factorio instance where needed.
        """
        return optimize_for_outputs_via_reference_model(self.families, self.reference_list, targets, known_technologies, reference_model)
    
    def solve_looped_pricing_model(self, targets: CompressedVector, known_technologies: TechnologicalLimitation) -> tuple[sp.sparray, sp.sparray]:
        """
        Wrapper for calculate_pricing_model_via_prebuilt that feeds info about the Factorio instance where needed.
        """
        raise NotImplementedError
        return calculate_pricing_model_via_prebuilt()
    
    def translate_starter_base(self, base: CompressedVector):
        """
        Attempts to translate a starter base of just strings with values into something that makes a little more sense to the program.
        """
        raise NotImplementedError

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

    def calculate_optimal_factory(self, reference_model: CompressedVector | FactorioFactory) -> bool:
        """
        Calculates a optimal factory with the now avaiable reference model.

        Parameters
        ----------
        reference_model:
            CompressedVector of the reference to use to decide how much every component costs.

        Returns
        -------
        If pricing model has changed.
        """
        if isinstance(reference_model, CompressedVector): #build it based on some other reference
            s_i, p_j = self.instance.solve_for_target(self.targets, self.known_technologies, reference_model)
        else: #isinstance(reference_model, FactorioFactory): #build it based on itself
            p_j = self.instance.solve_looped_pricing_model(self.targets, self.known_technologies, reference_model)
        self.optimal_factory, self.optimal_pricing_model
        raise NotImplementedError
    
    def time_estimate(self) -> Fraction:
        """
        Estimate how long it will take this factory to complete all the work assigned to it. Optimal factory must already be calculated.
        NOTE: THIS IS A REALLY LOW ESTIMATE
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
    def startup_base(cls, instance: FactorioInstance, base_buildings: LinearConstruct, base_building_setup: CompressedVector, 
                     starting_techs: TechnologicalLimitation, material_targets: CompressedVector) -> FactorioMaterialFactory:
        """
        Alternative initialization from a prebuilt base.
        """
        inst = cls(instance, starting_techs, None, material_targets)
        inst.last_material_factory = inst
        inst.optimal_pricing_model = inst.calculate_optimal_factory()


        return inst


class FactorioScienceFactory(FactorioFactory):
    """
    A factory in factorio that completes research.

    Added Members
    -------------
    last_material_factory
        Factory to base the pricing model of this factory on
    """
    last_material_factory: FactorioMaterialFactory
    previous_science_factories: list[FactorioScienceFactory]

    def __init__(self, instance: FactorioInstance, previous_science_factories: list[FactorioScienceFactory], last_material_factory: FactorioMaterialFactory, science_target: CompressedVector):
        super().__init__(instance, previous_science_factories[-1].get_technological_coverage(), science_target)
        self.last_material_factory = last_material_factory
        self.previous_science_factories = previous_science_factories
    
    def get_technological_coverage(self) -> TechnologicalLimitation:
        """
        Determine what technologies will be unlocked when this factory is done.
        """
        return None


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

    def startup(self, base_buildings: LinearConstruct, base_building_setup: CompressedVector, 
                starting_techs: TechnologicalLimitation, material_targets: CompressedVector) -> None:
        """
        Initialize the first factory of the chain from a prebuilt design.
        """
        self.chain.append(FactorioMaterialFactory.startup_base(self.instance, base_buildings, base_building_setup, starting_techs, material_targets))
    
    def add(self, target: CompressedVector):
        """
        Adds a factory to the chain.
        """
        #science vs non-science?
        raise NotImplementedError
        chain.append("???")

    def compute(self) -> list[FactorioMaterialFactory | FactorioScienceFactory]:
        """
        Computes all pricing models for chain iteratively and returns it.
        """
        raise NotImplementedError
    
    def compute_iteratively(self) -> list[FactorioMaterialFactory | FactorioScienceFactory]:
        """
        Computes the chain then iterates on it until pricing models no longer change.
        Completing this should generally reduce the time_estimate
        """
        raise NotImplementedError
    
    def time_estimate(self) -> Fraction:
        """
        Estimate how long it takes for this chain to complete.
        """
        raise NotImplementedError
    
    

