import pytest

from utils import *
from tools import *
from lpproblems import *


def compressed_vectors_test_1():
    compA = CompressedVector({"A": 1, "B": 2, "C": 3, "D": 5, "E": 7})
    compB = CompressedVector({"A": 2, "E": 1, "C": -1, "F": 2, "G": -4})
    added = CompressedVector({"A": 1, "B": 2, "C": 2, "D": 5, "E": 8, "F": 2, "G": -4})
    added_via_func = compA + compB
    assert added==added_via_func
    multiA = CompressedVector({"A": 2, "B": 4, "C": 6, "D": 10, "E": 14})
    multiA_via_func = 2 * compA
    assert multiA==multiA_via_func


@pytest.fixture
def tech_limit_testing_limits():
    base_limits = [TechnologicalLimitation([[tech]]) for tech in "A,B,C,D,E,F,G,H,I,J,K,L".split(",")]+[TechnologicalLimitation()]
    advanced_limits = [base_limits[0]+base_limits[1]+base_limits[3],
                       base_limits[2]+base_limits[3],
                       base_limits[4]+base_limits[2]+base_limits[6]+base_limits[11],
                       base_limits[0]+base_limits[10],
                       base_limits[6]+base_limits[4]+base_limits[3],
                       base_limits[0]+base_limits[9]+base_limits[9]]
    return base_limits, advanced_limits

def tech_limit_test_1(tech_limit_testing_limits):
    base_limits, advanced_limits = tech_limit_testing_limits
    assert base_limits[0] <= advanced_limits[0] and advanced_limits[0] > base_limits[0]
    assert base_limits[11] <= advanced_limits[2] and advanced_limits[2] > base_limits[11]
    assert base_limits[9] <= advanced_limits[5] and advanced_limits[5] > base_limits[9]
    assert advanced_limits[1] == advanced_limits[1] and advanced_limits[1] >= advanced_limits[1] and advanced_limits[1] <= advanced_limits[1]
    for bl in base_limits[:-1]:
        assert bl >= base_limits[-1]


def list_union_test():
    l1 = ["A", "B", "C", "D"]
    l2 = ["E", "F", "A", "B", "G"]
    actual = ["A", "B", "C", "D", "E", "F", "G"]
    via_func = list_union(l1, l2)

    assert set(actual)==set(via_func)
    

def convert_value_to_base_units_test():
    values = ["112k", "132K", "12M", "1.4M", "3.22E", "4Z"]
    real_values = [112000, 132000, 12000000, 1400000000, 3220000000000000000, 4000000000000000000000]
    via_func = [convert_value_to_base_units(val) for val in values]

    assert all([rv==vf for rv, vf in zip(real_values, via_func)])


def vanilla_instance_test():
    """
    ident: str
    drain: CompressedVector
    deltas: CompressedVector
    effect_effects: dict[str, list[str]]
    allowed_modules: list[tuple[str, bool, bool]]
    internal_module_limit: int
    base_inputs: CompressedVector
    cost: CompressedVector
    limit: TechnologicalLimitation
    base_productivity: Fraction
    universal_reference_list: list[str]
    catalyzing_deltas: list[str]
    """
    filename = 'vanilla-rawdata.json'
    vanilla = FactorioInstance(filename)
    for construct in vanilla.uncompiled_constructs:
        assert isinstance(construct.ident, str)
        assert isinstance(construct.drain, CompressedVector)
        for k, v in construct.drain.items():
            assert isinstance(k, str)
            assert isinstance(v, Fraction)
        assert isinstance(construct.deltas, CompressedVector)
        for k, v in construct.deltas.items():
            assert isinstance(k, str)
            assert isinstance(v, Fraction)
        assert isinstance(construct.effect_effects, dict)
        for k, v in construct.effect_effects.items():
            assert isinstance(k, str)
            assert isinstance(v, list)
            for e in v:
                assert isinstance(e, str)
                assert e in construct.deltas.keys()
        assert isinstance(construct.allowed_modules, list)
        for am in construct.allowed_modules:
            assert isinstance(am[0], str)
            assert isinstance(am[1], bool)
            assert isinstance(am[2], bool)
        assert isinstance(construct.internal_module_limit, int)
        assert isinstance(construct.base_inputs, CompressedVector)
        for k, v in construct.base_inputs.items():
            assert isinstance(k, str)
            assert isinstance(v, Fraction)
        assert isinstance(construct.cost, CompressedVector)
        for k, v in construct.cost.items():
            assert isinstance(k, str)
            assert isinstance(v, Fraction)
        assert isinstance(construct.limit, TechnologicalLimitation)
        assert isinstance(construct.base_productivity, Fraction)
        assert isinstance(construct.reference_list, list)
        assert isinstance(construct.catalyzing_deltas, list)


#TODO: testing for solvers