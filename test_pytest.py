import pytest

from globalsandimports import *
from utils import *
from tools import *
from lpproblems import *
from vanillarun import vanilla_main

def test_compressed_vectors():
    compA = CompressedVector({"A": 1, "B": 2, "C": 3, "D": 5, "E": 7})
    compB = CompressedVector({"A": 2, "E": 1, "C": -1, "F": 2, "G": -4})
    added = CompressedVector({"A": 3, "B": 2, "C": 2, "D": 5, "E": 8, "F": 2, "G": -4})
    added_via_func = compA + compB
    print(added_via_func)
    print(added)
    assert added==added_via_func
    multiA = CompressedVector({"A": 2, "B": 4, "C": 6, "D": 10, "E": 14})
    multiA_via_func = compA * Fraction(2)
    assert multiA==multiA_via_func


def test_list_union():
    l1 = ["A", "B", "C", "D"]
    l2 = ["E", "F", "A", "B", "G"]
    actual = ["A", "B", "C", "D", "E", "F", "G"]
    via_func = list_union(l1, l2)

    assert set(actual)==set(via_func)
    

def test_convert_value_to_base_units():
    values = ["112k", "132K", "12M", "1.4M", "3.22E", "4Z"]
    real_values = [112000, 132000, 12000000, 1400000, 3220000000000000000, 4000000000000000000000]
    via_func = [convert_value_to_base_units(val) for val in values]

    assert all([rv==vf for rv, vf in zip(real_values, via_func)]), via_func


def test_vanilla_instance():
    filename = 'vanilla-rawdata.json'
    vanilla = FactorioInstance(filename)
    vanilla.compile()


def test_vanilla_full_run():
    vanilla_main()
