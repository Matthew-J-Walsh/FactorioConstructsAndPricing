from utils import *
import pytest

def add_dicts_test_1():
    d1 = {0: 1, 1: 3, 2: 7}
    d2 = {1: -1, 2: 3, 5: 2}
    assert add_dicts(d1, d2)=={5: 2, 1: 2, 2: 10, 0: 1}


def multi_dict_test_1():
    d = {0: 1, 1: 3, 2: 7}
    m = 3
    assert multi_dict(m, d)=={0: 3, 1: 9, 2: 21}


def dnf_addition_test_1():
    form1 = [["A", "B"], ["A", "C"], ["-C", "D"]]
    form2 = [["-B"], ["B", "C", "-D"], ["E", "F", "-G"]]
    added = [["A", "B", "-B"], ["A", "B", "C", "-D"], ["A", "B", "E", "F", "-G"],
             ["A", "C", "-B"], ["A", "C", "B", "-D"], ["A", "C", "E", "F", "-G"],
             ["-C", "D", "-B"], ["-C", "D", "B", "C", "-D"], ["-C", "D", "E", "F", "-G"]]
    via_func = dnf_and(form1, form2)

    added_set = set()
    for form in added:
        added_set.add(frozenset(form))
    via_func_set = set()
    for form in via_func:
        via_func_set.add(frozenset(form))
    assert added_set==via_func_set


def numericalize_standard_forms_test_1(): #placeholder test, not sure how to test this one
    assert True 


def neg_standard_form_test_1():
    form1 = [["A", "B"], ["A", "C"], ["-C", "D"]]
    form2 = [["-B"], ["B", "C", "-D"], ["E", "F", "-G"]]
    form1, form2 = numericalize_standard_expressions(form1, form2)
    negative_form1 = neg_standard_form(form1)
    assert True


def dnf_to_cnf_test_1():
    dnf = [["A", "B"], ["A", "C"], ["-C", "D"]]
    cnf = [["A", "-C"], ["A", "D"], ["A", "C", "-C"], ["A", "C", "D"],
           ["B", "A", "-C"], ["B", "A", "D"], ["B", "C", "-C"], ["B", "C", "D"]]
    via_func = dnf_to_cnf(dnf)
    
    cnf_set = set()
    for form in cnf:
        cnf_set.add(frozenset(form))
    via_func_set = set()
    for form in via_func:
        via_func_set.add(frozenset(form))
    assert cnf_set==via_func_set


def tech_limit_testing_techs():
    return [{'name': l} for l in "A,B,C,D,E,F,G,H,I,J,K,L".split(",")]

@pytest.fixture
def tech_limit_testing_limits():
    all_techs = tech_limit_testing_techs()
    base_limits = [TechnologicalLimitation([[tech]]) for tech in all_techs]
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






