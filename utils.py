
from __future__ import annotations
import re
import pycosat
import logging
import numpy as np
import scipy as sp
import scipy.sparse
from fractions import Fraction
from globalvalues import *
from typing import TypeVar, Callable, Any
T = TypeVar('T')

class CompressedVector(dict):
    """
    CompressedVector's are dicts where the values are all numbers and the values represent a dimension.
    """
    def __add__(self, other: CompressedVector) -> CompressedVector:
        d3 = CompressedVector()
        for k, v in self.items():
            d3.update({k: v})
        for k, v in other.items():
            if k in d3.keys():
                d3[k] = d3[k] + v
            else:
                d3.update({k: v})
        return d3

    def __mul__(self, multiplier: Fraction) -> CompressedVector:
        dn = CompressedVector()
        for k, v in self.items():
            dn.update({k: multiplier * v})
        return dn

def count_via_lambda(l: list[T], func: Callable[[T], bool] = lambda x: True) -> int:
    """
    Counting helper for lists. Counts how many elements in the list return true from func.

    Parameters
    ----------
    l:
        A list
    func:
        A function with 1 list element as input.
    
    Returns
    -------
    Number of instances where func returned Truthy on list elements
    """
    return sum([1 if func(e) else 0 for e in l])

def dnf_and(dnf1: list[list[T]], dnf2: list[list[T]]) -> list[list[T]]:
    """
    And operation betwen two dnf expressions.

    Parameters
    ----------
    dnf1:
        First disjunctive normal form expression
    dnf2:
        Second disjunctive normal form expression
    
    Returns
    -------
    dnf1p2:
        Disjunctive normal form of the result
    """
    dnf1p2 = []
    for a1 in dnf1:
        for a2 in dnf2:
            tot = []
            for v1 in a1:
                tot.append(v1)
            for v2 in a2:
                if v2 not in tot:
                    tot.append(v2)
            dnf1p2.append(tot)
    return dnf1p2
    
def numericalize_standard_expressions(*std_forms: list[list[T]], reference: Callable[[T], Any] = lambda x: x['name']) -> tuple[list[list[T]]]:
    """
    Replaces dicts in a standard logical expression with a reference number so that pycosat can be used. Each literal should return a unique result with the reference function
    Currently only support positive literals.

    Parameters
    ----------
    *std_forms:
        DNF or CNF (both are lists of lists of literals) logical expressions.
    reference
        Function that returns a unique value for every literal. This is usually used on dicts with a 'name' key, so that is chosen as default.
    
    Returns
    -------
    Tuple where each element is the transformed respective standard logical expression.
    """
    universal_reference = {}
    i = 1
    for form in std_forms:
        for junc in form:
            for e in junc:
                if reference(e) not in universal_reference.keys():
                    universal_reference.update({reference(e): i})
                    i += 1
    return tuple([[[universal_reference[reference(e)] for e in junc] for junc in form] for form in std_forms])
    
def neg_standard_form(std_form: list[list[T]]) -> list[list[T]]:
    """
    Calculates the negation of a standard form with integral literals.
    Returns the opposite form, so if given a DNF it returns a CNF and vise versa.

    Parameters
    ----------
    std_form:
        DNF or CNF logical expressions.
    
    Returns
    -------
    CNF or DNF logical expression that is the negative of std_form
    """
    return [[-1*e for e in junc] for junc in std_form]
    
def dnf_to_cnf(dnf: list[list[T]]) -> list[list[T]]:
    """
    Calculates the conjunctive normal form of a disjunctive normal form.

    Parameters
    ----------
    dnf:
        Logical expression in disjunctive normal form.
    
    Returns
    -------
    cnf:
        Logical expression in conjunctive normal form.
    """
    if len(dnf)==0:
        return []
    elif len(dnf)==1:
        return [[e] for e in dnf[0]]
    return [item for sub in [[[e]+disj for disj in dnf_to_cnf(dnf[1:])] for e in dnf[0]] for item in sub]

def simplify_dnf_helper(dnf: list[list[int]]) -> list[list[int]]:
    """
    Helper function for dnf simplification that only works on hashable types

    Parameters
    ----------
    dnf:
        Logical expression in disjunctive normal form with integral terms.
    
    Returns
    -------
    Logical expression in disjunctive normal in simplified form.
    """
    logging.warning("simplify_dnf_helper not implemented yet")
    return dnf
    dnf = [list(set(junc)) for junc in dnf]


    changed = True
    while changed:
        changed = False
        
        i = 0
        while i < len(dnf):
            reduced_dnf = copy.deepcopy(dnf)
            del reduced_dnf[i]
            if not isinstance(pycosat.solve(reduced_dnf), str):
                
                break
            i += 1
        
def simplified_dnf(dnf: list[list[T]], reference: Callable[[T], Any] = lambda x: x['name']) -> list[list[T]]:
    """
    Simplifies a logical expression in disjunctive normal form.

    Parameters
    ----------
    dnf:
        Logical expression in disjunctive normal form.
    reference
        Function that returns a unique value for every literal. This is usually used on dicts with a 'name' key, so that is chosen as default.
    
    Returns
    -------
    Logical expression in disjunctive normal in simplified form.
    """
    numerical_reference = {}
    numerical_recall = {}
    i = 1
    for junc in dnf:
        for e in junc:
            if reference(e) not in numerical_reference.keys():
                numerical_reference.update({reference(e): i})
                numerical_recall.update({i: e})
                i += 1

    simplified = simplify_dnf_helper([[numerical_reference[reference(e)] for e in junc] for junc in dnf])

    return [[numerical_recall(i) for i in junc] for junc in simplified]

class TechnologicalLimitation:
    """
    Shortened to a 'limit' elsewhere in the program. Represents the technologies that must be researched in order
    to unlock the specific object (recipe, machine, module, etc.)

    Members
    -------
    dnf:
        Disjunctive normal form of the limitation
    cnf:
        Conjunctive normal form of the limitation
    """
    dnf: list[list[T]]
    cnf: list[list[T]]

    def __init__(self, dnf: list[list[T]]) -> None:
        """
        Parameters
        ----------
        dnf:
            Logical expression in disjunctive normal form. Literals should be dicts with at least a 'name' key. Only positive literals are supported (and needed).
        """
        if len(dnf)==0:
            self.dnf = [[]]
        else:
            self.dnf = simplified_dnf(dnf)
        self.cnf = dnf_to_cnf(dnf)
        
    def __repr__(self) -> str:
        return ", or\n\t".join([" and ".join([tech['name'] for tech in disj]) for disj in self.dnf])
        
    def __add__(self, other: TechnologicalLimitation) -> TechnologicalLimitation:
        """
        Addition betwen two TechnologicalLimitations is an AND operation.
        """
        return TechnologicalLimitation(dnf_and(self.dnf, other.dnf))
        
    def __radd__(self, other: TechnologicalLimitation) -> TechnologicalLimitation:
        return TechnologicalLimitation(dnf_and(self.dnf, other.dnf))
        
    def __lt__(self, other: TechnologicalLimitation) -> bool:
        return self <= other and not other <= self
        
    def __le__(self, other: TechnologicalLimitation) -> bool:
        """
        Return True iff other->self is tautological. Done via boolean SAT of negation of other->self 
        will return true if other->self is non-tautological.
        
        Napkin logic:
        -(other->self)
        -(-other v self)
        (other ^ -self)
        """
        other_num_form, self_num_form = numericalize_standard_expressions(other.cnf, self.dnf)
        problem = other_num_form + neg_standard_form(self_num_form)
        for sol in pycosat.itersolve(problem):
            return False
        return True
        
    def __eq__(self, other: TechnologicalLimitation) -> bool:
        return self <= other and other <= self
        
    def __ne__(self, other: TechnologicalLimitation) -> bool:
        return NotImplemented
        
    def __gt__(self, other: TechnologicalLimitation) -> bool:
        return other < self
        
    def __ge__(self, other: TechnologicalLimitation) -> bool:
        return other <= self

def list_union(l1: list[T], l2: list[T]) -> list[T]:
    """
    Union operation between lists. Used for when we cant freeze dicts into a set.

    Parameters
    ----------
    l1:
        First list.
    l2:
        Second list.
    
    Returns
    -------
    Union of lists.
    """
    l3 = []
    for v in l1:
        l3.append(v)
    for v in l2:
        if not v in l3:
            l3.append(v)
    return l3

power_letters = {'k': 10**3, 'K': 10**3, 'M': 10**6, 'G': 10**9, 'T': 10**12, 'P': 10**15, 'E': 10**18, 'Z': 10**21, 'Y': 10**24}
def convert_value_to_base_units(string: str) -> Fraction:
    """
    Converts various values (power, energy, etc.) into their expanded form.

    Parameters
    ----------
    string:
        String in some sort of Factorio base unit.
    
    Returns
    -------
    Value in Fraction form.
    """
    try:
        value = Fraction(re.findall(r'[\d.]+', string)[0])
        value*= power_letters[re.findall(r'[k,K,M,G,T,P,E,Z,Y]', string)[0]]
        return value
    except:
        raise ValueError(string)

def list_of_dicts_by_key(list_of_dicts: list[dict], key, value) -> list[dict]:
    """
    Filters a list of dictonaries for only dictionaries with the specified key and the specified value for that key.

    Parameters
    ----------
    list_of_dicts:
        List of dictionaries to filter.
    key:
        Key to filter for.
    value:
        Value key must be.
    
    Returns
    -------
    Filtered list.
    """
    return filter(lambda d: key in d.keys() and d[key]==value, list_of_dicts)

def technological_limitation_from_specification(data: dict, fully_automated: list[str] = [], extra_technologies: list[str] = [], extra_recipes: list[str] = []) -> TechnologicalLimitation:
    """
    Generates a TechnologicalLimitation from a specification. Works as a more 'user' friendly way of getting useful TechnologicalLimitations.

    Parameters
    ----------
    data:
        Entire data.raw. https://wiki.factorio.com/Data.raw
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
    tech_obj = TechnologicalLimitation([[]])
    
    for pack in fully_automated:
        assert pack in data['tool'].keys() #https://lua-api.factorio.com/latest/prototypes/ToolPrototype.html
    for tech in extra_technologies:
        assert tech in data['technology'].keys() #https://lua-api.factorio.com/latest/prototypes/TechnologyPrototype.html
    for recipe in extra_recipes:
        assert recipe in data['recipe'].keys() #https://lua-api.factorio.com/latest/prototypes/RecipeCategory.html


    for tech in data['technology'].values(): #https://lua-api.factorio.com/latest/prototypes/TechnologyPrototype.html
        if COST_MODE in tech.keys():
            unit = tech[COST_MODE]['unit']
        else:
            unit = tech['unit'] #https://lua-api.factorio.com/latest/prototypes/TechnologyPrototype.html#unit
        if all([ing_name in fully_automated for ing_name in [ingred['name'] if isinstance(ingred, dict) else ingred[0] for ingred in unit['ingredients']]]):
            tech_obj = tech_obj + TechnologicalLimitation([[tech]])
    
    for tech in extra_technologies:
        tech_obj = tech_obj + data['technology'][tech]['limit']
    
    for recipe in extra_recipes:
        tech_obj = tech_obj + data['recipe'][recipe]['limit']
    
    return tech_obj

