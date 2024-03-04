from __future__ import annotations

import logging
from fractions import Fraction
import numexpr
import re

from globalvalues import *

from typing import TypeVar, Callable, Hashable, Iterable, Any
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
    
    def __rmul__(self, multiplier: Fraction) -> CompressedVector:
        return self.__mul__(multiplier)
    
    def key_addition(self, key, value) -> None:
        """
        Single line adding key or adding to value in key
        """
        if key in self.keys():
            self[key] += value
        else:
            self[key] = value

def count_via_lambda(l: list[T] | dict[Any, T], func: Callable[[T], bool] = lambda x: True) -> int:
    """
    Counting helper for lists. Counts how many elements in the list return true when passed to function func.

    Parameters
    ----------
    l:
        The list.
    func:
        A function with 1 list element as input.
    
    Returns
    -------
    Number of instances where func returned Truthy on list elements.
    """
    if isinstance(l, list):
        return sum([1 if func(e) else 0 for e in l])
    else: #isinstance(l, dict):
        return sum([1 if func(e) else 0 for e in l.values()])

class TechnologicalLimitation:
    """
    Shortened to a 'limit' elsewhere in the program. Represents the technologies that must be researched in order
    to unlock the specific object (recipe, machine, module, etc.).

    Members
    -------
    canonical_form:
        List of sets making up the canonical form.
    """
    canonical_form: frozenset[frozenset[str]]

    def __init__(self, list_of_nodes: Iterable[Iterable[str]] = []) -> None:
        """
        Parameters
        ----------
        list_of_nodes:
            Iterable of iterable over nodes to make up this limitation.
        """
        self.canonical_form = set()
        for nodes in list_of_nodes:
            self.canonical_form.add(frozenset(nodes))
        self.canonical_form = frozenset(self.canonical_form)
        
    def __repr__(self) -> str:
        return str(self.canonical_form)
        #return ", or\n\t".join([" and ".join(nodes) for nodes in self.canonical_form])

    def __add__(self, other: TechnologicalLimitation) -> TechnologicalLimitation:
        """
        Addition betwen two TechnologicalLimitations is an AND operation.
        """
        if len(self.canonical_form)==0:
            return other
        if len(other.canonical_form)==0:
            return self
        added = set()
        for s1 in self.canonical_form:
            for s2 in other.canonical_form:
                added.add(s1.union(s2))
        return TechnologicalLimitation(added)

    def __sub__(self, other: Iterable[str]) -> list[frozenset[str]]:
        """
        Returns a list of possible additions to a set of researches that will result in the completion of limitation.
        """
        return [frozenset(nodes.difference(other)) for nodes in self.canonical_form]
        
    #def __eq__(self, other: TechnologicalLimitation) -> bool:
    #    return self.canonical_form == other.canonical_form
        
    #def __ne__(self, other: TechnologicalLimitation) -> bool:
    #    return self.canonical_form != other.canonical_form

    #def __lt__(self, other: TechnologicalLimitation) -> bool:
    #    return self <= other and self != other
        
    #def __le__(self, other: TechnologicalLimitation) -> bool:
    #    return other >= self
        
    #def __gt__(self, other: TechnologicalLimitation) -> bool:
    #    return self >= other and self != other
        
    def __ge__(self, other: TechnologicalLimitation) -> bool:
        if len(other.canonical_form)==0 and len(self.canonical_form)!=0:
            return True
        if len(self.canonical_form)==0 and len(other.canonical_form)!=0:
            return False
        for nodes_self in self.canonical_form:
            if not any([nodes_self >= nodes_other for nodes_other in other.canonical_form]):
                return False
        return True

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
        value = Fraction(re.findall(r'[\d.]+', string)[0]).limit_denominator()
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

def technological_limitation_from_specification(data: dict, fully_automated: list[str] = [], extra_technologies: list[str] = [], extra_recipes: list[str] = [],
                                                COST_MODE: str = 'normal') -> TechnologicalLimitation:
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
    tech_obj = set()
    
    assert len(fully_automated)+len(extra_technologies)+len(extra_recipes) > 0, "??"
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
        if all([tool_name in fully_automated for tool_name in [ingred['name'] for ingred in unit['ingredients']]]):
            tech_obj.add(tech['name'])
    
    for tech in extra_technologies:
        tech_obj = tech_obj.union(next(iter(data['technology'][tech]['limit'].canonical_form)))
        tech_obj.add(tech)
    
    for recipe in extra_recipes:
        tech_obj = tech_obj.union(data['recipe'][recipe]['limit'])
    
    return TechnologicalLimitation([tech_obj])

def evaluate_formulaic_count(expression: str, level: int) -> int:
    """
    Evaluates the count for a formulaic expression for a level.

    Parameters
    ----------
    expression:
        Expression for the formulaic count. https://lua-api.factorio.com/latest/types/TechnologyUnit.html#count_formula
    level:
        What level to calculate at.
    
    Returns
    -------
    Count value.
    """
    fixed_expression = expression.replace("l", str(level)).replace("L", str(level)).replace("^", "**")
    fixed_expression = re.sub(r'(\d)\(', r'\1*(', fixed_expression)
    logging.warning(fixed_expression)
    return numexpr.evaluate(fixed_expression).item() #TODO: is this line as safe as i hope?