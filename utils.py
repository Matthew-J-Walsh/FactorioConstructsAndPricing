
import re
import pycosat
import logging
from sparsetensors import *
from globalvalues import *

def add_dicts(d1, d2):
    """
    Adds two dictionaries containing numerical values together. Combines them and where there is overlap
    in keys uses __add__ on d1's element passing d2's element.

    Parameters
    ----------
    d1:
        First dictionary
    d2:
        First dictionary
    
    Returns
    -------
    d3:
        Combined dictionary
    """
    d3 = {}
    for k, v in d1.items():
        d3.update({k: v})
    for k, v in d2.items():
        if k in d3.keys():
            d3[k] = d3[k] + v
        else:
            d3.update({k: v})
    return d3

def multi_dict(m, d):
    """
    Multiplies all values of a dictionary by a number.

    Parameters
    ----------
    m:
        Multiplier
    d:
        Dictionary
    
    Returns
    -------
    dn:
        Copy of d with every value multiplied by m
    """
    dn = {}
    for k, v in d.items():
        dn.update({k: m*v})
    return dn

def count_via_lambda(l, func=lambda x: True):
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
    count:
        Number of instances where func returned Truthy on list elements
    """
    return sum([1 if func(e) else 0 for e in l])

def dnf_and(dnf1, dnf2):
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
    
def numericalize_standard_expressions(*std_forms, reference=lambda x: x['name']):
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
    
def neg_standard_form(std_form):
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
    
def dnf_to_cnf(dnf):
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

def simplified_dnf(dnf, reference=lambda x: x['name']):
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
    logging.warning("simplified_dnf currently doesnt simplify. its a TODO")
    return dnf

class TechnologicalLimitation:
    """
    Shortened to a 'limit' elsewhere in the program. Represents the technologies that must be researched in order
    to unlock the specific object (recipe, machine, module, etc.)

    Parameters
    ----------
    dnf:
        Logical expression in disjunctive normal form. Literals should be dicts with at least a 'name' key. Only positive literals are supported (and needed).
    """
    def __init__(self, dnf):
        if len(dnf)==0:
            self.dnf = [[]]
        else:
            self.dnf = simplified_dnf(dnf)
        self.cnf = dnf_to_cnf(dnf)
        
    def __repr__(self):
        return ", or\n\t".join([" and ".join([tech['name'] for tech in disj]) for disj in self.dnf])
        
    def __add__(self, other):
        """
        Addition betwen two TechnologicalLimitations is an AND operation.
        """
        return TechnologicalLimitation(dnf_and(self.dnf, other.dnf))
        
    def __radd__(self, other):
        return TechnologicalLimitation(dnf_and(self.dnf, other.dnf))
        
    def __lt__(self, other):
        return self <= other and not other <= self
        
    def __le__(self, other):
        """
        Return True iff other->self is tautological. Done via boolean SAT of negation of other->self 
        will return true if other->self is non-tautological.
        
        Napkin logic:
        -(other->self)
        -(-other v self)
        (other ^ -self)
        """
        other_num_form, self_num_form = numericalize_standard_expressions(other.cnf, self.dnf)
        problem = other_num_form+neg_standard_form(self_num_form)
        for sol in pycosat.itersolve(problem):
            return False
        return True
        
    def __eq__(self, other):
        return self <= other and other <= self
        
    def __ne__(self, other):
        return NotImplemented
        
    def __gt__(self, other):
        return other < self
        
    def __ge__(self, other):
        return other <= self

def list_union(l1, l2):
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
    l3:
        Union of list.
    """
    l3 = []
    for v in l1:
        l3.append(v)
    for v in l2:
        if not v in l3:
            l3.append(v)
    return l3

power_letters = {'k': 10**3, 'K': 10**3, 'M': 10**6, 'G': 10**9, 'T': 10**12, 'P': 10**15, 'E': 10**18, 'Z': 10**21, 'Y': 10**24}
def convert_value_to_base_units(string):
    """
    Converts various values (power, energy, etc.) into their expanded form.

    Parameters
    ----------
    string:
        String in some sort of Factorio base unit.
    
    Returns
    -------
    value:
        Value in integer form.
    """
    try:
        value = float(re.findall(r'[\d.]+', string)[0])
        value*= power_letters[re.findall(r'[k,K,M,G,T,P,E,Z,Y]', string)[0]]
        return int(value)
    except:
        raise ValueError(string)

def list_of_dicts_by_key(list_of_dicts, key, value):
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

def technological_limitation_from_specification(data, fully_automated=[], extra_technologies=[], extra_recipes=[]):
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
    Filtered list.
    """
    logging.info("Making a tech specification via dict.")
    
    tech_obj = TechnologicalLimitation([[]])
    
    for pack in fully_automated:
        assert pack in data['tool'].keys()
    for tech in extra_technologies:
        assert tech in data['technology'].keys()
    for recipe in extra_recipes:
        assert recipe in data['recipe'].keys()


    for tech in data['technology'].values():
        if COST_MODE in tech.keys():
            unit = tech[COST_MODE]['unit']
        else:
            unit = tech['unit']
        if all([ing_name in fully_automated for ing_name in [ingred['name'] if isinstance(ingred, dict) else ingred[0] for ingred in unit['ingredients']]]):
            tech_obj = tech_obj + TechnologicalLimitation([[tech]])
    
    for tech in extra_technologies:
        tech_obj = tech_obj + data['technology'][tech]['limit']
    
    for recipe in extra_recipes:
        tech_obj = tech_obj + data['recipe'][recipe]['limit']
    
    logging.info("Tech specification generated: "+str(tech_obj))
    return tech_obj

