import json
import collections
import functools
from functools import lru_cache
import re
import numpy as np
import pycosat
import copy
import logging
from sparsetensors import *
import scipy.optimize

RELEVENT_FLUID_TEMPERATURES = {} #keys are fluid names, values are a dict with keys of temperature and values of energy density
COST_MODE = 'normal' #can be set to 'expensive' for the other recipes

def add_dicts(d1, d2):
    """
    Adds two dictionaries containing numerical values together. Combines them and where there is overlap
    in keys uses __add__ on d1's element passing d2's element.
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
    """
    dn = {}
    for k, v in d.items():
        dn.update({k: m*v})
    return dn

def dnf_addition(vals1, vals2):
    """
    'Adds' two disjunctive normal forms together, representing an AND operation between them.
    """
    vals3 = []
    for a1 in vals1:
        for a2 in vals2:
            tot = []
            for v1 in a1:
                tot.append(v1)
            for v2 in a2:
                if v2 not in tot:
                    tot.append(v2)
            vals3.append(tot)
    return vals3
    
def numericalize_standard_forms(*std_forms):
    """
    Replaces objects in a standard logical form with a reference number so that pycosat can be used.
    """
    universal_reference = {}
    i = 1
    for form in std_forms:
        for junc in form:
            for e in junc:
                if e['name'] not in universal_reference.keys():
                    universal_reference.update({e['name']: i})
                    i += 1
    return tuple([[[universal_reference[e['name']] for e in junc] for junc in form] for form in std_forms])
    
def neg_standard_form(std_form):
    """
    Calculates the negation of a standard form.
    Returns the opposite form, so if given a DNF it returns a CNF and vise versa.
    """
    return [[-1*e for e in junc] for junc in std_form]
    
def dnf_to_cnf(dnf):
    """
    Calculates the conjunctive normal form of a disjunctive normal form
    """
    if len(dnf)==0:
        return []
    elif len(dnf)==1:
        return [[e] for e in dnf[0]]
    return [item for sub in [[[e]+disj for disj in dnf_to_cnf(dnf[1:])] for e in dnf[0]] for item in sub]

class TechnologicalLimitation:
    """
    Shortened to a 'limit' elsewhere in the program. Represents the technologies that must be researched in order
    to unlock the specific object (recipe, machine, module, etc.)
    """
    def __init__(self, dnf):
        if len(dnf)==0:
            self.dnf = [[]]
        else:
            self.dnf = dnf
        self.cnf = dnf_to_cnf(dnf)
        
    def __repr__(self):
        return ", or\n\t".join([" and ".join([tech['name'] for tech in disj]) for disj in self.dnf])
        
    def __add__(self, other):
        return TechnologicalLimitation(dnf_addition(self.dnf, other.dnf))
        
    def __radd__(self, other):
        return TechnologicalLimitation(dnf_addition(self.dnf, other.dnf))
        
    def __lt__(self, other):
        return self <= other and not other <= self
        
    def __le__(self, other):
        """
        Return True iff other->self is tautological. Doone via boolean SAT of negation of other->self 
        will return true if other->self is non-tautological.
        
        Napkin logic:
        -(other->self)
        -(-other v self)
        (other ^ -self)
        (other.cnf + "-"self.dnf)
        """
        other_num_form, self_num_form = numericalize_standard_forms(other.cnf, self.dnf)
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

def listunion(l1, l2):
    """
    Union operation between lists.
    """
    raise DeprecationWarning
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
    """
    try:
        value = float(re.findall(r'[\d.]+', string)[0])
        value*= power_letters[re.findall(r'[k,K,M,G,T,P,E,Z,Y]', string)[0]]
        return value
    except:
        raise ValueError(string)

def add_complex_type(energy_source, data):
    """
    Adds a complex_type to energy sources, complex_types are the specific type of energy, 
    for example there is one electrical type but many burner types for each burning material
    """
    try:
        if energy_source['type']=='electric':
            energy_source['complex_type'] = ['electric']
        elif energy_source['type']=='burner':
            if 'fuel_categories' in energy_source.keys():
                energy_source['complex_type'] = [('burner-'+x) for x in energy_source['fuel_categories']]
            elif 'fuel_category' in energy_source.keys():
                energy_source['complex_type'] = ['burner-'+energy_source['fuel_category']]
            else:
                raise ValueError("Maybe all fuel categories?")
        elif energy_source['type']=='heat':
            energy_source['complex_type'] = ['heat']
        elif energy_source['type']=='fluid':
            if 'maximum_temperature' in energy_source.keys():
                if 'filter' in energy_source['fluid_box'].keys():
                    energy_source['complex_type'] = [energy_source['fluid_box']['filter']+'@'+energy_source['maximum_temperature']]
                else:
                    raise ValueError("Wtf is this fuel requirement?")
                    #energy_source['complex_type'] = ['fluid@'+energy_source['maximum_temperature']]
            else:
                if 'filter' in energy_source['fluid_box'].keys():
                    energy_source['complex_type'] = [energy_source['fluid_box']['filter']]
                else:
                    energy_source['complex_type'] = ['fluid']
        elif energy_source['type']=='void':
            raise ValueError("wtf void?")
        else:
            raise ValueError(energy_source)
    except:
        raise ValueError(energy_source)

def dereference_complex_type(complex_types, data): 
    """
    Given a set of complex types returns a list of tuples containing specific fuel names and their energy density.
    """
#gives a list of allowed (fuel name, fuel 'density' value (J/unit fuel)) for a complex type
    if len(complex_types)==0:
        return []
    elif complex_types[0] in ['electric', 'heat', 'void']:
        return [(complex_types[0], 1)]
    elif 'burner-' in complex_types[0]:
        allowed_categories = [ct.split("-", 1)[1] for ct in complex_types]
        allowed_fuels = []
        for item in data['item'].values():
            if 'fuel_category' in item.keys() and item['fuel_category'] in allowed_categories:
                allowed_fuels.append((item['name'], item['fuel_value_raw']))
        return allowed_fuels
    elif '@' in complex_types[0]:
        fluid, temp = complex_types[0].split("@")
        return [(fluid, RELEVENT_FLUID_TEMPERATURES[fluid][temp])]
    elif complex_types[0] in ['fluid']:
        return list(filter(lambda x: x, [(fluid['name'], fluid['fuel_value_raw']) if 'fuel_value' in fluid.keys() else False for fluid in data['fluid'].values()]))
    else:
        raise ValueError(complex_types)

def link_techs(data):
    logging.debug("Beginning the linking of technologies. This first step adds several lists to each technology.\n\t\'prereq_set\' is a list of all technologies that are the technology's prerequisite.\n\t\'prereq_of\' is a list of all technologies that have the technology as a prerequisite.\n\t\'all_prereq\' is a list of all technologies that are required to be unlocked (including the technology itself) to complete the research of the technology.\n\t\'enabled_recipes\' is a list of recipes that the technology enables.")
    for tech in data['technology'].values():
        if not 'enabled' in tech.keys() or tech['enabled']:
            tech['prereq_set'] = []
            tech['prereq_of'] = []
            tech['all_prereq'] = []
            tech['enabled_recipes'] = []

    for tech in data['technology'].values():
        if not 'enabled' in tech.keys() or tech['enabled']:
            if 'prerequisites' in tech.keys():
                for prereq in tech['prerequisites']:
                    for otech in data['technology'].values():
                        if not 'enabled' in otech.keys() or otech['enabled']:
                            if otech['name']==prereq:
                                tech['prereq_set'].append(otech)
                                otech['prereq_of'].append(tech)
    
    def get_all_prereqs(t):
        s = [t]
        for st in t['prereq_set']:
            s.append(st)
            s = listunion(s,get_all_prereqs(st))
        return s

    for tech in data['technology'].values():
        if not 'enabled' in tech.keys() or tech['enabled']:
            tech['all_prereq'] = get_all_prereqs(tech)

    logging.debug("Beginning the linking of all recipes to their technologies.")
    for recipe in data['recipe'].values():
        bigbreak = False
        recipe_tech = []
        for tech in data['technology'].values():
            if 'effects' in tech.keys():
                for effect in tech['effects']:
                    if effect['type']=='unlock-recipe' and effect['recipe']==recipe['name']:
                        recipe_tech.append(tech['all_prereq'])
                        tech['enabled_recipes'].append(recipe)
        recipe.update({'limit': TechnologicalLimitation(recipe_tech)})

    logging.debug("Beginning the linking of all machines to their technologies.")
    #link all vector-izeable machines to their technology unlock
    for ref in ['assembling-machine', 'rocket-silo', 'boiler', 'burner-generator', 'furnace', 'generator', 'mining-drill', 'offshore-pump', 'reactor', 'solar-panel']:
        for machine in data[ref].values():
            if machine['name'] in data['recipe'].keys():
                machine['limit'] = data['recipe'][machine['name']]['limit'] #highjack the recipe's link
                
    logging.debug("Beginning the linking of all modules to their technologies.")
    #link all modules to their technology unlock
    for module in data['module'].values():
        if module['name'] in data['recipe'].keys():
            module['limit'] = data['recipe'][module['name']]['limit'] #highjack the recipe's link

def standardize_power(data):
    logging.debug("Beginning the standardization of power. This adds a raw version of many energy values.")
    
    for machine in data['assembling-machine'].values():
        machine['energy_usage_raw'] = convert_value_to_base_units(machine['energy_usage'])
        add_complex_type(machine['energy_source'], data)

    for machine in data['rocket-silo'].values():
        machine['energy_usage_raw'] = convert_value_to_base_units(machine['active_energy_usage'])+convert_value_to_base_units(machine['idle_energy_usage'])
        add_complex_type(machine['energy_source'], data)
    
    for machine in data['boiler'].values():
        machine['energy_consumption_raw'] = convert_value_to_base_units(machine['energy_consumption'])
        add_complex_type(machine['energy_source'], data)
    
    for machine in data['burner-generator'].values():
        machine['max_power_output_raw'] = convert_value_to_base_units(machine['max_power_output'])
        if not 'type' in machine['burner'].keys():
            machine['burner'].update({'type': 'burner'})
        add_complex_type(machine['burner'], data)
        add_complex_type(machine['energy_source'], data)
    
    for machine in data['furnace'].values():
        machine['energy_usage_raw'] = convert_value_to_base_units(machine['energy_usage'])
        add_complex_type(machine['energy_source'], data)
    
    for machine in data['generator'].values():
        if 'max_power_output' in machine.keys():
            machine['max_power_output_raw'] = convert_value_to_base_units(machine['max_power_output'])
        else:
            machine['max_power_output_raw'] = 60*machine['fluid_usage_per_tick']*\
                                              ((100-data['fluid']['water']['default_temperature'])*convert_value_to_base_units(data['fluid']['water']['heat_capacity'])+\
                                               (machine['maximum_temperature']-100)*convert_value_to_base_units(data['fluid']['steam']['heat_capacity']))
        add_complex_type(machine['energy_source'], data)
    
    for machine in data['mining-drill'].values():
        machine['energy_usage_raw'] = convert_value_to_base_units(machine['energy_usage'])
        add_complex_type(machine['energy_source'], data)
    
    for machine in data['reactor'].values():
        machine['consumption_raw'] = convert_value_to_base_units(machine['consumption'])
        add_complex_type(machine['energy_source'], data)
        if not 'type' in machine['heat_buffer'].keys():
            machine['heat_buffer'].update({'type': 'heat'})
        add_complex_type(machine['heat_buffer'], data)
    
    for machine in data['solar-panel'].values():
        machine['production_raw'] = convert_value_to_base_units(machine['production'])
        add_complex_type(machine['energy_source'], data)
    
    for item in data['item'].values():
        if 'fuel_value' in item.keys():
            item['fuel_value_raw'] = convert_value_to_base_units(item['fuel_value'])

    for item in data['fluid'].values():
        if 'fuel_value' in item.keys():
            item['fuel_value_raw'] = convert_value_to_base_units(item['fuel_value'])

def prep_resources(data):
    logging.debug("Doing some slight edits to resources and linking them to their categories.")
    for resource in data['resource'].values():
        if not 'category' in resource.keys():
            resource.update({'category': 'basic-solid'})

    for cata in data['resource-category'].values():
        cata.update({'resource_list': []})
        for resource in data['resource'].values():
            if cata['name'] == resource['category']:
                cata['resource_list'].append(resource)

def vectorize_recipes(data):
    logging.debug("Beginning the vectorization of recipes. This adds a \'base_inputs\' and \'vector\' component to each recipe. The \'base_inputs\' are values stored for catalyst cost calculations. While the \'vector\' component represents how the recipe acts.")
    for recipe in data['recipe'].values():
        changes = {}
        if COST_MODE in recipe.keys(): 
            recipe_dict = recipe[COST_MODE]
        else:
            recipe_dict = recipe
            
        for ingred in recipe_dict['ingredients']:
            if isinstance(ingred, dict):
                fixed_name = ingred['name']
                if 'minimum_temperature' in ingred.keys():
                    fixed_name += '@'+str(ingred['minimum_temperature'])+'-'+str(ingred['maximum_temperature'])
                changes.update({ingred['name']: -1*ingred['amount']})
            elif isinstance(ingred, list):
                changes.update({ingred[0]: -1*ingred[1]})
            else:
                raise ValueError(ingred)

        if 'results' in recipe_dict.keys():
            for result in recipe_dict['results']:
                if isinstance(result, dict):
                    fixed_name = result['name']
                    if 'temperature' in result.keys():
                        fixed_name += '@'+str(result['temperature'])
                        if not result['name'] in RELEVENT_FLUID_TEMPERATURES.keys():
                            RELEVENT_FLUID_TEMPERATURES.update({result['name']: {}})
                        if not result['temperature'] in RELEVENT_FLUID_TEMPERATURES[result['name']].keys():
                            RELEVENT_FLUID_TEMPERATURES[result['name']].append({result['temperature']: None})
                    if 'amount' in result.keys():
                        changes.update({fixed_name: result['amount']})
                    else:
                        changes.update({fixed_name: .5*(result['amount_max']+result['amount_min'])})
                    if 'probability' in result.keys():
                        changes[fixed_name] *= result['probability']
                elif isinstance(result, list):
                    changes.update({result[0]: result[1]})
                else:
                    raise ValueError(result)
        else:
            if 'result_count' in recipe_dict.keys():
                changes.update({recipe_dict['result']: recipe_dict['result_count']})
            else:
                changes.update({recipe_dict['result']: 1})

        energy = .5
        if 'energy_required' in recipe.keys():
            energy = recipe['energy_required']
        else:
            recipe.update({'energy_required': .5})#fight me

        base_inputs = {}
        for c, v in changes.items():
            if v < 0:
                base_inputs.update({c: v})
        recipe.update({'vector': multi_dict(1.0/energy, changes),
                       'base_inputs': base_inputs})

def link_modules(data):
    logging.debug("Starting the linking of modules into the recipe and resource lists. %d recipes are detected, %d resources are detected, and %d modules are detected. Info will be added to the each recipe and resource under the \'allowed_modules\' key.",
        len(data['recipe'].values()),
        len(data['resource'].values()),
        len(data['module'].values()))
    for recipe in data['recipe'].values():
        recipe.update({'allowed_modules': []})
    for resource in data['resource'].values():
        resource.update({'allowed_modules': []})
    for module in data['module'].values():
        if 'limitation' in module.keys():
            for recipe_name in module['limitation']:
                data['recipe'][recipe_name]['allowed_modules'].append(module)
        else:
            for recipe in data['recipe'].values():
                recipe['allowed_modules'].append(module)
            for resource in data['resource'].values():
                resource['allowed_modules'].append(module)

def complete_premanagement(data):
    logging.debug("Beginning the premanagement of the game \'data.raw\' object. This will link technologies, standardize power, recategorize resources, simplify recipes, and link modules.")
    link_techs(data)
    standardize_power(data)
    prep_resources(data)
    vectorize_recipes(data)
    link_modules(data)

def list_of_dicts_by_key(list_of_dicts, key, value):
    return filter(lambda d: key in d.keys() and d[key]==value, list_of_dicts)

def tech_objection_via_spec(tech_spec, data):
    assert isinstance(tech_spec, dict)
    for k in tech_spec.keys():
        assert k in ["fully_automated", "extra_technologies", "extra_recipes"]
    logging.info("Making a tech specification via dict.")
    
    tech_obj = TechnologicalLimitation([[]])
    
    if "fully_automated" in tech_spec.keys():
        for v in tech_spec["fully_automated"]:
            assert v in data['tool'].keys()
        for tech in data['technology'].values():
            if COST_MODE in tech.keys():
                unit = tech[COST_MODE]['unit']
            else:
                unit = tech['unit']
            if all([ing_name in tech_spec["fully_automated"] for ing_name in [ingred['name'] if isinstance(ingred, dict) else ingred[0] for ingred in unit['ingredients']]]):
                tech_obj = tech_obj + TechnologicalLimitation([[tech]])
    
    if "extra_technologies" in tech_spec.keys():
        for v in tech_spec["extra_technologies"]:
            assert v in data['technology'].keys()
            tech_obj = tech_obj + data['technology']
        raise NotImplemented
    
    if "extra_recipes" in tech_spec.keys():
        for v in tech_spec["extra_recipes"]:
            assert v in data['recipe'].keys()
            tech_obj = tech_obj + data['recipe'][v]['limit']
        raise NotImplemented
    
    logging.info("Tech specification: "+str(tech_obj))
    return tech_obj

def linprog_wrapper_bad_edition(x0, A_ub, b_ub, A_eq, b_eq, bounds, method, debug_info=(None, None, None)):
    #does all the stuff scipy.optimize.linprog should already do
    #remove irrelevent rows such as subsumed inequality constraints and rows of pure 0s
    #also does presolving steps
    assert A_ub.shape[1]==x0.shape[0], "A_ub is mishapen compared to x0"
    assert A_eq.shape[1]==x0.shape[0], "A_eq is mishapen compared to x0"
    assert A_ub.shape[0]==b_ub.shape[0], "b_ub is mishapen compared to A_ub"
    assert A_eq.shape[0]==b_eq.shape[0], "b_eq is mishapen compared to A_eq"
    assert isinstance(bounds, tuple), "Sorry its hardcoded until it doesn't need to be"
    assert bounds[0]==None and bounds[1]==None, "Sorry its hardcoded until it doesn't need to be"

    pre_sol = np.empty(x0.shape[0], dtype=x0.dtype)
    pre_sol[:] = np.nan
    lb, ub = -np.inf * np.ones(n), np.inf * np.ones(n)

    #Presolving step from: https://github.com/scipy/scipy/blob/main/scipy/optimize/_linprog_util.py
    last_count = 0
    singleton_row = np.array(np.sum(A_eq != 0, axis=1) == 1).flatten()
    while last_count != np.count_nonzero(singleton_row):
        last_count = np.count_nonzero(singleton_row)

        #find singleton rows 
        s_rows = np.where(singleton_row)[0]
        s_cols = np.where(A_eq[s_rows, :])[1]
        if len(s_rows) > 0:
            for row, col in zip(s_rows, s_cols):
                val = b_eq[row] / A_eq[row, col]
                pre_sol[col] = val
                lb[col] = val
                ub[col] = val

        singleton_row = np.array(np.sum(A_eq != 0, axis=1) == 1).flatten()
    
    empty_row_eq = np.array(np.logical_and(np.sum(A_eq != 0, axis=1) == 0, b_eq == 0)).flatten()
    empty_row_ub = np.array(np.logical_and(np.sum(A_ub != 0, axis=1) == 0, b_ub >= 0)).flatten()

def linprog_wrapper(x0, A_ub, b_ub, A_eq, b_eq, bounds, method):
    assert A_ub.shape[1]==x0.shape[0], "A_ub is mishapen compared to x0"
    assert A_eq.shape[1]==x0.shape[0], "A_eq is mishapen compared to x0"
    assert A_ub.shape[0]==b_ub.shape[0], "b_ub is mishapen compared to A_ub"
    assert A_eq.shape[0]==b_eq.shape[0], "b_eq is mishapen compared to A_eq"

    res = scipy.optimize.linprog(x0, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                                 bounds=bounds, method=method)
    
    if res.success:
        return res
    else:
        target_status = res.status
        mi = 0
        while mi < 10000:
            mi += 1

            i = 0
            while i < b_ub.shape[0]:
                nA_ub = np.delete(A_ub, i, axis=0)
                nb_ub = np.delete(b_ub, i, axis=0)
                nres = scipy.optimize.linprog(x0, A_ub=nA_ub, b_ub=nb_ub, A_eq=A_eq, b_eq=b_eq,
                                              bounds=bounds, method=method)
                if nres.status == target_status:
                    break
                i += 1
            if i != b_ub.shape[0]:
                A_ub = np.delete(A_ub, i, axis=0)
                b_ub = np.delete(b_ub, i, axis=0)
                continue

            i = 0
            while i < b_eq.shape[0]:
                nA_eq = np.delete(A_eq, i, axis=0)
                nb_eq = np.delete(b_eq, i, axis=0)
                nres = scipy.optimize.linprog(x0, A_ub=A_ub, b_ub=b_ub, A_eq=nA_eq, b_eq=nb_eq,
                                              bounds=bounds, method=method)
                if nres.status == target_status:
                    break
                i += 1
            if i != b_eq.shape[0]:
                A_eq = np.delete(A_eq, i, axis=0)
                b_eq = np.delete(b_eq, i, axis=0)
                continue
            
            break
        assert mi != 10000, "Uhh, some infinite loop issue"

        print("Found minimal breaking set containing "+str(b_ub.shape[0])+" inequality constraints and "+str(b_eq.shape[0])+" equality constraints. Saving Them.")
        np.savetxt("A_ub_minimal.txt", A_ub, delimiter='\t')
        np.savetxt("b_ub_minimal.txt", b_ub, delimiter='\t')
        np.savetxt("A_eq_minimal.txt", A_eq, delimiter='\t')
        np.savetxt("b_eq_minimal.txt", b_eq, delimiter='\t')

        assert res.success, res.message

def zero_out_via_constraints(tensor, *constraints):
    #This function zeros out values in a tensor that dont need to be non zero.
    #It decides this by comparing constraint violation in the non-zero and zero cases.
    violations = []
    for constraint in constraints:
        viol = constraint(tensor)
        assert np.isscalar(viol), "Constraints must return singular values, not arrays."
        violations.append(viol)
    total_violation = np.sum(violations)

    for indicies in zip(*tensor.nonzero()):
        ntensor = tensor.copy()
        ntensor[indicies] = 0
        new_violations = []
        for constraint in constraints:
            new_violations.append(constraint(ntensor))
        new_total_violation = np.sum(new_violations)
        if np.isclose(total_violation, new_total_violation) or new_total_violation <= total_violation:
            tensor = ntensor
    
    return tensor
        
