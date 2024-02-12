import re
import itertools
import logging
import copy
from utils import *
from constructs import *

def generate_fuel_vectors_depreciated(complex_type, data):
    """
    makes vectors for standardizaiton of fuel to its use
    """
    result = []
    if complex_type in ['electric', 'heat', 'void']:
        result = [] #there are no specific types for these energies

    elif 'burner'==complex_type.split('-')[0]:
        for fuel in data['item']:
            if 'fuel_category' in fuel.keys() and fuel['fuel_category']==complex_type.split('-')[1]:
                result.append({fuel['name']: -1, complex_type: 1})

    elif complex_type=='fluid':
        for fuel in data['fluid']:
            if 'fuel_value' in fuel.keys():
                result.append({fuel['name']: -1, 'fluid': 1})

    return result

def solid_ingredient_count(recipe):
    if COST_MODE in recipe.keys():
        count = 0
        for ingred in recipe[COST_MODE]['ingredients']:
            if type(ingred)==type([]) or ingred['type']=='solid':
                count += 1
        return count
    else:
        count = 0
        for ingred in recipe['ingredients']:
            if type(ingred)==type([]) or ingred['type']=='solid':
                count += 1
        return count

def liquid_ingredient_count(recipe):
    if COST_MODE in recipe.keys():
        count = 0
        for ingred in recipe[COST_MODE]['ingredients']:
            if type(ingred)==type({}) and ingred['type']=='fluid':
                count += 1
        return count
    else:
        count = 0
        for ingred in recipe['ingredients']:
            if type(ingred)==type({}) and ingred['type']=='fluid':
                count += 1
        return count

def valid_crafting_machine(machine, recipe):
    if ('category' in recipe.keys() and recipe['category'] in machine['crafting_categories']) or (not 'category' in recipe.keys() and 'crafting' in machine['crafting_categories']):
        if (not 'ingredient_count' in machine.keys() or machine['ingredient_count'] >= solid_ingredient_count(recipe)):
            if liquid_ingredient_count(recipe)==0:
                return True
            if not 'fluid_boxes' in machine.keys():
                return False
            else:
                boxes = None
                if type(machine['fluid_boxes'])==type({}):
                    boxes = machine['fluid_boxes'].values()
                else:
                    boxes = machine['fluid_boxes']
                if len([x for x in boxes if 
                               (type(x)==type({}) and 
                                ('production_type' in x.keys()) and 
                                x['production_type']=="input")])>=\
                    liquid_ingredient_count(recipe):
                    return True
    return False

def generate_crafting_construct(machine, recipe, data):
    """
    Gnerates UncompiledConstructs of a machine that does recipe crafting.
    """
    assert machine['type']=='assembling-machine' or machine['type']=='rocket-silo' or machine['type']=='furnace'
    logging.debug("Generating construct family for: %s in %s", recipe['name'], machine['name'])
        
    if 'base_productivity' in machine.keys() and machine['base_productivity']!=0:
        logging.error("Bad news, I haven't implemeneted base productivity yet. No constructs will be created from %s", machine['name'])
        return None
    
    for fuel_name, fuel_value in dereference_complex_type(machine['energy_source']['complex_type'], data):
        if any([k.split('@')[0] in RELEVENT_FLUID_TEMPERATURES for k in recipe['vector'].keys()]+[k.split('@')[0] in RELEVENT_FLUID_TEMPERATURES for k in recipe['base_inputs'].keys()]):
            for prod in itertools.product(*[[(k.split('@')[0], temp) for temp in RELEVENT_FLUID_TEMPERATURES[k.split('@')[0]].keys()] for k in recipe['vector'].keys() if k.split('@')[0] in RELEVENT_FLUID_TEMPERATURES.keys()]):
                temps = {}
                for t in prod:
                    temps.update({t[0]: t[1]})
                invalid_temp = False
                for k in list(recipe['vector'].keys())+list(recipe['base_inputs'].keys()):
                    if len(k.split('@'))>1:
                        if len((k.split('@')[1]).split('-'))>1:
                            min_temp, max_temp = (k.split('@')[1]).split('-')
                        else:
                            min_temp, max_temp = k.split('@')[1], k.split('@')[1]
                            
                        if temps[k.split('@')[0]] < min_temp or temps[k.split('@')[0]] > max_temp:
                            invalid_temp = True
                        
                if not invalid_temp:
                    yield generate_crafting_construct_helper(machine, recipe, fuel_name, fuel_value, data, temperature_settings=temps)
        else:
            yield generate_crafting_construct_helper(machine, recipe, fuel_name, fuel_value, data)

def generate_crafting_construct_helper(machine, recipe, fuel_name, fuel_value, data, temperature_settings={}):
    ident = recipe['name']+" in "+machine['name']+" with "+" & ".join([fuel_name]+[k+'@'+str(v) for k, v in temperature_settings.items()])
    
    if 'drain' in machine['energy_source']:
        drain = {fuel_name: machine['energy_source']['drain_raw']/fuel_value}
        logging.debug("Found a drain value of %d", drain[fuel_name])
    elif fuel_name=='electric':
        drain = {fuel_name: (machine['energy_usage_raw']/30.0)/fuel_value}
        logging.debug("Machine is electrical so we assume drain has value of %d", drain[fuel_name])
    else:
        drain = {}
    
    def apply_temperature_settings(vec):
        fixed = {}
        for k, v in vec.items():
            if k in temperature_settings:
                fixed.update({k+'@'+str(temperature_settings[k]): v})
            else:
                fixed.update({k: v})
        return fixed
    
    vector = copy.deepcopy(recipe['vector'])
    vector = multi_dict(machine['crafting_speed'], vector)
    vector = apply_temperature_settings(vector)
    
    effect_effects = {'speed': [], 'productivity': [], 'consumption': [], 'pollution': []}
    if 'allowed_effects' in machine.keys():
        if 'speed' in machine['allowed_effects']:
            effect_effects.update({'speed': [e for e in vector.keys()]})
        if 'productivity' in machine['allowed_effects']:
            effect_effects.update({'productivity': [k for k, v in vector.items() if v > 0]})
        if 'consumption' in machine['allowed_effects']:
            effect_effects.update({'consumption': [fuel_name]})
        if 'pollution' in machine['allowed_effects']:
            effect_effects.update({'pollution': []})
    
    vector.update({fuel_name: -1.0*machine['energy_usage_raw']/fuel_value})#add this in after so it doesn't mess with 'speed'
    
    max_mods = 0
    if 'module_specification' in machine.keys() and 'module_slots' in machine['module_specification'].keys():
        max_mods = machine['module_specification']['module_slots']
    if max_mods==0:
        logging.warning("Didn't detect any module slots in %s this will disable beacon effects too.", machine['name'])
        allowed_modules = []
    else:
        allowed_modules = [(module, max_mods) for module in recipe['allowed_modules'] if all([eff in machine['allowed_effects'] for eff in module['effect'].keys()])]
        logging.debug("Found a total of %d allowed modules.", len(allowed_modules))
        
    base_inputs = copy.deepcopy(recipe['base_inputs'])
    base_inputs.update({fuel_name: (-1.0*machine['energy_usage_raw']/fuel_value)*recipe['energy_required']/machine['crafting_speed']})
    base_inputs = apply_temperature_settings(base_inputs)
    
    cost = {machine['name']: 1}
    
    limit = machine['limit'] + recipe['limit']

    return UncompiledConstruct(ident, drain, vector, effect_effects, allowed_modules, base_inputs, cost, limit)

def generate_boiler_machine_constructs(machine, data):
    """
    Generates UncompiledConstructs for a boiler.
    """
    assert machine['type']=='boiler'
    logging.debug("Generating construct family for: %s", machine['name'])
    
    if 'mode' in machine.keys() and machine['mode']=='heat-water-inside':
        logging.error("heat-water-inside wtf. Skipping.")
        return None
    
    if ('filter' in machine['fluid_box'].keys() and machine['fluid_box']['filter']!='water') or\
       ('filter' in machine['output_fluid_box'].keys() and machine['output_fluid_box']['filter']!='steam'):
        logging.error("Don't give me a weird ass boiler. Skipping.")
        return None
    fluid = 'steam'
    joules_per_unit = (100-data['fluid']['water']['default_temperature'])*convert_value_to_base_units(data['fluid']['water']['heat_capacity'])+\
                      (machine['target_temperature']-100)*convert_value_to_base_units(data['fluid'][fluid]['heat_capacity'])
    effectivity = 1
    if 'effectivity' in machine['energy_source'].keys():
        effectivity = machine['energy_source']['effectivity']
        logging.debug("Found a non 1 effectivity of %d", effectivity)
    units_per_second = effectivity*machine['energy_consumption_raw']/joules_per_unit

    if fluid not in RELEVENT_FLUID_TEMPERATURES.keys():
        RELEVENT_FLUID_TEMPERATURES.update({fluid: {}})
    RELEVENT_FLUID_TEMPERATURES[fluid].update({machine['target_temperature']: joules_per_unit})
    
    for fuel_name, fuel_value in dereference_complex_type(machine['energy_source']['complex_type'], data):
        ident = "water to steam@"+str(machine['target_temperature'])+" in "+machine['name']+" via "+fuel_name
        
        drain = {}
        
        vector = {fuel_name: -1*machine['energy_consumption_raw']/fuel_value, 
                  'water': -1*units_per_second, 
                  fluid+'@'+str(machine['target_temperature']): units_per_second}
                  
        effect_effects = {'speed': [], 'productivity': [], 'consumption': [], 'pollution': []}
        
        allowed_modules = []
        
        base_inputs = {fuel_name: -1*machine['energy_consumption_raw']/fuel_value}
        
        cost = {machine['name']: 1}
        
        limit = machine['limit']
        
        yield UncompiledConstruct(ident, drain, vector, effect_effects, allowed_modules, base_inputs, cost, limit)

def generate_mining_drill_construct(machine, data):
    """
    Generates UncompiledConstructs for a mining drill.
    """
    assert machine['type']=='mining-drill'
    logging.debug("Generating construct family for: %s", machine['name'])
        
    if 'base_productivity' in machine.keys() and machine['base_productivity']!=0:
        logging.error("Bad news, I haven't implemeneted base productivity yet. No constructs will be created from %s", machine['name'])
        return None
    
    for cata in machine['resource_categories']:
        for resource in data['resource-category'][cata]['resource_list']:
            for fuel_name, fuel_value in dereference_complex_type(machine['energy_source']['complex_type'], data):
                ident = machine["name"]+" mining: "+resource["name"]
                
                drain = {}
                
                vector = {resource["name"]: machine['mining_speed']/resource['minable']['mining_time'],
                          fuel_name: -1.0*machine['energy_usage_raw']/fuel_value}
                
                effect_effects = {'speed': [], 'productivity': [], 'consumption': [], 'pollution': []}
                if 'allowed_effects' in machine.keys():
                    if 'speed' in machine['allowed_effects']:
                        effect_effects.update({'speed': [resource["name"]]})
                    if 'productivity' in machine['allowed_effects']:
                        effect_effects.update({'productivity': [resource["name"]]})
                    if 'consumption' in machine['allowed_effects']:
                        effect_effects.update({'consumption': [fuel_name]})
                    if 'pollution' in machine['allowed_effects']:
                        effect_effects.update({'pollution': []})
                
                max_mods = 0
                if 'module_specification' in machine.keys() and 'module_slots' in machine['module_specification'].keys():
                    max_mods = machine['module_specification']['module_slots']
                if max_mods==0:
                    logging.warning("Didn't detect any module slots in %s this will disable beacon effects too.", machine['name'])
                    allowed_modules = []
                else:
                    allowed_modules = [(module, max_mods) for module in resource['allowed_modules']]
                    logging.debug("Found a total of %d allowed modules.", len(allowed_modules))
                
                base_inputs = {fuel_name: -1.0*machine['energy_usage_raw']/fuel_value}
        
                cost = {machine['name']: 1}
                
                limit = machine['limit']
                
                yield UncompiledConstruct(ident, drain, vector, effect_effects, allowed_modules, base_inputs, cost, limit)

def generate_all_constructs(data):
    logging.debug("Beginning the generation of all Uncompiled Constructs from \'data.raw\'")
    construct_families = []
    for ref in ['boiler', 'burner-generator', 'offshore-pump', 'reactor', 'generator', 'furnace', 'mining-drill', 'solar-panel', 'rocket-silo', 'assembling-machine']:
        logging.info("Starting processing of category: %s", ref)
        for machine in data[ref].values():
            logging.debug("Starting processing of machine: %s", machine['name'])
            if not machine['name'] in data['recipe'].keys():
                logging.debug("%s is a fake machine because you cant build it. Skipping.", machine['name'])
                continue
            #cost = {machine['name']: 1}
            try:
                if machine['type']=='assembling-machine' or machine['type']=='rocket-silo' or machine['type']=='furnace': #all of these are children of CraftingMachinePrototype
                    if 'fixed-recipe' in machine.keys():
                        logging.debug("%s has a fixed recipe named %s", machine['name'], machine['fixed-recipe'])
                        for fam in generate_crafting_construct(machine, data['recipe'][machine['fixed-recipe']], data):
                            if fam:
                                construct_families.append(fam)
                    for recipe in data['recipe'].values():
                        if valid_crafting_machine(machine, recipe):
                            for fam in generate_crafting_construct(machine, recipe, data):
                                if fam:
                                    construct_families.append(fam)
                
                elif machine['type']=='boiler':
                    for fam in generate_boiler_machine_constructs(machine, data):
                        construct_families.append(fam)
                
                elif machine['type']=='burner-generator':
                    for fuel_name, fuel_value in dereference_complex_type(machine['energy_source']['complex_type'], data):
                        effectivity = 1.0
                        if 'effectivity' in machine['energy_source'].keys():
                            effectivity = machine['energy_source']['effectivity']
                            logging.debug("Found a non 1 effectivity of %d", effectivity)
                            
                        construct_families.append(UncompiledConstruct("electric from "+machine['name']+" via "+fuel_name, 
                                                                      {}, 
                                                                      {fuel_name: -1.0*machine['max_power_output_raw']/effectivity/fuel_value,
                                                                       'electric': machine['max_power_output_raw']}, 
                                                                      {'speed': [], 'productivity': [], 'consumption': [], 'pollution': []}, 
                                                                      [], 
                                                                      {fuel_name: -1.0*machine['max_power_output_raw']/effectivity/fuel_value}, 
                                                                      {machine['name']: 1}, 
                                                                      machine['limit']))
                
                elif machine['type']=='generator': 
                    if not 'max_power_output_raw' in machine.keys():
                        logging.error("Max power output raw not calced for %s, better do that before getting here!", machine['name'])
                        continue
                    if not 'filter' in machine['fluid_box'].keys() or not machine['fluid_box']['filter'] in RELEVENT_FLUID_TEMPERATURES.keys() or not machine['maximum_temperature'] in RELEVENT_FLUID_TEMPERATURES[machine['fluid_box']['filter']].keys():
                        logging.error("No clue what %s is supposed to be consuming.", machine['name'])
                        continue
                    max_density = RELEVENT_FLUID_TEMPERATURES[machine['fluid_box']['filter']][machine['maximum_temperature']]
                    for relevent_temp, energy_density in RELEVENT_FLUID_TEMPERATURES[machine['fluid_box']['filter']].items():
                        if energy_density=="?":
                            logging.error("Generator %s ran into a weird fluid %s@%d that has no energy density calculated.", machine['name'], machine['fluid_box']['filter'], relevent_temp)
                        fluid_usage = min(60*machine['fluid_usage_per_tick'], machine['max_power_output_raw']/min(max_density, energy_density))
                        
                        effectivity = 1.0
                        if 'effectivity' in machine.keys():
                            effectivity = machine['effectivity']
                            logging.debug("Found a non 1 effectivity of %d", effectivity)
                        if 'effectivity' in machine['energy_source'].keys():
                            if effectivity!=1.0:
                                logging.error("More than one effectivity what is going on?")
                            effectivity = machine['energy_source']['effectivity']
                            logging.debug("Found a non 1 effectivity of %d", effectivity)
                        
                        construct_families.append(UncompiledConstruct("electric from "+machine['name']+" via "+machine['fluid_box']['filter']+"@"+str(relevent_temp), 
                                                                      {}, 
                                                                      {machine['fluid_box']['filter']+"@"+str(relevent_temp): -1.0*fluid_usage,
                                                                       'electric': effectivity*fluid_usage*min(max_density, energy_density)}, 
                                                                      {'speed': [], 'productivity': [], 'consumption': [], 'pollution': []}, 
                                                                      [], 
                                                                      {machine['fluid_box']['filter']+"@"+str(relevent_temp): -1.0*fluid_usage}, 
                                                                      {machine['name']: 1}, 
                                                                      machine['limit']))
                
                elif machine['type']=='mining-drill':
                    for fam in generate_mining_drill_construct(machine, data):
                        construct_families.append(fam)
                
                elif machine['type']=='offshore-pump':
                    construct_families.append(UncompiledConstruct(machine['name'], 
                                                                  {},
                                                                  {machine['fluid']: 60.0*machine['pumping_speed']},
                                                                  {'speed': [], 'productivity': [], 'consumption': [], 'pollution': []},
                                                                  [],
                                                                  {},
                                                                  {machine['name']: 1},
                                                                  machine['limit']))

                elif machine['type']=='reactor':
                    bonus = 1
                    if 'neighbour_bonus' in machine.keys():
                        bonus = 1 + 3 * machine['neighbour_bonus']
                    for fuel_name, fuel_value in dereference_complex_type(machine['energy_source']['complex_type'], data):
                        construct_families.append(UncompiledConstruct("heat from "+machine['name']+" via "+fuel_name, 
                                                                      {},
                                                                      {'heat': machine['consumption_raw']*bonus,
                                                                       fuel_name: -1*machine['consumption_raw']/fuel_value},
                                                                      {'speed': [], 'productivity': [], 'consumption': [], 'pollution': []},
                                                                      [],
                                                                      {fuel_name: -1},
                                                                      {machine['name']: 1},
                                                                      machine['limit']))

                elif machine['type']=='solar-panel':
                    construct_families.append(UncompiledConstruct(machine['name'], 
                                                                  {},
                                                                  {'electric': machine['production_raw']},
                                                                  {'speed': [], 'productivity': [], 'consumption': [], 'pollution': []},
                                                                  [],
                                                                  {},
                                                                  {machine['name']: 1},
                                                                  machine['limit']))

                else:
                    logging.error("Unknown type %s", machine['type'])

            except:
                logging.error("Unable to complete build for %s", machine['name'])
                raise ValueError("STUFF")
    return construct_families
