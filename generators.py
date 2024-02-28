import re
import itertools
import logging
import copy
from utils import *
from datarawparse import *
from linearconstructs import *
import typing
from typing import Generator

def recipe_element_count(recipe: dict, recipe_key: str, key_type: str, COST_MODE: str = 'normal') -> int:
    """
    Calculates how many recipe_key's with a key_type type a recipe uses.

    Parameters
    ----------
    recipe:
        A recipe. https://lua-api.factorio.com/latest/prototypes/RecipePrototype.html
    recipe_key:
        A key for the recipe. (usually 'ingredients', 'results', or 'result')
    key_type:
        A 'type' that dict elements of of the recipe key needs to be.
    
    Returns
    -------
    Number of liquid ingredients
    """
    if COST_MODE in recipe.keys():
        if not recipe_key in recipe[COST_MODE].keys():
            return 0
        elements = recipe[COST_MODE][recipe_key]
    else:
        if not recipe_key in recipe.keys():
            return 0
        elements = recipe[recipe_key]

    if not isinstance(elements, list):
        elements = [elements]
    
    return count_via_lambda(elements, lambda ingred: (isinstance(ingred, dict) and \
                                                      ((not 'type' in ingred.keys() and key_type=='solid') or \
                                                       ('type' in ingred.keys() and ingred['type']==key_type))) or \
                                                     (key_type=='solid'))

def valid_crafting_machine(machine: dict, recipe: dict, **kwargs: str | dict) -> bool:
    """
    Calculates if a crafting machine is able to complete a recipe.

    Parameters
    ----------
    machine:
        A crafting machine. https://lua-api.factorio.com/latest/prototypes/CraftingMachinePrototype.html
    recipe:
        A recipe. https://lua-api.factorio.com/latest/prototypes/RecipePrototype.html
    
    Returns
    -------
    True if the crafting machine can craft the recipe, otherwise False.
    """
    #We currently block all instances of recipes that "do nothing" here. Eventually we want to move this code to catalyst calculations
    #but I haven't found a good way of handling that yet. Many mods have other types of container and one could imagine a mod where some
    #fluid does need to be in a container for a process and therefor access to such recipes will be nessisary.
    if any([phrase in recipe['name'] for phrase in ['fill, empty']]) and ('barrel' in recipe['name']):
        return False

    if not all([(recipe['category'] if 'category' in recipe.keys() else 'crafting') in machine['crafting_categories'],
                (machine['ingredient_count'] if 'ingredient_count' in machine.keys() else 255) >= recipe_element_count(recipe, 'ingredients', 'solid', **kwargs)]):
        return False
    
    if (recipe_element_count(recipe, 'ingredients', 'fluid', **kwargs) != 0 or recipe_element_count(recipe, 'results', 'fluid', **kwargs) + recipe_element_count(recipe, 'result', 'fluid', **kwargs) != 0) and (not 'fluid_boxes' in machine.keys()):
        return False

    if 'fluid_boxes' in machine.keys():
        fluid_boxes = machine['fluid_boxes']
        if isinstance(fluid_boxes, dict):
            fluid_boxes = [fluid_boxes]
        machine_fluid_inputs = count_via_lambda(fluid_boxes, lambda box: ('production_type' in box.keys()) and (box['production_type'] == "input"))
        machine_fluid_input_outputs = count_via_lambda(fluid_boxes, lambda box: ('production_type' in box.keys()) and (box['production_type'] == "input-output"))
        machine_fluid_outputs = count_via_lambda(fluid_boxes, lambda box: ('production_type' in box.keys()) and (box['production_type'] == "output"))
    else:
        machine_fluid_inputs, machine_fluid_input_outputs, machine_fluid_outputs, = 0, 0, 0
    recipe_fluid_inputs = recipe_element_count(recipe, 'ingredients', 'fluid', **kwargs)
    recipe_fluid_outputs = recipe_element_count(recipe, 'results', 'fluid', **kwargs) + recipe_element_count(recipe, 'result', 'fluid', **kwargs)
    return machine_fluid_inputs + machine_fluid_input_outputs >= recipe_fluid_inputs and \
        machine_fluid_outputs + machine_fluid_input_outputs >= recipe_fluid_outputs and \
        machine_fluid_inputs + machine_fluid_outputs + machine_fluid_input_outputs >= recipe_fluid_inputs + recipe_fluid_outputs

def temperature_setting_generator(vector_source: dict, fuel_name: str | None, RELEVENT_FLUID_TEMPERATURES: dict) -> Generator[dict, None, None]:
    """
    Generates all potential temperature settings for a recipe.

    Parameters
    ----------
    vector_source:
        A recipe, resource, or technology with a vector.
    fuel_name:
        Name of fuel being used or None

    Yields
    ------
    Settings of combinations of temperatures for input fluids.
    """
    #elements are only in RELEVENT_FLUID_TEMPERATURES's keys as their true name, so if we have a vectored fluid at a temperature it won't hit
    knobs = list(set([k for k, v in vector_source['vector'].items() if k in RELEVENT_FLUID_TEMPERATURES.keys() if v <= 0] + \
                     [k for k, v in vector_source['base_inputs'].items() if k in RELEVENT_FLUID_TEMPERATURES.keys() if v <= 0] + \
                     ([fuel_name] if fuel_name in RELEVENT_FLUID_TEMPERATURES.keys() else [])))
    #we only look at inputs

    if len(knobs) == 0:
        yield {}
    else:
        for prod in itertools.product(*[list(RELEVENT_FLUID_TEMPERATURES[k].keys()) for k in knobs]):
            yield {k: v for k, v in zip(knobs, prod)}

def generate_fueled_construct_helper(machine: dict, vector_source: dict, fuel: tuple[str, Fraction, typing.Optional[str]], temperature_settings: dict) -> UncompiledConstruct:
    """
    Generate a UncompiledConstruct given the machine, recipe, fuel, and ingredient temperatures.

    Parameters
    ----------
    machine:
        A crafting machine. https://lua-api.factorio.com/latest/prototypes/CraftingMachinePrototype.html
    vector_source:
        A recipe, resource, or technology with a name, vector, and time_multiplier.
    fuel:
        Tuple in the form of: (name of fuel being used, energy density of fuel being used, result of burning the fuel)
    temperature_settings:
        Specific temperatures of inputs and fuels to use.
    
    Returns
    -------
    An UncompiledConstruct
    """
    fuel_name, fuel_value, fuel_burnt_result = fuel

    ident = vector_source['name']+" in "+machine['name']+" with "+" & ".join([fuel_name]+[k+'@'+str(v) for k, v in temperature_settings.items()])
    
    drain = calculate_drain(machine['energy_source'], machine['energy_usage_raw'], fuel_name, fuel_value)
        
    vector = machine['speed'] * vector_source['vector'] #'speed' populated during normalization
    #'vector' populated in vectorize_recipes
    if fuel_name in vector.keys(): #add in the fuel cost 
        vector[fuel_name] += -1 * machine['energy_usage_raw'] / fuel_value
    else:
        vector[fuel_name] = -1 * machine['energy_usage_raw'] / fuel_value
    if not fuel_burnt_result is None: #https://lua-api.factorio.com/latest/prototypes/ItemPrototype.html#burnt_result
        if fuel_burnt_result in vector.keys():
            vector[fuel_burnt_result] += machine['energy_usage_raw'] / fuel_value
        else:
            vector[fuel_burnt_result] = machine['energy_usage_raw'] / fuel_value
    for k in list(vector_source['vector'].keys()): #fixing temperature settings
        if k in temperature_settings.keys():
            vector.update({k+'@'+str(temperature_settings[k]): vector[k]})
            del vector[k]
    
    effect_effects = {'speed': [], 'productivity': [], 'consumption': []}
    if 'allowed_effects' in machine.keys():
        #https://lua-api.factorio.com/latest/types/EffectTypeLimitation.html
        if 'speed' in machine['allowed_effects']:
            effect_effects['speed'] += [e for e in vector.keys() if e!=fuel_name] #speed affects everything but fuel
        if 'productivity' in machine['allowed_effects']:
            effect_effects['productivity'] += [k for k, v in vector.items() if v > 0] #productivity only affects products
        if 'consumption' in machine['allowed_effects']:
            if fuel_name in temperature_settings.keys():
                effect_effects['consumption'] += [fuel_name+'@'+str(temperature_settings[fuel_name])]
            else:
                effect_effects['consumption'] += [fuel_name] #consumption only affects fuel
            if not fuel_burnt_result is None: #https://lua-api.factorio.com/latest/prototypes/ItemPrototype.html#burnt_result
                effect_effects['consumption'] += [fuel_burnt_result]
    
    max_internal_mods = 0 #TODO: add beacon support
    if 'module_specification' in machine.keys() and 'module_slots' in machine['module_specification'].keys():
        max_internal_mods = machine['module_specification']['module_slots'] #https://lua-api.factorio.com/latest/types/ModuleSpecification.html
    allowed_modules = []
    if max_internal_mods==0:
        logging.warning("Didn't detect any module slots in %s this will disable beacon effects too.", machine['name'])
    else: #allowed_modules was populated in link_modules
        if 'allowed_effects' in machine.keys():
            allowed_modules = [(module['name'], True, True) for module in vector_source['allowed_modules'] if all([eff in machine['allowed_effects'] for eff in module['effect'].keys()])] #TODO: unsure about this line
    
    base_inputs = copy.deepcopy(vector_source['base_inputs'])
    base_inputs.update({fuel_name: (-1 * machine['energy_usage_raw'] / fuel_value) * (vector_source['time_multiplier'] / machine['speed'])}) #'time_multiplier' populated during normalization
    for k in list(vector_source['base_inputs'].keys()): #fixing temperature settings
        if k in temperature_settings.keys():
            base_inputs.update({k+'@'+str(temperature_settings[k]): base_inputs[k]})
            del base_inputs[k]
        
    cost = CompressedVector({machine['name']: 1})
    
    try:
        limit = machine['limit'] + vector_source['limit']
    except:
        print('limit' in machine.keys())
        print('limit' in vector_source.keys())
        print(vector_source)
        raise ValueError(vector_source)

    return UncompiledConstruct(ident, drain, vector, effect_effects, allowed_modules, max_internal_mods, base_inputs, cost, limit, 
                               base_productivity = Fraction(machine['base_productivity']) if 'base_productivity' in machine.keys() else Fraction(1))
    #https://lua-api.factorio.com/latest/prototypes/CraftingMachinePrototype.html#base_productivity

def generate_crafting_constructs(machine: dict, recipe: dict, data: dict, RELEVENT_FLUID_TEMPERATURES: dict) -> Generator[UncompiledConstruct, None, None]:
    """
    Generates UncompiledConstructs of a machine that does recipe crafting.

    Parameters
    ----------
    machine:
        A crafting machine. https://lua-api.factorio.com/latest/prototypes/CraftingMachinePrototype.html
    recipe:
        A recipe. https://lua-api.factorio.com/latest/prototypes/RecipePrototype.html
    data:
        Entire data.raw. https://wiki.factorio.com/Data.raw
    
    Yields
    ------
    UncompiledConstructs
    """
    logging.debug("Generating uncompiled constructs for: %s in %s", recipe['name'], machine['name'])
    
    for fuel_name, fuel_value, fuel_burnt_result in fuels_from_energy_source(machine['energy_source'], data, RELEVENT_FLUID_TEMPERATURES):
        for temperature_setting in temperature_setting_generator(recipe, fuel_name, RELEVENT_FLUID_TEMPERATURES):
            yield generate_fueled_construct_helper(machine, recipe, (fuel_name, fuel_value, fuel_burnt_result), temperature_setting)

def generate_boiler_machine_constructs(machine: dict, data: dict, RELEVENT_FLUID_TEMPERATURES: dict) -> Generator[UncompiledConstruct, None, None]:
    """
    Generates UncompiledConstructs for a boiler.

    Parameters
    ----------
    machine:
        A boiler machine. https://lua-api.factorio.com/latest/prototypes/BoilerPrototype.html
    data:
        Entire data.raw. https://wiki.factorio.com/Data.raw
    
    Yields
    ------
    UncompiledConstructs
    """
    logging.debug("Generating uncompiled constructs for: %s", machine['name'])
    
    input_fluid = machine['fluid_box']['filter'] #https://lua-api.factorio.com/latest/prototypes/BoilerPrototype.html#fluid_box
    output_fluid = machine['output_fluid_box']['filter'] #https://lua-api.factorio.com/latest/prototypes/BoilerPrototype.html#output_fluid_box

    if 'mode' in machine.keys() and machine['mode']=='heat-water-inside': #https://lua-api.factorio.com/latest/prototypes/BoilerPrototype.html#mode
        machine['target_temperature'] = data['fluid'][output_fluid]['max_temperature'] #https://lua-api.factorio.com/latest/prototypes/BoilerPrototype.html#target_temperature
    
    joules_per_unit = (machine['target_temperature'] - data['fluid'][output_fluid]['default_temperature']) * convert_value_to_base_units(data['fluid'][output_fluid]['heat_capacity'])
    #https://lua-api.factorio.com/latest/prototypes/FluidPrototype.html#heat_capacity
    
    units_per_second = machine['energy_source']['effectivity'] * machine['energy_consumption_raw']/joules_per_unit #https://lua-api.factorio.com/latest/types/BurnerEnergySource.html#effectivity similar for other EnergySource types

    if not output_fluid in RELEVENT_FLUID_TEMPERATURES.keys():
        RELEVENT_FLUID_TEMPERATURES.update({output_fluid: {}})
    RELEVENT_FLUID_TEMPERATURES[output_fluid][machine['target_temperature']] = joules_per_unit
    
    for fuel_name, fuel_value, fuel_burnt_result in fuels_from_energy_source(machine['energy_source'], data, RELEVENT_FLUID_TEMPERATURES):
        ident = input_fluid+" to "+output_fluid+"@"+str(machine['target_temperature'])+" in "+machine['name']+" using "+fuel_name
        
        drain = {}
        
        vector = {fuel_name: -1 * machine['energy_consumption_raw'] / fuel_value,
                  input_fluid: -1*units_per_second, 
                  output_fluid+'@'+str(machine['target_temperature']): units_per_second}
        if not fuel_burnt_result is None:
            vector[fuel_burnt_result] = machine['energy_consumption_raw'] / fuel_value
                
        effect_effects = {'speed': [], 'productivity': [], 'consumption': [], 'pollution': []}
        
        allowed_modules = []
        
        base_inputs = {fuel_name: -1*machine['energy_consumption_raw']/fuel_value}
        
        cost = CompressedVector({machine['name']: 1})
        
        limit = machine['limit']
        
        yield UncompiledConstruct(ident, drain, vector, effect_effects, allowed_modules, 0, base_inputs, cost, limit)

def generate_mining_drill_construct(machine: dict, data: dict, RELEVENT_FLUID_TEMPERATURES: dict) -> Generator[UncompiledConstruct, None, None]:
    """
    Generates UncompiledConstructs for a mining drill.

    Parameters
    ----------
    machine:
        A mining drill. https://lua-api.factorio.com/latest/prototypes/MiningDrillPrototype.html
    data:
        Entire data.raw. https://wiki.factorio.com/Data.raw
    
    Yields
    ------
    UncompiledConstructs
    """
    logging.debug("Generating uncompiled constructs for: %s", machine['name'])
    
    for cata in machine['resource_categories']:
        for resource in data['resource-category'][cata]['resource_list']:
            for fuel_name, fuel_value, fuel_burnt_result in fuels_from_energy_source(machine['energy_source'], data, RELEVENT_FLUID_TEMPERATURES):
                for temperature_setting in temperature_setting_generator(resource, fuel_name, RELEVENT_FLUID_TEMPERATURES):
                    yield generate_fueled_construct_helper(machine, resource, (fuel_name, fuel_value, fuel_burnt_result), temperature_setting)

def generate_burner_generator(machine: dict, data: dict, RELEVENT_FLUID_TEMPERATURES: dict) -> Generator[UncompiledConstruct, None, None]:
    """
    Generates UncompiledConstructs for a burner generator.

    Parameters
    ----------
    machine:
        A burner generator. https://lua-api.factorio.com/latest/prototypes/BurnerGeneratorPrototype.html
    data:
        Entire data.raw. https://wiki.factorio.com/Data.raw
    
    Yields
    ------
    UncompiledConstructs
    """
    for fuel_name, fuel_value, fuel_burnt_result in fuels_from_energy_source(machine['energy_source'], data, RELEVENT_FLUID_TEMPERATURES):
        vector = CompressedVector({fuel_name: -1 * (machine['max_power_output_raw'] / fuel_value) / machine['energy_source']['effectivity'], #https://lua-api.factorio.com/latest/prototypes/BurnerGeneratorPrototype.html#max_power_output
                                   'electric': machine['max_power_output_raw']})
        if not fuel_burnt_result is None:
            vector[fuel_burnt_result] = (machine['max_power_output_raw'] / fuel_value) / machine['energy_source']['effectivity']

        yield UncompiledConstruct("electricity from "+machine['name']+" via "+fuel_name, 
                                  CompressedVector(), 
                                  vector, 
                                  {'speed': [], 'productivity': [], 'consumption': [], 'pollution': []}, 
                                  [], 
                                  0,
                                  CompressedVector({fuel_name: -1 * (1 / fuel_value) / machine['energy_source']['effectivity']}), 
                                  CompressedVector({machine['name']: 1}), 
                                  machine['limit'])

def generate_generator_constructs(machine: dict, data: dict, RELEVENT_FLUID_TEMPERATURES: dict) -> Generator[UncompiledConstruct, None, None]:
    """
    Generates UncompiledConstructs for a generator.

    Parameters
    ----------
    machine:
        A burner generator. https://lua-api.factorio.com/latest/prototypes/BurnerGeneratorPrototype.html
    data:
        Entire data.raw. https://wiki.factorio.com/Data.raw
    
    Yields
    ------
    UncompiledConstructs
    """
    if (not 'filter' in machine['fluid_box'].keys()) or (not machine['fluid_box']['filter'] in RELEVENT_FLUID_TEMPERATURES.keys()) or (not machine['maximum_temperature'] in RELEVENT_FLUID_TEMPERATURES[machine['fluid_box']['filter']].keys()):
        raise ValueError("No clue what %s is supposed to be consuming.", machine['name'])

    fluid_usage = 60 * machine['fluid_usage_per_tick'] #https://lua-api.factorio.com/latest/prototypes/GeneratorPrototype.html#fluid_usage_per_tick
    effectivity = machine['effectivity'] #https://lua-api.factorio.com/latest/prototypes/GeneratorPrototype.html#effectivity

    if 'burns_fluid' in machine.keys() and machine['burns_fluid']: #https://lua-api.factorio.com/latest/prototypes/GeneratorPrototype.html#burns_fluid
        fuel_name, fuel_value = machine['fluid_box']['filter'], data['fluid'][machine['fluid_box']['filter']]['fuel_value_raw']
        if 'max_power_output' in machine.keys(): #https://lua-api.factorio.com/latest/prototypes/GeneratorPrototype.html#max_power_output
            corrected_fluid_usage = min(fluid_usage, machine['max_power_output_raw'] / fuel_value / effectivity)

        yield UncompiledConstruct("electric from "+machine['name']+" via "+machine['fluid_box']['filter'], 
                                  CompressedVector(), 
                                  CompressedVector({fuel_name: -1 * corrected_fluid_usage, 
                                                    'electric': effectivity * corrected_fluid_usage * fuel_value}), 
                                  {'speed': [], 'productivity': [], 'consumption': [], 'pollution': []}, 
                                  [], 
                                  0, 
                                  CompressedVector({fuel_name: -1 * corrected_fluid_usage}), 
                                  CompressedVector({machine['name']: 1}), 
                                  machine['limit'])
        
    else:
        max_energy_density = RELEVENT_FLUID_TEMPERATURES[machine['fluid_box']['filter']][machine['maximum_temperature']]
        for relevent_temp, energy_density in RELEVENT_FLUID_TEMPERATURES[machine['fluid_box']['filter']].items():
            if 'max_power_output' in machine.keys(): #https://lua-api.factorio.com/latest/prototypes/GeneratorPrototype.html#max_power_output
                corrected_fluid_usage = min(fluid_usage, machine['max_power_output_raw'] / min(energy_density, max_energy_density) / effectivity)
            else:
                corrected_fluid_usage = fluid_usage
            
            yield UncompiledConstruct("electric from "+machine['name']+" via "+machine['fluid_box']['filter']+"@"+str(relevent_temp), 
                                     CompressedVector(), 
                                     CompressedVector({machine['fluid_box']['filter']+"@"+str(relevent_temp): -1 * corrected_fluid_usage,
                                                       'electric': effectivity * corrected_fluid_usage * min(energy_density, max_energy_density)}), 
                                     {'speed': [], 'productivity': [], 'consumption': [], 'pollution': []}, 
                                     [], 
                                     0, 
                                     CompressedVector({machine['fluid_box']['filter']+"@"+str(relevent_temp): -1 * corrected_fluid_usage}), 
                                     CompressedVector({machine['name']: 1}), 
                                     machine['limit'])

def generate_reactor_constructs(machine: dict, data: dict, RELEVENT_FLUID_TEMPERATURES: dict) -> Generator[UncompiledConstruct, None, None]:
    """
    Generates UncompiledConstructs for a reactor.

    Parameters
    ----------
    machine:
        A reactor. https://lua-api.factorio.com/latest/prototypes/ReactorPrototype.html
    data:
        Entire data.raw. https://wiki.factorio.com/Data.raw
    
    Yields
    ------
    UncompiledConstructs
    """
    bonus = 1
    if 'neighbour_bonus' in machine.keys():
        bonus = 1 + 3 * machine['neighbour_bonus']
    
    for fuel_name, fuel_value, fuel_burnt_result in fuels_from_energy_source(machine['energy_source'], data, RELEVENT_FLUID_TEMPERATURES): #https://lua-api.factorio.com/latest/prototypes/ReactorPrototype.html#energy_source
        vector = CompressedVector({'heat': machine['energy_source']['effectivity'] * machine['consumption_raw'] * bonus, 
                                   fuel_name: -1 * machine['consumption_raw'] / fuel_value})
        if not fuel_burnt_result is None:
            vector[fuel_burnt_result] = machine['consumption_raw'] / fuel_value

        yield UncompiledConstruct("heat from "+machine['name']+" via "+fuel_name, 
                                  CompressedVector(), 
                                  vector, 
                                  {'speed': [], 'productivity': [], 'consumption': [], 'pollution': []}, 
                                  [], 
                                  0, 
                                  CompressedVector({fuel_name: -1}), 
                                  CompressedVector({machine['name']: 1}), 
                                  machine['limit'])

def valid_lab(machine: dict, technology: dict) -> bool:
    """
    Calculates if a lab is able to be used to research a technology.

    Parameters
    ----------
    machine:
        A crafting machine. https://lua-api.factorio.com/latest/prototypes/CraftingMachinePrototype.html
    technology:
        A technology. https://lua-api.factorio.com/latest/prototypes/TechnologyPrototype.html
    
    Returns
    -------
    True if the lab is able to be used to research a technology, otherwise False.
    """
    for tool in technology['base_inputs'].keys():
        if not tool in machine['inputs']: #https://lua-api.factorio.com/latest/prototypes/LabPrototype.html#inputs
            return False
    return True

def generate_lab_construct(machine: dict, technology: dict, data: dict, RELEVENT_FLUID_TEMPERATURES: dict) -> Generator[UncompiledConstruct, None, None]:
    """
    Generates UncompiledConstructs for a lab researching a technology.

    Parameters
    ----------
    machine:
        A lab. https://lua-api.factorio.com/latest/prototypes/LabPrototype.html
    technology:
        A technology. https://lua-api.factorio.com/latest/prototypes/TechnologyPrototype.html
    data:
        Entire data.raw. https://wiki.factorio.com/Data.raw
    
    Yields
    ------
    UncompiledConstructs
    """
    logging.debug("Generating uncompiled constructs for: %s in %s", technology['name'], machine['name'])
    
    for fuel_name, fuel_value, fuel_burnt_result in fuels_from_energy_source(machine['energy_source'], data, RELEVENT_FLUID_TEMPERATURES):
        for temperature_setting in temperature_setting_generator(technology, fuel_name, RELEVENT_FLUID_TEMPERATURES):
            yield generate_fueled_construct_helper(machine, technology, (fuel_name, fuel_value, fuel_burnt_result), temperature_setting)

def generate_all_constructs(data: dict, RELEVENT_FLUID_TEMPERATURES: dict, **kwargs: str | dict) -> list[UncompiledConstruct]:
    """
    Generates UncompiledConstructs for all machines in the version of the game represented in data.

    Parameters
    ----------
    data:
        Entire data.raw. https://wiki.factorio.com/Data.raw
    
    Yields
    ------
    A list of all UncompiledConstructs
    """
    logging.debug("Beginning the generation of all UncompiledConstructs from data.raw")

    all_uncompiled_constructs = []

    for ref in ['boiler', 'burner-generator', 'offshore-pump', 'reactor', 'generator', 'furnace', 'mining-drill', 'solar-panel', 'rocket-silo', 'assembling-machine', 'lab']:
        logging.info("Starting construct generation of machines in category: %s", ref)
        for machine in data[ref].values():
            logging.debug("Starting processing of machine: %s", machine['name'])

            if not machine['name'] in data['recipe'].keys():
                logging.debug("%s is a fake machine because you cant build it. Skipping.", machine['name'])
                continue


            if machine['type']=='assembling-machine' or machine['type']=='rocket-silo' or machine['type']=='furnace': #all children of https://lua-api.factorio.com/latest/prototypes/CraftingMachinePrototype.html
                if 'fixed-recipe' in machine.keys(): #https://lua-api.factorio.com/latest/prototypes/AssemblingMachinePrototype.html#fixed_recipe
                    logging.debug("%s has a fixed recipe named %s", machine['name'], machine['fixed-recipe'])
                    for construct in generate_crafting_constructs(machine, data['recipe'][machine['fixed-recipe']], data, RELEVENT_FLUID_TEMPERATURES):
                        if construct:
                            all_uncompiled_constructs.append(construct)

                for recipe in data['recipe'].values():
                    if valid_crafting_machine(machine, recipe, **kwargs):
                        for construct in generate_crafting_constructs(machine, recipe, data, RELEVENT_FLUID_TEMPERATURES):
                            if construct:
                                all_uncompiled_constructs.append(construct)
            

            elif machine['type']=='boiler': #https://lua-api.factorio.com/latest/prototypes/BoilerPrototype.html
                for construct in generate_boiler_machine_constructs(machine, data, RELEVENT_FLUID_TEMPERATURES):
                    all_uncompiled_constructs.append(construct)
            

            elif machine['type']=='burner-generator': #https://lua-api.factorio.com/latest/prototypes/BurnerGeneratorPrototype.html
                for construct in generate_burner_generator(machine, data, RELEVENT_FLUID_TEMPERATURES):
                    all_uncompiled_constructs.append(construct)
            

            elif machine['type']=='generator': #https://lua-api.factorio.com/latest/prototypes/GeneratorPrototype.html
                for construct in generate_generator_constructs(machine, data, RELEVENT_FLUID_TEMPERATURES):
                    all_uncompiled_constructs.append(construct)
            

            elif machine['type']=='mining-drill': #https://lua-api.factorio.com/latest/prototypes/MiningDrillPrototype.html
                for fam in generate_mining_drill_construct(machine, data, RELEVENT_FLUID_TEMPERATURES):
                    all_uncompiled_constructs.append(fam)
            

            elif machine['type']=='offshore-pump': #https://lua-api.factorio.com/latest/prototypes/OffshorePumpPrototype.html
                all_uncompiled_constructs.append(UncompiledConstruct(machine['name'], 
                                                                     CompressedVector(), 
                                                                     CompressedVector({machine['fluid']: Fraction(60 * machine['pumping_speed'])}), #https://lua-api.factorio.com/latest/prototypes/OffshorePumpPrototype.html#pumping_speed
                                                                     {'speed': [], 'productivity': [], 'consumption': [], 'pollution': []}, 
                                                                     [], 
                                                                     0,
                                                                     CompressedVector(), 
                                                                     CompressedVector({machine['name']: Fraction(1)}), 
                                                                     machine['limit']))


            elif machine['type']=='reactor': #https://lua-api.factorio.com/latest/prototypes/ReactorPrototype.html
                for construct in generate_reactor_constructs(machine, data, RELEVENT_FLUID_TEMPERATURES):
                    all_uncompiled_constructs.append(construct)


            elif machine['type']=='solar-panel': #https://lua-api.factorio.com/latest/prototypes/SolarPanelPrototype.html
                all_uncompiled_constructs.append(UncompiledConstruct(machine['name'], 
                                                                     CompressedVector(), 
                                                                     CompressedVector({'electric': machine['production_raw']}), 
                                                                     {'speed': [], 'productivity': [], 'consumption': [], 'pollution': []}, 
                                                                     [], 
                                                                     0,
                                                                     CompressedVector(), 
                                                                     CompressedVector({machine['name']: Fraction(1)}), 
                                                                     machine['limit']))


            elif machine['type']=='lab': #https://lua-api.factorio.com/latest/prototypes/LabPrototype.html
                for technology in data['technology'].values():
                    if valid_lab(machine, technology):
                        for construct in generate_lab_construct(machine, technology, data, RELEVENT_FLUID_TEMPERATURES):
                            if construct:
                                all_uncompiled_constructs.append(construct)


            else:
                raise ValueError("Unknown type %s", machine['type'])

    return all_uncompiled_constructs

