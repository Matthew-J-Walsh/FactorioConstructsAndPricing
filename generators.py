from __future__ import annotations
from globalsandimports import *

from utils import *
from datarawparse import *
from constructs import *

if TYPE_CHECKING:
    from tools import FactorioInstance

def recipe_element_count(recipe: dict, recipe_key: str, classification: str, instance: FactorioInstance) -> int:
    """Calculates how many recipe_key's with a classification type a recipe uses

    Parameters
    ----------
    recipe : dict
        A recipe https://lua-api.factorio.com/latest/prototypes/RecipePrototype.html
    recipe_key : str
         A key for the recipe (usually 'ingredients', 'results', or 'result')
    classification : str
        A 'type' that dict elements of of the recipe key needs to be
    instance : FactorioInstance
        FactorioInstance to use

    Returns
    -------
    int
        Number of elements in a recipe that have a typing match
    """
    if instance.COST_MODE in recipe.keys():
        if not recipe_key in recipe[instance.COST_MODE].keys():
            return 0
        elements = recipe[instance.COST_MODE][recipe_key]
    else:
        if not recipe_key in recipe.keys():
            return 0
        elements = recipe[recipe_key]
    
    return count_via_lambda(elements, lambda ingred: ingred['type']==classification)

def valid_crafting_machine(machine: dict, recipe: dict, instance: FactorioInstance) -> bool:
    """Calculates if a crafting machine is able to complete a recipe

    Parameters
    ----------
    machine : dict
        A crafting machine https://lua-api.factorio.com/latest/prototypes/CraftingMachinePrototype.html
    recipe : dict
        A recipe https://lua-api.factorio.com/latest/prototypes/RecipePrototype.html
    instance : FactorioInstance
        FactorioInstance to use

    Returns
    -------
    bool
        If the crafting machine can craft the recipe
    """
    #We currently block all instances of recipes that "do nothing" here. Eventually we want to move this code to catalyst calculations
    #but I haven't found a good way of handling that yet. Many mods have other types of container and one could imagine a mod where some
    #fluid does need to be in a container for a process and therefor access to such recipes will be nessisary.
    #TODO
    if any([phrase in recipe['name'] for phrase in ['fill', 'empty']]) and ('barrel' in recipe['name']): #'fill', 'empty'
        return False

    #block all recipes that cannot be enabled (cheat mode only)
    if not recipe['enableable']:
        return False

    if not (recipe['category'] if 'category' in recipe.keys() else 'crafting') in machine['crafting_categories'] or \
        not (machine['ingredient_count'] if 'ingredient_count' in machine.keys() else 255) >= recipe_element_count(recipe, 'ingredients', 'solid', instance):
        return False
    
    if 'fluid_boxes' in machine.keys():
        machine_fluid_inputs = count_via_lambda(machine['fluid_boxes'], lambda box: isinstance(box, dict) and ('production_type' in box.keys()) and (box['production_type'] == "input"))
        machine_fluid_input_outputs = count_via_lambda(machine['fluid_boxes'], lambda box: isinstance(box, dict) and ('production_type' in box.keys()) and (box['production_type'] == "input-output"))
        machine_fluid_outputs = count_via_lambda(machine['fluid_boxes'], lambda box: isinstance(box, dict) and ('production_type' in box.keys()) and (box['production_type'] == "output"))
    else:
        machine_fluid_inputs, machine_fluid_input_outputs, machine_fluid_outputs, = 0, 0, 0
    recipe_fluid_inputs = recipe_element_count(recipe, 'ingredients', 'fluid', instance)
    recipe_fluid_outputs = recipe_element_count(recipe, 'results', 'fluid', instance)
    return machine_fluid_inputs + machine_fluid_input_outputs >= recipe_fluid_inputs and \
           machine_fluid_outputs + machine_fluid_input_outputs >= recipe_fluid_outputs and \
           machine_fluid_inputs + machine_fluid_outputs + machine_fluid_input_outputs >= recipe_fluid_inputs + recipe_fluid_outputs

def temperature_setting_generator(fuel_name: str, instance: FactorioInstance, vector_source: dict | None = None) -> Generator[dict, None, None]:
    """Generates all potential temperature settings for a recipe

    Parameters
    ----------
    fuel_name : str
        Name of fuel being used or None
    instance : FactorioInstance
        FactorioInstance to use
    vector_source : dict | None, optional
        A recipe, resource, or technology with a vector, by default None

    Yields
    ------
    Generator[dict, None, None]
        Settings of combinations of temperatures for input fluids
    """
    #elements are only in RELEVENT_FLUID_TEMPERATURES's keys as their true name, so if we have a vectored fluid at a temperature it won't hit
    if vector_source is None:
        knobs = [fuel_name] if fuel_name in instance.RELEVENT_FLUID_TEMPERATURES.keys() else []
    else:
        knobs = list(set([k for k in vector_source['base_inputs'].keys() if k in instance.RELEVENT_FLUID_TEMPERATURES.keys()] + \
                        ([fuel_name] if fuel_name in instance.RELEVENT_FLUID_TEMPERATURES.keys() else [])))

    if len(knobs) == 0:
        yield {}
    else:
        for prod in itertools.product(*[list(instance.RELEVENT_FLUID_TEMPERATURES[k].keys()) for k in knobs]):
            yield {k: v for k, v in zip(knobs, prod)}

def module_specification_calculation(machine: dict, vector_source: dict) -> tuple[int, list[tuple[str, bool, bool]]]:
    """Calculates the module specifications of a machine with a vector source of module reference if no vector source exists

    Parameters
    ----------
    machine : dict
        A crafting machine, mining drill, or lab
    vector_source : dict
        A recipe, resource, technology, or dict that contains key 'allowed_modules'

    Returns
    -------
    tuple[int, list[tuple[str, bool, bool]]]
        maximum amount of interal modules allowed in machine,
        list of tuples indicating what modules can be used where in the machine
    """
    allowed_modules = []
    max_internal_mods: int = machine['module_specification']['module_slots'] #https://lua-api.factorio.com/latest/types/ModuleSpecification.html

    allowed_modules: list[tuple[str, bool, bool]] = [(module['name'], True, "productivity" not in module['name']) for module in vector_source['allowed_modules'] if all([eff in machine['allowed_effects'] for eff in module['effect'].keys()])]
    return max_internal_mods, allowed_modules

def fix_temperature_settings(temperature_settings: dict, vector: CompressedVector) -> None:
    """Edits the temperature settings of a vector, replacing fluids with specific temperatures with their set temperature

    Parameters
    ----------
    temperature_settings : dict
        Temperature settings generated from temperature_setting_generator
    vector : CompressedVector
        Vector representing changes made by a construct
    """
    for k in list(vector.keys()):
        if k in temperature_settings.keys():
            vector[k+'@'+str(temperature_settings[k])] = vector[k]
            del vector[k]
    
def generate_fueled_construct_helper(machine: dict, vector_source: dict, fuel: tuple[str, Fraction, str | None], temperature_settings: dict, research_effected: list[str] | None = None) -> UncompiledConstruct:
    """Generate a UncompiledConstruct given the machine, recipe, fuel, and ingredient temperatures

    Parameters
    ----------
    machine : dict
        A crafting machine, mining drill, or lab
    vector_source : dict
        A recipe, resource, or technology
    fuel : tuple[str, Fraction, str  |  None]
        Name of fuel being used,
        Energy Density of fuel being used,
        Result of burning the fuel
    temperature_settings : dict
        Specific temperatures of inputs and fuels to use
    research_effected : list[str]
        List of special research effect that should affect this construct

    Returns
    -------
    UncompiledConstruct
        Constructed Construct
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
    fix_temperature_settings(temperature_settings, vector)
    
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
    
    max_internal_mods, allowed_modules = module_specification_calculation(machine, vector_source)
    
    base_inputs = copy.deepcopy(vector_source['base_inputs'])
    base_inputs[fuel_name] = (-1 * machine['energy_usage_raw'] / fuel_value) * (vector_source['time_multiplier'] / machine['speed']) #'time_multiplier' populated during normalization
    fix_temperature_settings(temperature_settings, base_inputs)
        
    cost = CompressedVector({machine['name']: Fraction(1)})
    
    limit = machine['limit'] + vector_source['limit']

    base_productivity = Fraction(machine['base_productivity']).limit_denominator() if 'base_productivity' in machine.keys() else Fraction(0) #https://lua-api.factorio.com/latest/prototypes/CraftingMachinePrototype.html#base_productivity

    return UncompiledConstruct(ident, drain, vector, effect_effects, allowed_modules, max_internal_mods, base_inputs, cost, limit, machine, base_productivity, research_effected)

def generate_crafting_constructs(machine: dict, recipe: dict, instance: FactorioInstance) -> Generator[UncompiledConstruct, None, None]:
    """Generates UncompiledConstructs of a machine that does recipe crafting

    Parameters
    ----------
    machine : dict
        A crafting machine https://lua-api.factorio.com/latest/prototypes/CraftingMachinePrototype.html
    recipe : dict
        A recipe https://lua-api.factorio.com/latest/prototypes/RecipePrototype.html
    instance : FactorioInstance
        FactorioInstance to use

    Yields
    ------
    Generator[UncompiledConstruct, None, None]
        Constructs for the machine recipe combo
    """
    logging.log(5, "Generating uncompiled constructs for: %s in %s", recipe['name'], machine['name'])
    
    for fuel_name, fuel_value, fuel_burnt_result in fuels_from_energy_source(machine['energy_source'], instance):
        for temperature_setting in temperature_setting_generator(fuel_name, instance, recipe):
            yield generate_fueled_construct_helper(machine, recipe, (fuel_name, fuel_value, fuel_burnt_result), temperature_setting)

def generate_boiler_machine_constructs(machine: dict, instance: FactorioInstance) -> Generator[UncompiledConstruct, None, None]:
    """Generates UncompiledConstructs for a boiler

    Parameters
    ----------
    machine : dict
        A boiler machine https://lua-api.factorio.com/latest/prototypes/BoilerPrototype.html
    instance : FactorioInstance
        FactorioInstance to use

    Yields
    ------
    Generator[UncompiledConstruct, None, None]
        Constructs for the boiler
    """
    logging.log(5, "Generating uncompiled constructs for: %s", machine['name'])
    
    input_fluid = machine['fluid_box']['filter'] #https://lua-api.factorio.com/latest/prototypes/BoilerPrototype.html#fluid_box
    output_fluid = machine['output_fluid_box']['filter'] #https://lua-api.factorio.com/latest/prototypes/BoilerPrototype.html#output_fluid_box

    if 'mode' in machine.keys() and machine['mode']=='heat-water-inside': #https://lua-api.factorio.com/latest/prototypes/BoilerPrototype.html#mode
        machine['target_temperature'] = instance.data_raw['fluid'][output_fluid]['max_temperature'] #https://lua-api.factorio.com/latest/prototypes/BoilerPrototype.html#target_temperature
    
    joules_per_unit = (machine['target_temperature'] - instance.data_raw['fluid'][output_fluid]['default_temperature']) * convert_value_to_base_units(instance.data_raw['fluid'][output_fluid]['heat_capacity'])
    #https://lua-api.factorio.com/latest/prototypes/FluidPrototype.html#heat_capacity
    
    units_per_second = machine['energy_source']['effectivity'] * machine['energy_consumption_raw']/joules_per_unit #https://lua-api.factorio.com/latest/types/BurnerEnergySource.html#effectivity similar for other EnergySource types

    if not output_fluid in instance.RELEVENT_FLUID_TEMPERATURES.keys():
        instance.RELEVENT_FLUID_TEMPERATURES.update({output_fluid: {}})
    instance.RELEVENT_FLUID_TEMPERATURES[output_fluid][machine['target_temperature']] = joules_per_unit
    
    for fuel_name, fuel_value, fuel_burnt_result in fuels_from_energy_source(machine['energy_source'], instance):
        ident = input_fluid+" to "+output_fluid+"@"+str(machine['target_temperature'])+" in "+machine['name']+" with "+fuel_name
        
        drain = CompressedVector()
        
        vector = CompressedVector({fuel_name: -1 * machine['energy_consumption_raw'] / fuel_value,
                                   input_fluid: -1*units_per_second, 
                                   output_fluid+'@'+str(machine['target_temperature']): units_per_second})
        if not fuel_burnt_result is None:
            vector[fuel_burnt_result] = machine['energy_consumption_raw'] / fuel_value
                
        effect_effects = {'speed': [], 'productivity': [], 'consumption': [], 'pollution': []}
        
        allowed_modules = []
        
        base_inputs = CompressedVector({fuel_name: Fraction(-1)})
        
        cost = CompressedVector({machine['name']: Fraction(1)})
        
        limit = machine['limit']
        
        yield UncompiledConstruct(ident, drain, vector, effect_effects, allowed_modules, 0, base_inputs, cost, limit, machine)

def generate_mining_drill_constructs(machine: dict, instance: FactorioInstance) -> Generator[UncompiledConstruct, None, None]:
    """Generates UncompiledConstructs for a mining drill

    Parameters
    ----------
    machine : dict
        A mining drill https://lua-api.factorio.com/latest/prototypes/MiningDrillPrototype.html
    instance : FactorioInstance
        FactorioInstance to use

    Yields
    ------
    Generator[UncompiledConstruct, None, None]
        Constructs for the mining drill
    """
    logging.log(5, "Generating uncompiled constructs for: %s", machine['name'])
    
    for cata in machine['resource_categories']:
        for resource in instance.data_raw['resource-category'][cata]['resource_list']:
            if 'required_fluid' in resource['minable'].keys() and not 'input_fluid_box' in machine.keys(): #drills without fluid boxes cannot mine resources requiring fluids.
                continue
            for fuel_name, fuel_value, fuel_burnt_result in fuels_from_energy_source(machine['energy_source'], instance):
                for temperature_setting in temperature_setting_generator(fuel_name, instance, resource):
                    yield generate_fueled_construct_helper(machine, resource, (fuel_name, fuel_value, fuel_burnt_result), temperature_setting, ["mining-drill-productivity-bonus"])

def generate_burner_generator_constructs(machine: dict, instance: FactorioInstance) -> Generator[UncompiledConstruct, None, None]:
    """Generates UncompiledConstructs for a burner generator

    Parameters
    ----------
    machine : dict
        A burner generator https://lua-api.factorio.com/latest/prototypes/BurnerGeneratorPrototype.html
    instance : FactorioInstance
        FactorioInstance to use

    Yields
    ------
    Generator[UncompiledConstruct, None, None]
        Constructs for the burner generator
    """
    for fuel_name, fuel_value, fuel_burnt_result in fuels_from_energy_source(machine['energy_source'], instance):
        for temperature_settings in temperature_setting_generator(fuel_name, instance):
            ident = "electricity from "+machine['name']+" via "+fuel_name

            vector = CompressedVector({fuel_name: -1 * (machine['max_power_output_raw'] / fuel_value) / machine['energy_source']['effectivity'], #https://lua-api.factorio.com/latest/prototypes/BurnerGeneratorPrototype.html#max_power_output
                                    'electric': machine['max_power_output_raw']})
            if not fuel_burnt_result is None:
                vector[fuel_burnt_result] = (machine['max_power_output_raw'] / fuel_value) / machine['energy_source']['effectivity']
            fix_temperature_settings(temperature_settings, vector)
            
            base_inputs = CompressedVector({fuel_name: -1 * (1 / fuel_value) / machine['energy_source']['effectivity']})
            fix_temperature_settings(temperature_settings, base_inputs)

            cost = CompressedVector({machine['name']: Fraction(1)})

            yield UncompiledConstruct(ident, CompressedVector(), vector, 
                                    {'speed': [], 'productivity': [], 'consumption': [], 'pollution': []}, [], 0,
                                    base_inputs, cost, machine['limit'], machine)

def generate_generator_constructs(machine: dict, instance: FactorioInstance) -> Generator[UncompiledConstruct, None, None]:
    """Generates UncompiledConstructs for a generator

    Parameters
    ----------
    machine : dict
         A burner generator https://lua-api.factorio.com/latest/prototypes/BurnerGeneratorPrototype.html
    instance : FactorioInstance
        FactorioInstance to use

    Yields
    ------
    Generator[UncompiledConstruct, None, None]
        Constructs for the generator

    Raises
    ------
    ValueError
        If the program cannot understand what the generator's input is
    """
    if (not 'filter' in machine['fluid_box'].keys()) or (not machine['fluid_box']['filter'] in instance.RELEVENT_FLUID_TEMPERATURES.keys()) or (not machine['maximum_temperature'] in instance.RELEVENT_FLUID_TEMPERATURES[machine['fluid_box']['filter']].keys()):
        raise ValueError("No clue what %s is supposed to be consuming.", machine['name'])

    fluid_usage = Fraction(60 * machine['fluid_usage_per_tick']).limit_denominator() #https://lua-api.factorio.com/latest/prototypes/GeneratorPrototype.html#fluid_usage_per_tick
    effectivity = Fraction(machine['effectivity'] if 'effectivity' in machine.keys() else 1).limit_denominator() #https://lua-api.factorio.com/latest/prototypes/GeneratorPrototype.html#effectivity

    if 'burns_fluid' in machine.keys() and machine['burns_fluid']: #https://lua-api.factorio.com/latest/prototypes/GeneratorPrototype.html#burns_fluid
        fuel_name, fuel_value = machine['fluid_box']['filter'], instance.data_raw['fluid'][machine['fluid_box']['filter']]['fuel_value_raw']

        ident = "electric from "+machine['name']+" via "+fuel_name

        if 'max_power_output' in machine.keys(): #https://lua-api.factorio.com/latest/prototypes/GeneratorPrototype.html#max_power_output
            corrected_fluid_usage = Fraction(min(fluid_usage, machine['max_power_output_raw'] / fuel_value / effectivity)).limit_denominator()
        else:
            corrected_fluid_usage = fluid_usage

        vector = CompressedVector({fuel_name: -1 * corrected_fluid_usage, 'electric': effectivity * corrected_fluid_usage * fuel_value})
        
        base_inputs = CompressedVector({fuel_name: -1 * corrected_fluid_usage})

        cost = CompressedVector({machine['name']: Fraction(1)})

        yield UncompiledConstruct(ident, CompressedVector(), vector, 
                                  {'speed': [], 'productivity': [], 'consumption': [], 'pollution': []}, [], 0, 
                                  base_inputs, cost, machine['limit'], machine)
        
    else:
        max_energy_density = instance.RELEVENT_FLUID_TEMPERATURES[machine['fluid_box']['filter']][machine['maximum_temperature']]
        for relevent_temp, energy_density in instance.RELEVENT_FLUID_TEMPERATURES[machine['fluid_box']['filter']].items():
            ident = "electric from "+machine['name']+" via "+machine['fluid_box']['filter']+"@"+str(relevent_temp)
        
            if 'max_power_output' in machine.keys(): #https://lua-api.factorio.com/latest/prototypes/GeneratorPrototype.html#max_power_output
                corrected_fluid_usage = Fraction(min(fluid_usage, machine['max_power_output_raw'] / min(energy_density, max_energy_density) / effectivity)).limit_denominator()
            else:
                corrected_fluid_usage = Fraction(fluid_usage).limit_denominator()
            
            vector = CompressedVector({machine['fluid_box']['filter']+"@"+str(relevent_temp): -1 * corrected_fluid_usage,
                                       'electric': effectivity * corrected_fluid_usage * min(energy_density, max_energy_density)})
            
            base_inputs = CompressedVector({machine['fluid_box']['filter']+"@"+str(relevent_temp): -1 * corrected_fluid_usage})

            cost = CompressedVector({machine['name']: Fraction(1)})
            
            yield UncompiledConstruct(ident, CompressedVector(), vector, 
                                     {'speed': [], 'productivity': [], 'consumption': [], 'pollution': []}, [], 0, 
                                     base_inputs, cost, machine['limit'], machine)

def generate_reactor_constructs(machine: dict, instance: FactorioInstance) -> Generator[UncompiledConstruct, None, None]:
    """Generates UncompiledConstructs for a reactor

    Parameters
    ----------
    machine : dict
        A reactor https://lua-api.factorio.com/latest/prototypes/ReactorPrototype.html
    instance : FactorioInstance
        FactorioInstance to use

    Yields
    ------
    Generator[UncompiledConstruct, None, None]
        Constructs for the reactor
    """
    bonus = 1
    if 'neighbour_bonus' in machine.keys():
        bonus = 1 + 3 * machine['neighbour_bonus']
    
    for fuel_name, fuel_value, fuel_burnt_result in fuels_from_energy_source(machine['energy_source'], instance): #https://lua-api.factorio.com/latest/prototypes/ReactorPrototype.html#energy_source
        for temperature_settings in temperature_setting_generator(fuel_name, instance):
            ident = "heat from "+machine['name']+" via "+fuel_name
            
            vector = CompressedVector({'heat': machine['energy_source']['effectivity'] * machine['consumption_raw'] * bonus, 
                                       fuel_name: -1 * machine['consumption_raw'] / fuel_value})
            if not fuel_burnt_result is None:
                vector[fuel_burnt_result] = machine['consumption_raw'] / fuel_value
            fix_temperature_settings(temperature_settings, vector)

            base_inputs = CompressedVector({fuel_name: Fraction(-1)})
            fix_temperature_settings(temperature_settings, base_inputs)

            cost = CompressedVector({machine['name']: Fraction(1)})

            #https://lua-api.factorio.com/latest/types/HeatBuffer.html
            if 'min_working_temperature' in machine['heat_buffer']: #need to calculate warm-up cost, https://lua-api.factorio.com/latest/types/HeatBuffer.html#min_working_temperature
                startup_energy = machine['heat_buffer']['specific_heat_raw'] * Fraction(machine['heat_buffer']['min_working_temperature'] - machine['heat_buffer']['default_temperature']).limit_denominator()
                cost['fuel_name'] = startup_energy / fuel_value

            yield UncompiledConstruct(ident, CompressedVector(), vector, 
                                    {'speed': [], 'productivity': [], 'consumption': [], 'pollution': []}, [], 0, 
                                    base_inputs, cost, machine['limit'], machine)

def valid_lab(machine: dict, technology: dict) -> bool:
    """Calculates if a lab is able to be used to research a technology

    Parameters
    ----------
    machine : dict
        A crafting machine https://lua-api.factorio.com/latest/prototypes/CraftingMachinePrototype.html
    technology : dict
        A technology https://lua-api.factorio.com/latest/prototypes/TechnologyPrototype.html

    Returns
    -------
    bool
        If the lab is able to be used to research a technology
    """
    for tool in technology['base_inputs'].keys():
        if not tool in machine['inputs']: #https://lua-api.factorio.com/latest/prototypes/LabPrototype.html#inputs
            return False
    return True

def generate_lab_construct(machine: dict, technology: dict, instance: FactorioInstance) -> Generator[UncompiledConstruct, None, None]:
    """Generates UncompiledConstructs for a lab researching a technology

    Parameters
    ----------
    machine : dict
        A lab https://lua-api.factorio.com/latest/prototypes/LabPrototype.html
    technology : dict
        A technology https://lua-api.factorio.com/latest/prototypes/TechnologyPrototype.html
    instance : FactorioInstance
        FactorioInstance to use

    Yields
    ------
    Generator[UncompiledConstruct, None, None]
        Constructs for the lab
    """
    logging.log(5, "Generating uncompiled constructs for: %s in %s", technology['name'], machine['name'])
    
    for fuel_name, fuel_value, fuel_burnt_result in fuels_from_energy_source(machine['energy_source'], instance):
        for temperature_setting in temperature_setting_generator(fuel_name, instance, technology):
            yield generate_fueled_construct_helper(machine, technology, (fuel_name, fuel_value, fuel_burnt_result), temperature_setting, ["laboratory-productivity", "laboratory-speed"])

def generate_rocket_result_construct(machine: dict, item: dict, instance: FactorioInstance) -> Generator[UncompiledConstruct, None, None]:
    """Generates UncompiledConstruct of a rocket being launched

    Parameters
    ----------
    machine : dict
        A rocket silo https://lua-api.factorio.com/latest/prototypes/RocketSiloPrototype.html
    item : dict
        An item with (a) rocket launch product(s) https://lua-api.factorio.com/latest/prototypes/ItemPrototype.html
    instance : FactorioInstance
        FactorioInstance to use

    Yields
    ------
    Generator[UncompiledConstruct, None, None]
        Constructs for the item in a rocket
    """
    for fuel_name, fuel_value, fuel_burnt_result in fuels_from_energy_source(machine['energy_source'], instance): #https://lua-api.factorio.com/latest/prototypes/RocketSiloPrototype.html#energy_source
        for temperature_settings in temperature_setting_generator(fuel_name, instance):
            ident = " & ".join([prod['name'] for prod in item['rocket_launch_products']]) + " from launch of "+item['name']+" in "+machine['name']+" via "+fuel_name

            vector = CompressedVector({fuel_name: -1 * (machine['energy_usage_raw'] + machine['active_energy_usage_raw']) / fuel_value,
                                       item['name']: Fraction(-1),
                                       instance.data_raw['recipe'][machine['fixed_recipe']]['result']: Fraction(-1 * machine['rocket_parts_required'])})
            if not fuel_burnt_result is None:
                vector[fuel_burnt_result] = -1 * vector[fuel_name]

            drain = calculate_drain(machine['energy_source'], (machine['energy_usage_raw'] + machine['active_energy_usage_raw']), fuel_name, fuel_value)

            for prod in item['rocket_launch_products']:
                vector[prod['name']] = average_result_amount(prod)
            fix_temperature_settings(temperature_settings, vector)
            
            effect_effects = {'speed': [], 'productivity': [], 'consumption': []}
            if 'allowed_effects' in machine.keys():
                #https://lua-api.factorio.com/latest/types/EffectTypeLimitation.html
                if 'speed' in machine['allowed_effects']:
                    effect_effects['speed'] += [] #speed effects nothing
                if 'productivity' in machine['allowed_effects']:
                    effect_effects['productivity'] += [] #productivity effects nothing
                if 'consumption' in machine['allowed_effects']:
                    if fuel_name in temperature_settings.keys():
                        effect_effects['consumption'] += [fuel_name+'@'+str(temperature_settings[fuel_name])]
                    else:
                        effect_effects['consumption'] += [fuel_name] #consumption only affects fuel
                    if not fuel_burnt_result is None: #https://lua-api.factorio.com/latest/prototypes/ItemPrototype.html#burnt_result
                        effect_effects['consumption'] += [fuel_burnt_result]
            
            max_internal_mods, allowed_modules = module_specification_calculation(machine, {'allowed_modules': list(instance.data_raw['module'].values())})

            base_inputs = CompressedVector({item['name']: -1 * Fraction(1)})
            base_inputs[fuel_name] = (-1 * (machine['energy_usage_raw'] + machine['active_energy_usage_raw']) / fuel_value) * Fraction(3684, 60) #https://wiki.factorio.com/Rocket_silo#Conclusions
            fix_temperature_settings(temperature_settings, base_inputs)
                
            cost = CompressedVector({machine['name']: Fraction(1)})

            limit = machine['limit']

            yield UncompiledConstruct(ident, drain, vector, effect_effects, allowed_modules, max_internal_mods, base_inputs, cost, limit, machine)

def generate_all_constructs(instance: FactorioInstance) -> tuple[UncompiledConstruct, ...]:
    """Generates UncompiledConstructs for all machines in the version of the game represented in data

    Parameters
    ----------
    instance : FactorioInstance
        FactorioInstance to use

    Returns
    -------
    tuple[UncompiledConstruct, ...]
        All the UncompiledConstructs

    Raises
    ------
    ValueError
        If a machine cannot be typed
    """
    logging.log(5, "Beginning the generation of all UncompiledConstructs from data.raw")

    all_uncompiled_constructs = []

    for building_type in ['boiler', 'burner-generator', 'offshore-pump', 'reactor', 'generator', 'furnace', 'mining-drill', 'solar-panel', 'rocket-silo', 'assembling-machine', 'lab']:
        logging.debug("Starting construct generation of machines in category: %s", building_type)
        for machine in instance.data_raw[building_type].values():
            logging.log(5, "Starting processing of machine: %s", machine['name'])

            if not machine['name'] in instance.data_raw['recipe'].keys():
                logging.log(5, "%s is a fake machine because you cant build it. Skipping.", machine['name'])
                continue


            if machine['type']=='rocket-silo':
                for item in instance.data_raw['item'].values():
                    if 'rocket_launch_products' in item.keys():
                        for construct in generate_rocket_result_construct(machine, item, instance):
                            all_uncompiled_constructs.append(construct)


            if machine['type']=='assembling-machine' or machine['type']=='rocket-silo' or machine['type']=='furnace': #all children of https://lua-api.factorio.com/latest/prototypes/CraftingMachinePrototype.html
                if 'fixed-recipe' in machine.keys(): #https://lua-api.factorio.com/latest/prototypes/AssemblingMachinePrototype.html#fixed_recipe
                    logging.log(5, "%s has a fixed recipe named %s", machine['name'], machine['fixed-recipe'])
                    for construct in generate_crafting_constructs(machine, instance.data_raw['recipe'][machine['fixed-recipe']], instance):
                        all_uncompiled_constructs.append(construct)

                for recipe in instance.data_raw['recipe'].values():
                    if valid_crafting_machine(machine, recipe, instance):
                        for construct in generate_crafting_constructs(machine, recipe, instance):
                            all_uncompiled_constructs.append(construct)
            

            elif machine['type']=='boiler': #https://lua-api.factorio.com/latest/prototypes/BoilerPrototype.html
                for construct in generate_boiler_machine_constructs(machine, instance):
                    all_uncompiled_constructs.append(construct)
            

            elif machine['type']=='burner-generator': #https://lua-api.factorio.com/latest/prototypes/BurnerGeneratorPrototype.html
                for construct in generate_burner_generator_constructs(machine, instance):
                    all_uncompiled_constructs.append(construct)
            

            elif machine['type']=='generator': #https://lua-api.factorio.com/latest/prototypes/GeneratorPrototype.html
                for construct in generate_generator_constructs(machine, instance):
                    all_uncompiled_constructs.append(construct)
            

            elif machine['type']=='mining-drill': #https://lua-api.factorio.com/latest/prototypes/MiningDrillPrototype.html
                for fam in generate_mining_drill_constructs(machine, instance):
                    all_uncompiled_constructs.append(fam)
            

            elif machine['type']=='offshore-pump': #https://lua-api.factorio.com/latest/prototypes/OffshorePumpPrototype.html
                all_uncompiled_constructs.append(UncompiledConstruct(machine['name'], CompressedVector(), 
                                                                     CompressedVector({machine['fluid']: Fraction(60 * machine['pumping_speed']).limit_denominator()}), #https://lua-api.factorio.com/latest/prototypes/OffshorePumpPrototype.html#pumping_speed
                                                                     {'speed': [], 'productivity': [], 'consumption': [], 'pollution': []}, [], 0,
                                                                     CompressedVector(), CompressedVector({machine['name']: Fraction(1)}), machine['limit'], machine))


            elif machine['type']=='reactor': #https://lua-api.factorio.com/latest/prototypes/ReactorPrototype.html
                for construct in generate_reactor_constructs(machine, instance):
                    all_uncompiled_constructs.append(construct)


            elif machine['type']=='solar-panel': #https://lua-api.factorio.com/latest/prototypes/SolarPanelPrototype.html
                for accumulator in instance.data_raw['accumulator'].values():
                    #For more information on these calculations see https://forums.factorio.com/viewtopic.php?f=5&t=5594
                    total_day_time = (DAYTIME_VARIABLES['daytime'] + DAYTIME_VARIABLES['nighttime'] + 2 * DAYTIME_VARIABLES['dawntime/dusktime'])
                    energy_gain_time = (DAYTIME_VARIABLES['daytime'] + DAYTIME_VARIABLES['dawntime/dusktime'])
                    energy_backup_factor = (DAYTIME_VARIABLES['nighttime'] + DAYTIME_VARIABLES['dawntime/dusktime'] * energy_gain_time / total_day_time) / total_day_time
                    all_uncompiled_constructs.append(UncompiledConstruct(machine['name']+" with "+accumulator['name'], CompressedVector(), 
                                                                         CompressedVector({'electric': machine['production_raw'] * energy_gain_time/total_day_time}), 
                                                                         {'speed': [], 'productivity': [], 'consumption': [], 'pollution': []}, [], 0,
                                                                         CompressedVector(), 
                                                                         CompressedVector({machine['name']: Fraction(1), 
                                                                                           accumulator['name']: energy_gain_time * energy_backup_factor * machine['production_raw'] / accumulator['energy_source']['buffer_capacity_raw']}), 
                                                                         machine['limit'] + accumulator['limit'], machine))


            elif machine['type']=='lab': #https://lua-api.factorio.com/latest/prototypes/LabPrototype.html
                for technology in instance.data_raw['technology'].values():
                    if valid_lab(machine, technology):
                        for construct in generate_lab_construct(machine, technology, instance):
                            all_uncompiled_constructs.append(construct)


            else:
                raise ValueError("Unknown type %s", machine['type'])

    return tuple(all_uncompiled_constructs)

def generate_manual_constructs(instance: FactorioInstance) -> tuple[ManualConstruct, ...]:
    """Generates all the manual constructs. Hand crafting and pickaxes.

    Parameters
    ----------
    instance : FactorioInstance
        FactorioInstance to use

    Returns
    -------
    tuple[ManualConstruct, ...]
        All the manual constructs
    """    
    all_manual_constructs: list[ManualConstruct] = []
    for recipe in instance.data_raw['recipe'].values():
        hand_craftable = True
        if not recipe['enableable']:
            hand_craftable = False
        if 'crafting_categories' in instance.data_raw['character'] and not recipe['category'] in instance.data_raw['character']['crafting_categories']:
            hand_craftable = False
        if 'hide_from_player_crafting' in recipe.keys() and recipe['hide_from_player_crafting']:
            hand_craftable = False
        for k in recipe['vector'].keys():
            if k in instance.data_raw['fluid'].keys():
                hand_craftable = False

        if hand_craftable:
            all_manual_constructs.append(ManualConstruct(recipe['name']+" hand-crafted", recipe['vector'], recipe['limit'], instance))
    
    for resource in instance.data_raw['resource'].values():
        hand_minable = True
        if 'required_fluid' in resource['minable'].keys():
            hand_minable = False
        if 'mining_categories' in instance.data_raw['character'] and not resource['category'] in instance.data_raw['character']['mining_categories']:
            hand_minable = False
        for k in resource['vector'].keys():
            if k in instance.data_raw['fluid'].keys():
                hand_minable = False
        
        if hand_minable:
            mining_speed = 1
            if 'mining_speed' in instance.data_raw['character'].keys():
                mining_speed = instance.data_raw['character']['mining_speed']
            all_manual_constructs.append(ManualConstruct(resource['name']+" hand-mined", resource['vector'] * mining_speed, resource['limit'], instance))
    
    return tuple(all_manual_constructs)

