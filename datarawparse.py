
import logging
from oldstuff.sparsetensors import *
from utils import *
from globalvalues import *
import typing
from typing import Generator

def fuels_from_energy_source(energy_source: dict, data: dict) -> Generator[tuple[str, Fraction, typing.Optional[str]], None, None]:
    """
    Given a set energy source (https://lua-api.factorio.com/latest/types/EnergySource.html) calculates a list of possible fuels.
    RELEVENT_FLUID_TEMPERATURES must already be populated.
    Note that some fuels leave a burnt result as a leftover when used (https://lua-api.factorio.com/latest/prototypes/ItemPrototype.html#burnt_result)

    Parameters
    ----------
    energy_source:
        An EnergySource instance. https://lua-api.factorio.com/latest/types/EnergySource.html
    data:
        Entire data.raw. https://wiki.factorio.com/Data.raw
    
    Returns
    -------
    allowed_fuels:
        A list of tuples in the form (fuel name, energy density, burnt result)
    """
    effectivity = energy_source['effectivity']

    if energy_source['type']=='electric': #https://lua-api.factorio.com/latest/types/ElectricEnergySource.html
        return [('electric', 1, None)]
    
    elif energy_source['type']=='burner': #https://lua-api.factorio.com/latest/types/BurnerEnergySource.html
        if 'fuel_categories' in energy_source.keys():
            return [(item['name'], item['fuel_value_raw'] * effectivity, item['burnt_result'] if 'burnt_result' in item.keys() else None) for item in data['item'].values() 
                    if 'fuel_category' in item.keys() and item['fuel_category'] in energy_source['fuel_categories']]
        elif 'fuel_category' in energy_source.keys():
            return [(item['name'], item['fuel_value_raw'] * effectivity, item['burnt_result'] if 'burnt_result' in item.keys() else None) for item in data['item'].values() 
                    if 'fuel_category' in item.keys() and item['fuel_category']==energy_source['fuel_category']]
        else:
            raise ValueError("Category-less burner energy source: "+str(energy_source))
        
    elif energy_source['type']=='heat': #https://lua-api.factorio.com/latest/types/HeatEnergySource.html
        return [('heat', 1, None)]
    
    elif energy_source['type']=='fluid': #https://lua-api.factorio.com/latest/types/FluidEnergySource.html
        if 'burns_fluid' in energy_source.keys(): #https://lua-api.factorio.com/latest/types/FluidEnergySource.html#burns_fluid
            if 'filter' in energy_source['fluid_box'].keys(): #https://lua-api.factorio.com/latest/types/FluidBox.html#filter
                return [(energy_source['fluid_box']['filter'], data['fluid'][energy_source['fluid_box']['filter']]['fuel_value'] * effectivity, None)]
            else:
                return [(fluid['name'], fluid['fuel_value'] * effectivity, None) for fluid in data['fluid'].values()]
        else:
            if not 'filter' in energy_source['fluid_box'].keys():
                raise ValueError("Non-burning fluid energy source without filter: "+str(energy_source))
            fluid = energy_source['fluid_box']['filter'] #https://lua-api.factorio.com/latest/types/FluidBox.html#filter
            return [(fluid+'@'+temp, RELEVENT_FLUID_TEMPERATURES[fluid][temp] * effectivity, None) 
                    for temp in RELEVENT_FLUID_TEMPERATURES[fluid].keys() 
                    if temp <= energy_source['maximum_temperature']] #there is some detail to maximum_temperature but this doesn't effect energy density and fuel calcs and is outside the scope of this function (see https://lua-api.factorio.com/latest/types/FluidEnergySource.html#maximum_temperature)
            
    elif energy_source['type']=='void': #https://lua-api.factorio.com/latest/types/VoidEnergySource.html
        raise ValueError("Unsupported energy source (void): "+str(energy_source)) #This is a non-standard source that isn't supported because I don't really understand it
    else:
        raise ValueError("Unrecognized energy source: "+str(energy_source))

def calculate_drain(energy_source: dict, energy_usage: Fraction, fuel_name: str, fuel_value: Fraction) -> CompressedVector:
    """
    Calculates a drain vector (dict) from an energy source

    Parameters
    ----------
    energy_source:
        An EnergySource instance. https://lua-api.factorio.com/latest/types/EnergySource.html
    energy_usage:
        Baseline energy usage for default drain calculation. https://lua-api.factorio.com/latest/types/Energy.html
    fuel_name:
        Name of fuel being used for the EnergySource
    fuel_value:
        Energy density of fuel being used for the EnergySource
    
    Returns
    -------
    drain:
        A dict with the drain of the energy source
    """
    drain = CompressedVector()
    if 'drain' in energy_source: #https://lua-api.factorio.com/latest/types/ElectricEnergySource.html#drain some other sources have drain option too
        drain[fuel_name] = energy_source['drain_raw'] / fuel_value
    elif fuel_name=='electric': #electric sources have a default 1/30th drain
        drain[fuel_name] = (energy_usage / 30) / fuel_value
    return drain

def link_techs(data: dict) -> None:
    """
    Links the technologies from data.raw to eachother to the recipes, to the machines, and to the modules and all inbetween in all the funs ways. 
    Modifies data.raw directly.

    Parameters
    ----------
    data:
        Entire data.raw. https://wiki.factorio.com/Data.raw
    """
    enabled_technologies = filter(lambda tech: not 'enabled' in tech.keys() or tech['enabled'], data['technology'].values())
    #https://lua-api.factorio.com/latest/prototypes/TechnologyPrototype.html#enabled

    logging.debug("Beginning the linking of technologies.")
    for tech in enabled_technologies:
        tech['prereq_set'] = [] #will eventually contain technologies that are direct requirements to unlocking this one
        tech['prereq_of'] = [] #will eventually contain technologies that this is a direct unlock requirement of
        tech['all_prereq'] = [] #will eventually contain ALL technologies that have to be researched to research this one
        tech['enabled_recipes'] = [] #will eventually contain all recipes that researching this technology will unlock

    logging.debug("Compiling first order prerequisite.")
    """
    #Old version:
    for tech in data['technology'].values():
        if not 'enabled' in tech.keys() or tech['enabled']:
            if 'prerequisites' in tech.keys():
                for prereq in tech['prerequisites']:
                    for otech in data['technology'].values():
                        if not 'enabled' in otech.keys() or otech['enabled']:
                            if otech['name']==prereq:
                                tech['prereq_set'].append(otech)
                                otech['prereq_of'].append(tech)
    """
    for tech1, tech2 in itertools.permutations(enabled_technologies, 2):
        if 'prerequisites' in tech1.keys(): #https://lua-api.factorio.com/latest/prototypes/TechnologyPrototype.html#prerequisites
            if tech2['name'] in tech1['prerequisites']: #https://lua-api.factorio.com/latest/types/TechnologyID.html
                tech1['prereq_set'].append(tech2)
                tech2['prereq_of'].append(tech1)
    
    def recursive_prereqs(tech):
        tech_set = [tech]
        for st in tech['prereq_set']:
            tech_set.append(st)
            tech_set = list_union(tech_set, recursive_prereqs(st)) #we cant freeze elements for hashing so we have to do list based union
        return tech_set
    for tech in enabled_technologies:
        tech['all_prereq'] = recursive_prereqs(tech)

    logging.debug("Linking of all recipes to their technologies.")
    for recipe in data['recipe'].values(): #https://lua-api.factorio.com/latest/prototypes/RecipePrototype.html
        recipe_techs = []
        for tech in enabled_technologies:
            if 'effects' in tech.keys():
                for effect in tech['effects']: #https://lua-api.factorio.com/latest/prototypes/TechnologyPrototype.html#effects
                    if effect['type']=='unlock-recipe' and effect['recipe']==recipe['name']: #https://lua-api.factorio.com/latest/types/UnlockRecipeModifier.html
                        recipe_techs.append(tech['all_prereq']) #each element of recipe_techs will be a list representing a combination of techs that lets the recipe be used
                        tech['enabled_recipes'].append(recipe)
        recipe.update({'limit': TechnologicalLimitation(recipe_techs)})

    logging.debug("Linking of all machines to their technologies.")
    for cata in MACHINE_CATEGORIES: #https://lua-api.factorio.com/latest/prototypes/EntityWithOwnerPrototype.html
        for machine in data[cata].values(): 
            if machine['name'] in data['recipe'].keys():
                machine['limit'] = data['recipe'][machine['name']]['limit'] #highjack the recipe's link
                
    logging.debug("Linking of all modules to their technologies.")
    for module in data['module'].values(): #https://lua-api.factorio.com/latest/prototypes/ModulePrototype.html
        if module['name'] in data['recipe'].keys():
            module['limit'] = data['recipe'][module['name']]['limit'] #highjack the recipe's link

def standardize_power(data: dict) -> None:
    """
    Standardizes all power usages and values across all machine types and items/fluids. 
    Modifies data.raw directly to add new "_raw" values to power usage and value.

    Parameters
    ----------
    data:
        Entire data.raw. https://wiki.factorio.com/Data.raw
    """
    logging.debug("Beginning the standardization of power. This adds a raw version of many energy values.")
    
    for crafting_machine_type in ['assembling-machine', 'rocket-silo', 'furnace']:
        for machine in data[crafting_machine_type].values(): #https://lua-api.factorio.com/latest/prototypes/CraftingMachinePrototype.html
            machine['energy_usage_raw'] = convert_value_to_base_units(machine['energy_usage']) #https://lua-api.factorio.com/latest/prototypes/CraftingMachinePrototype.html#energy_usage
            #add_complex_type(machine['energy_source'], data)
    
    for machine in data['boiler'].values(): #https://lua-api.factorio.com/latest/prototypes/BoilerPrototype.html
        machine['energy_consumption_raw'] = convert_value_to_base_units(machine['energy_consumption']) #https://lua-api.factorio.com/latest/prototypes/BoilerPrototype.html#energy_consumption
        #add_complex_type(machine['energy_source'], data)
    
    for machine in data['burner-generator'].values(): #https://lua-api.factorio.com/latest/prototypes/BurnerGeneratorPrototype.html
        machine['max_power_output_raw'] = convert_value_to_base_units(machine['max_power_output']) #https://lua-api.factorio.com/latest/prototypes/BurnerGeneratorPrototype.html#max_power_output
        if not 'type' in machine['burner'].keys():  #https://lua-api.factorio.com/latest/types/BurnerEnergySource.html
            machine['burner'].update({'type': 'burner'})
        #add_complex_type(machine['burner'], data)
        #add_complex_type(machine['energy_source'], data)
    
    for machine in data['generator'].values(): #https://lua-api.factorio.com/latest/prototypes/GeneratorPrototype.html
        if 'max_power_output' in machine.keys(): #https://lua-api.factorio.com/latest/prototypes/GeneratorPrototype.html#max_power_output
            machine['max_power_output_raw'] = convert_value_to_base_units(machine['max_power_output'])
        else: #https://lua-api.factorio.com/latest/prototypes/GeneratorPrototype.html#fluid_box
            machine['max_power_output_raw'] = 60*machine['fluid_usage_per_tick']*\
                                              ((100-data['fluid']['water']['default_temperature'])*convert_value_to_base_units(data['fluid']['water']['heat_capacity'])+\
                                               (machine['maximum_temperature']-100)*convert_value_to_base_units(data['fluid']['steam']['heat_capacity']))
        #add_complex_type(machine['energy_source'], data)
    
    for machine in data['mining-drill'].values(): #https://lua-api.factorio.com/latest/prototypes/MiningDrillPrototype.html
        machine['energy_usage_raw'] = convert_value_to_base_units(machine['energy_usage'])
        #add_complex_type(machine['energy_source'], data)
    
    for machine in data['reactor'].values(): #https://lua-api.factorio.com/latest/prototypes/ReactorPrototype.html
        machine['consumption_raw'] = convert_value_to_base_units(machine['consumption'])
        #add_complex_type(machine['energy_source'], data)
        if not 'type' in machine['heat_buffer'].keys():
            machine['heat_buffer'].update({'type': 'heat'})
        #add_complex_type(machine['heat_buffer'], data)
    
    for machine in data['solar-panel'].values(): #https://lua-api.factorio.com/latest/prototypes/SolarPanelPrototype.html
        machine['production_raw'] = convert_value_to_base_units(machine['production'])
        #add_complex_type(machine['energy_source'], data)
    
    for item in data['item'].values(): #https://lua-api.factorio.com/latest/prototypes/ItemPrototype.html
        if 'fuel_value' in item.keys():
            item['fuel_value_raw'] = convert_value_to_base_units(item['fuel_value'])

    for fluid in data['fluid'].values(): #https://lua-api.factorio.com/latest/prototypes/FluidPrototype.html
        if 'fuel_value' in fluid.keys():
            fluid['fuel_value_raw'] = convert_value_to_base_units(fluid['fuel_value'])

def prep_resources(data: dict) -> None:
    """
    Makes sure the 'category' key exists in resources and compiles a link to each resource under a category into that category.

    Parameters
    ----------
    data:
        Entire data.raw. https://wiki.factorio.com/Data.raw
    """
    logging.debug("Doing some slight edits to resources, linking them to their categories, and factoring in mining properties.")
    for resource in data['resource'].values():
        if not 'category' in resource.keys():
            resource.update({'category': 'basic-solid'})
        #https://lua-api.factorio.com/latest/types/MinableProperties.html#required_fluid

    for cata in data['resource-category'].values():
        cata.update({'resource_list': []})
        for resource in data['resource'].values():
            if cata['name'] == resource['category']:
                cata['resource_list'].append(resource)

def vectorize_recipes(data: dict) -> None:
    """
    Adds a base_inputs and vector component to each recipe. The vector component represents how the recipe function.
    While the base_inputs are values stored for catalyst cost calculations.

    Parameters
    ----------
    data:
        Entire data.raw. https://wiki.factorio.com/Data.raw
    """
    logging.debug("Beginning the vectorization of recipes.")
    for recipe in data['recipe'].values(): #https://lua-api.factorio.com/latest/prototypes/RecipePrototype.html
        changes = CompressedVector()

        if COST_MODE in recipe.keys(): #https://lua-api.factorio.com/latest/prototypes/RecipePrototype.html#normal or https://lua-api.factorio.com/latest/prototypes/RecipePrototype.html#expensive
            recipe_definition = recipe[COST_MODE]
        else: #https://lua-api.factorio.com/latest/prototypes/RecipePrototype.html#ingredients
            recipe_definition = recipe
            

        for ingred in recipe_definition['ingredients']: #https://lua-api.factorio.com/latest/types/IngredientPrototype.html both item and fluid have similar structure
            if isinstance(ingred, dict): #full definition
                fixed_name = ingred['name']

                if 'minimum_temperature' in ingred.keys():
                    fixed_name += '@'+str(ingred['minimum_temperature'])+'-'+str(ingred['maximum_temperature'])

                changes.update({ingred['name']: -1*ingred['amount']})

            else: #shortened (item only) definition (a list)
                changes.update({ingred[0]: -1*ingred[1]})


        if 'results' in recipe_definition.keys(): #https://lua-api.factorio.com/latest/prototypes/RecipePrototype.html#results both item and fluid have similar structure
            for result in recipe_definition['results']: 
                if isinstance(result, dict):
                    fixed_name = result['name']

                    if 'temperature' in result.keys(): #https://lua-api.factorio.com/latest/types/FluidProductPrototype.html#temperature
                        fluid = data['fluid'][result['name']]
                        fixed_name += '@'+str(result['temperature'])
                        if not result['name'] in RELEVENT_FLUID_TEMPERATURES.keys():
                            RELEVENT_FLUID_TEMPERATURES.update({result['name']: {}})
                        if not result['temperature'] in RELEVENT_FLUID_TEMPERATURES[result['name']].keys():
                            RELEVENT_FLUID_TEMPERATURES[result['name']].update({result['temperature']: (result['temperature'] - fluid['default_temperature']) * convert_value_to_base_units(fluid['heat_capacity'])}) #https://lua-api.factorio.com/latest/prototypes/FluidPrototype.html#heat_capacity

                    if 'amount' in result.keys(): #https://lua-api.factorio.com/latest/types/FluidProductPrototype.html#amount works the same for items
                        changes.update({fixed_name: result['amount']})
                    else: #https://lua-api.factorio.com/latest/types/FluidProductPrototype.html#amount_min works the same for items
                        changes.update({fixed_name: .5*(result['amount_max']+result['amount_min'])})

                    if 'probability' in result.keys(): #https://lua-api.factorio.com/latest/types/FluidProductPrototype.html#probability works the same for items
                        changes[fixed_name] *= result['probability']

                else: #shortened (item only) definition (a list)
                    changes.update({result[0]: result[1]})
        else: #https://lua-api.factorio.com/latest/prototypes/RecipePrototype.html#result
            changes.update({recipe_definition['result']: recipe_definition['result_count'] if 'result_count' in recipe_definition.keys() else 1}) #https://lua-api.factorio.com/latest/prototypes/RecipePrototype.html#result_count

        base_inputs = CompressedVector({c: v for c, v in changes.items() if v < 0}) #store singular run inputs for catalyst calculations.

        recipe.update({'vector': (1 / recipe['energy_required']) * changes,
                       'base_inputs': base_inputs})

def vectorize_resources(data: dict) -> None:
    """
    Adds a base_inputs and vector component to each resource. The vector component represents how the mining of the resource acts.
    While the base_inputs are values stored for catalyst cost calculations.

    Parameters
    ----------
    data:
        Entire data.raw. https://wiki.factorio.com/Data.raw
    """
    logging.debug("Beginning the vectorization of resources.")
    for resource in data['resource'].values(): #https://lua-api.factorio.com/latest/prototypes/ResourceEntityPrototype.html
        changes = CompressedVector()

        mining_definition = resource['minable']

        if 'required_fluid' in mining_definition: #https://lua-api.factorio.com/latest/types/MinableProperties.html#required_fluid
            changes.update({mining_definition['required_fluid']: -1 * mining_definition['fluid_amount']}) #https://lua-api.factorio.com/latest/types/MinableProperties.html#fluid_amount


        if 'results' in mining_definition.keys(): #https://lua-api.factorio.com/latest/types/MinableProperties.html#results
            for result in mining_definition['results']: 
                if isinstance(result, dict):
                    fixed_name = result['name']

                    if 'temperature' in result.keys(): #https://lua-api.factorio.com/latest/types/FluidProductPrototype.html#temperature
                        fluid = data['fluid'][result['name']]
                        fixed_name += '@'+str(result['temperature'])
                        if not result['name'] in RELEVENT_FLUID_TEMPERATURES.keys():
                            RELEVENT_FLUID_TEMPERATURES.update({result['name']: {}})
                        if not result['temperature'] in RELEVENT_FLUID_TEMPERATURES[result['name']].keys():
                            RELEVENT_FLUID_TEMPERATURES[result['name']].update({result['temperature']: (result['temperature'] - fluid['default_temperature']) * convert_value_to_base_units(fluid['heat_capacity'])}) #https://lua-api.factorio.com/latest/prototypes/FluidPrototype.html#heat_capacity

                    if 'amount' in result.keys(): #https://lua-api.factorio.com/latest/types/FluidProductPrototype.html#amount works the same for items
                        changes.update({fixed_name: result['amount']})
                    else: #https://lua-api.factorio.com/latest/types/FluidProductPrototype.html#amount_min works the same for items
                        changes.update({fixed_name: .5*(result['amount_max']+result['amount_min'])})

                    if 'probability' in result.keys(): #https://lua-api.factorio.com/latest/types/FluidProductPrototype.html#probability works the same for items
                        changes[fixed_name] *= result['probability']

                else: #shortened (item only) definition (a list)
                    changes.update({result[0]: result[1]})
        else: #https://lua-api.factorio.com/latest/types/MinableProperties.html#result
            changes.update({mining_definition['result']: mining_definition['count']}) #https://lua-api.factorio.com/latest/types/MinableProperties.html#count


        base_inputs = CompressedVector({c: v for c, v in changes.items() if v < 0}) #store singular run inputs for catalyst calculations.

        resource.update({'vector': (1.0/mining_definition['mining_time']) * changes,
                         'base_inputs': base_inputs})

def vectorize_technologies(data: dict) -> None:
    """
    Adds a vector component to each technology. The vector component represents how labs researching a technology function.

    Parameters
    ----------
    data:
        Entire data.raw. https://wiki.factorio.com/Data.raw
    """
    logging.debug("Beginning the vectorization of technologies.")
    for technology in data['technology'].values(): #https://lua-api.factorio.com/latest/prototypes/TechnologyPrototype.html
        changes = CompressedVector()

        if COST_MODE in technology.keys(): #https://lua-api.factorio.com/latest/prototypes/TechnologyPrototype.html#normal or https://lua-api.factorio.com/latest/prototypes/TechnologyPrototype.html#expensive
            cost_definition = technology[COST_MODE]
        else: #https://lua-api.factorio.com/latest/prototypes/TechnologyPrototype.html#unit
            cost_definition = technology

        for ingred in cost_definition['unit']['ingredients']: #https://lua-api.factorio.com/latest/types/TechnologyData.html#unit 
            #https://lua-api.factorio.com/latest/types/TechnologyUnit.html
            #https://lua-api.factorio.com/latest/types/IngredientPrototype.html
            if isinstance(ingred, dict): #full definition
                fixed_name = ingred['name']

                if 'minimum_temperature' in ingred.keys():
                    fixed_name += '@'+str(ingred['minimum_temperature'])+'-'+str(ingred['maximum_temperature'])

                changes.update({ingred['name']: -1*ingred['amount']})

            else: #shortened (item only) definition (a list)
                changes.update({ingred[0]: -1*ingred[1]})
        
        if 'count' in cost_definition['unit'].keys():
            changes = Fraction(cost_definition['unit']['count']) * changes
        else:
            logging.warning("Formulaic technological counts aren't supported yet. TODO")

        technology['base_inputs'] = CompressedVector({c: v for c, v in changes.items() if v < 0}) #store inputs for lab matching later.
        
        changes.update({technology['name']}) #The result of a technology vector is the researched technology. Enabled recipies are calculated as limits of these results.

        technology['vector'] = changes / (Fraction(cost_definition['time']) * Fraction(cost_definition['unit']['count']))

def link_modules(data: dict) -> None:
    """
    Adds the allowed_module component to each recipe and resource mining type representing which modules can be used in a machine running said operation.
    Also populates global MODULE_REFERENCE

    Parameters
    ----------
    data:
        Entire data.raw. https://wiki.factorio.com/Data.raw
    """
    logging.debug("Starting the linking of modules into the recipe and resource lists. %d recipes are detected, %d resources are detected, and %d modules are detected. Info will be added to the each recipe and resource under the \'allowed_modules\' key.",
        len(data['recipe'].values()),
        len(data['resource'].values()),
        len(data['module'].values()))
    for recipe in data['recipe'].values(): #https://lua-api.factorio.com/latest/prototypes/RecipePrototype.html
        recipe.update({'allowed_modules': []})
    for resource in data['resource'].values(): #https://lua-api.factorio.com/latest/prototypes/ResourceEntityPrototype.html
        resource.update({'allowed_modules': []})
    for module in data['module'].values(): #https://lua-api.factorio.com/latest/prototypes/ModulePrototype.html
        if 'limitation' in module.keys(): #https://lua-api.factorio.com/latest/prototypes/ModulePrototype.html#limitation
            for recipe_name in module['limitation']:
                data['recipe'][recipe_name]['allowed_modules'].append(module)

        elif 'limitation_blacklist' in module.keys(): #https://lua-api.factorio.com/latest/prototypes/ModulePrototype.html#limitation_blacklist
            for recipe in data['recipe'].values():
                if not recipe['name'] in module['limitation_blacklist']:
                    recipe['allowed_modules'].append(module)
            for resource in data['resource'].values(): #can a resource be blacklisted?
                resource['allowed_modules'].append(module)

        else: #allowed in everything
            for recipe in data['recipe'].values():
                recipe['allowed_modules'].append(module)
            for resource in data['resource'].values():
                resource['allowed_modules'].append(module)

        MODULE_REFERENCE[module['name']] = module

def set_defaults_and_normalize(data: dict) -> None:
    """
    Sets the defaults of various optional elements that are called later. Including:
    recipe's energy_required (default: .5)
    recipe's result_count (default: 1)
    resource's minable count (default: 1)
    machine's energy_source effectivity (default: 1)

    Additionally normalizes terms across machines:
    furnace's, rocket-silo's, and assembling-machines's 'crafting_speed' is added to 'speed' term
    mining-drill's 'mining_speed' is added to 'speed' term
    lab's 'researching_speed' is added to 'speed' term

    Additionally normalizes terms across recipe-likes:
    recipe's 'energy_required' is added to 'time_multiplier' term
    resources's 'minable' 'mining-time' is added to 'time_multiplier' term
    technology's 'unit' 'time' is added to 'time_multiplier' term

    Parameters
    ----------
    data:
        Entire data.raw. https://wiki.factorio.com/Data.raw
    """
    for recipe in data['recipe'].values():
        if not 'energy_required' in recipe.keys(): #https://lua-api.factorio.com/latest/prototypes/RecipePrototype.html#energy_required
            recipe['energy_required'] = .5
        if ('result' in recipe.keys()) and (not 'result_count' in recipe.keys()): #https://lua-api.factorio.com/latest/prototypes/RecipePrototype.html#result_count
            recipe['result_count'] = 1
    for resource in data['resource'].values():
        if 'result' in resource['minable'] and not 'count' in resource['minable']:
            resource['minable']['count'] = 1 #set the default: https://lua-api.factorio.com/latest/types/MinableProperties.html#count
    for ref in ['boiler', 'burner-generator', 'offshore-pump', 'reactor', 'generator', 'furnace', 'mining-drill', 'solar-panel', 'rocket-silo', 'assembling-machine']:
        for machine in data[ref].values():
            if ('energy_source' in machine.keys()) and (not 'effectivity' in machine['energy_source'].keys()):
                machine['energy_source']['effectivity'] = 1

    for ref in ['furnace', 'rocket-silo', 'assembling-machine']:
        for machine in data[ref].values():
            machine['speed'] = Fraction(machine['crafting_speed'])
    for machine in data['mining-drill'].values():
        machine['speed'] = Fraction(machine['mining_speed'])
    for machine in data['lab'].values():
        machine['speed'] = Fraction(machine['researching_speed'])

    for recipe in data['recipe'].values():
        if COST_MODE in recipe:
            recipe['time_multiplier'] = Fraction(recipe[COST_MODE]['energy_required'])
        else:
            recipe['time_multiplier'] = Fraction(recipe['energy_required'])
    for resource in data['resource'].values():
        resource['time_multiplier'] = Fraction(resource['minable']['mining_time'])
    for technology in data['technology'].values():
        if COST_MODE in technology:
            technology['time_multiplier'] = Fraction(technology[COST_MODE]['unit']['time'])
        else:
            technology['time_multiplier'] = Fraction(technology['unit']['time'])

def complete_premanagement(data: dict) -> None:
    """
    Does all the premanagment steps on data.raw in an appropriate order.

    Parameters
    ----------
    data:
        Entire data.raw. https://wiki.factorio.com/Data.raw
    """
    logging.debug("Beginning the premanagement of the game data.raw object. This will link technologies, standardize power, recategorize resources, simplify recipes, and link modules.")
    set_defaults_and_normalize(data)
    link_techs(data)
    standardize_power(data)
    prep_resources(data)
    vectorize_recipes(data)
    vectorize_resources(data)
    vectorize_technologies(data)
    link_modules(data)
