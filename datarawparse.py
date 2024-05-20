from utils import *
from globalsandimports import *

def fuels_from_energy_source(energy_source: dict, data: dict, RELEVENT_FLUID_TEMPERATURES: dict) -> Generator[tuple[str, Fraction, typing.Optional[str]], None, None] | list[tuple[str, Fraction, typing.Optional[str]]]:
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
    RELEVENT_FLUID_TEMPERATURES:
        Dict with keys of fluid names and values of a dict mapping temperatures to energy densities.
    
    Returns
    -------
    allowed_fuels:
        A list of tuples in the form (fuel name, energy density, burnt result)
    """
    effectivity = Fraction(energy_source['effectivity'] if 'effectivity' in energy_source.keys() else 1).limit_denominator()

    if energy_source['type']=='electric': #https://lua-api.factorio.com/latest/types/ElectricEnergySource.html
        return [('electric', Fraction(1), None)]
    
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
        return [('heat', Fraction(1), None)]
    
    elif energy_source['type']=='fluid': #https://lua-api.factorio.com/latest/types/FluidEnergySource.html
        if 'burns_fluid' in energy_source.keys(): #https://lua-api.factorio.com/latest/types/FluidEnergySource.html#burns_fluid
            if 'filter' in energy_source['fluid_box'].keys(): #https://lua-api.factorio.com/latest/types/FluidBox.html#filter
                return [(energy_source['fluid_box']['filter'], data['fluid'][energy_source['fluid_box']['filter']]['fuel_value_raw'] * effectivity, None)]
            else:
                return [(fluid['name'], fluid['fuel_value_raw'] * effectivity, None) for fluid in data['fluid'].values() if 'fuel_value_raw' in fluid.keys()]
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
        drain[fuel_name] = -1 * energy_source['drain_raw'] / fuel_value
    elif fuel_name=='electric': #electric sources have a default 1/30th drain
        drain[fuel_name] = -1 *(energy_usage / 30) / fuel_value
    return drain

def link_techs(data: dict, COST_MODE: str) -> TechnologyTree:
    """
    Links the technologies from data.raw to eachother to the recipes, to the machines, and to the modules and all inbetween in all the funs ways. 
    Modifies data.raw directly.
    Additionally adds 'enableable' key to all recipes to filter recipes that are cheat mode only.

    Parameters
    ----------
    data:
        Entire data.raw. https://wiki.factorio.com/Data.raw
    COST_MODE:
        What cost mode is being used. https://lua-api.factorio.com/latest/concepts.html#DifficultySettings
    """
    enabled_technologies = list(filter(lambda tech: not 'enabled' in tech.keys() or tech['enabled'], data['technology'].values()))
    #https://lua-api.factorio.com/latest/prototypes/TechnologyPrototype.html#enabled

    logging.warning("Beginning the linking of technologies. %d technologies found.", len(enabled_technologies))
    for tech in enabled_technologies:
        tech['prereq_set'] = [] #will eventually contain technologies that are direct requirements to unlocking this one
        tech['prereq_of'] = [] #will eventually contain technologies that this is a direct unlock requirement of
        tech['all_prereq'] = [] #will eventually contain ALL technologies that have to be researched to research this one (recursive prereq_set)
        tech['enabled_recipes'] = [] #will eventually contain all recipes that researching this technology will unlock

    logging.debug("Compiling first order prerequisite.")
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

    tech_tree = TechnologyTree(list(enabled_technologies))
    assert all(['tech_tree_identifier' in tech.keys() for tech in list(filter(lambda tech: not 'enabled' in tech.keys() or tech['enabled'], data['technology'].values()))]) #TODO remove me

    logging.info("Linking of all recipes to their technologies.")
    for recipe in data['recipe'].values(): #https://lua-api.factorio.com/latest/prototypes/RecipePrototype.html
        recipe_techs = []
        if COST_MODE in recipe.keys():
            enableable = (not 'enabled' in recipe[COST_MODE].keys()) or recipe[COST_MODE]['enabled']
        else:
            enableable = (not 'enabled' in recipe.keys()) or recipe['enabled']
        for tech in enabled_technologies:
            if 'effects' in tech.keys():
                for effect in tech['effects']: #https://lua-api.factorio.com/latest/prototypes/TechnologyPrototype.html#effects
                    if effect['type']=='unlock-recipe' and effect['recipe']==recipe['name']: #https://lua-api.factorio.com/latest/types/UnlockRecipeModifier.html
                        enableable = True
                        recipe_techs.append([t['name'] for t in tech['all_prereq']]) #each element of recipe_techs will be a list representing a combination of techs that lets the recipe be used
                        tech['enabled_recipes'].append(recipe)
        recipe['enableable'] = enableable
        recipe['limit'] = TechnologicalLimitation(tech_tree, recipe_techs)

    logging.info("Linking of all machines to their technologies.")
    for cata in ['boiler', 'burner-generator', 'offshore-pump', 'reactor', 'generator', 'furnace', 'mining-drill', 'solar-panel', 'rocket-silo', 'assembling-machine', 'lab']: #https://lua-api.factorio.com/latest/prototypes/EntityWithOwnerPrototype.html
        for machine in data[cata].values():
            machine['limit'] = TechnologicalLimitation(tech_tree)
            for recipe in data['recipe'].values():
                if any([k==machine["name"] and v > 0 for k, v in recipe['vector'].items()]):
                    logging.info("Found machine "+machine['name']+" being made via "+recipe['name']+" which has a limit of "+str(recipe['limit']))
                    machine['limit'] = machine['limit'] + recipe['limit']
                
    logging.info("Linking of all modules to their technologies.")
    for module in data['module'].values(): #https://lua-api.factorio.com/latest/prototypes/ModulePrototype.html
        module['limit'] = TechnologicalLimitation(tech_tree)
        for recipe in data['recipe'].values():
            if any([k==module["name"] and v > 0 for k, v in recipe['vector'].items()]):
                logging.info("Found module "+module['name']+" being made via "+recipe['name']+" which has a limit of "+str(recipe['limit']))
                module['limit'] = module['limit'] + recipe['limit']

    for beacon in data['beacon'].values(): #https://lua-api.factorio.com/latest/prototypes/BeaconPrototype.html
        beacon['limit'] = TechnologicalLimitation(tech_tree)
        for recipe in data['recipe'].values():
            if any([k==beacon["name"] and v > 0 for k, v in recipe['vector'].items()]):
                logging.info("Found module "+beacon['name']+" being made via "+recipe['name']+" which has a limit of "+str(recipe['limit']))
                beacon['limit'] = beacon['limit'] + recipe['limit']
    
    for resource in data['resource'].values(): #https://lua-api.factorio.com/latest/prototypes/ResourceEntityPrototype.html
        resource['limit'] = TechnologicalLimitation(tech_tree)

    for technology in data['technology'].values(): #https://lua-api.factorio.com/latest/prototypes/TechnologyPrototype.html
        """
        Settings technologies to the limit of their previous technologies would make it far harder to produce science factories in the current implementation.
        An interface (and the current tools.py interface does) use the difference of TechnologicalLimitations to properly find which technologies to research.
        """
        technology['limit'] = TechnologicalLimitation(tech_tree)
        #TechnologicalLimitation([t['name'] for t in tech['all_prereq'] if t['name']!=tech['name']])

    return tech_tree
        
def standardize_power(data: dict) -> None:
    """
    Standardizes all power usages and values across all machine types and items/fluids.
    Modifies data.raw directly to add new "_raw" values to power usages and values.
    Memoization step as these values are used many times in construct generation.

    Parameters
    ----------
    data:
        Entire data.raw. https://wiki.factorio.com/Data.raw
    """
    logging.debug("Beginning the standardization of power. This adds a raw version of many energy values.")
    
    for crafting_machine_type in ['assembling-machine', 'rocket-silo', 'furnace']:
        for machine in data[crafting_machine_type].values(): #https://lua-api.factorio.com/latest/prototypes/CraftingMachinePrototype.html
            machine['energy_usage_raw'] = convert_value_to_base_units(machine['energy_usage']) #https://lua-api.factorio.com/latest/prototypes/CraftingMachinePrototype.html#energy_usage
    
    for machine in data['rocket-silo'].values(): #https://lua-api.factorio.com/latest/prototypes/RocketSiloPrototype.html
        machine['active_energy_usage_raw'] = convert_value_to_base_units(machine['active_energy_usage']) #https://lua-api.factorio.com/latest/prototypes/RocketSiloPrototype.html#active_energy_usage

    for machine in data['boiler'].values(): #https://lua-api.factorio.com/latest/prototypes/BoilerPrototype.html
        machine['energy_consumption_raw'] = convert_value_to_base_units(machine['energy_consumption']) #https://lua-api.factorio.com/latest/prototypes/BoilerPrototype.html#energy_consumption
    
    for machine in data['burner-generator'].values(): #https://lua-api.factorio.com/latest/prototypes/BurnerGeneratorPrototype.html
        machine['max_power_output_raw'] = convert_value_to_base_units(machine['max_power_output']) #https://lua-api.factorio.com/latest/prototypes/BurnerGeneratorPrototype.html#max_power_output
        if not 'type' in machine['burner'].keys():  #https://lua-api.factorio.com/latest/types/BurnerEnergySource.html
            machine['burner']['type'] = 'burner'
    
    for machine in data['generator'].values(): #https://lua-api.factorio.com/latest/prototypes/GeneratorPrototype.html
        if 'max_power_output' in machine.keys(): #https://lua-api.factorio.com/latest/prototypes/GeneratorPrototype.html#max_power_output
            machine['max_power_output_raw'] = convert_value_to_base_units(machine['max_power_output'])
        else: #https://lua-api.factorio.com/latest/prototypes/GeneratorPrototype.html#fluid_box
            machine['max_power_output_raw'] = Fraction(60*machine['fluid_usage_per_tick']).limit_denominator()*\
                                              ((100-Fraction(data['fluid']['water']['default_temperature']).limit_denominator())*convert_value_to_base_units(data['fluid']['water']['heat_capacity'])+\
                                               (Fraction(machine['maximum_temperature']).limit_denominator()-100)*convert_value_to_base_units(data['fluid']['steam']['heat_capacity']))
    
    for machine in data['mining-drill'].values(): #https://lua-api.factorio.com/latest/prototypes/MiningDrillPrototype.html
        machine['energy_usage_raw'] = convert_value_to_base_units(machine['energy_usage'])
    
    for machine in data['reactor'].values(): #https://lua-api.factorio.com/latest/prototypes/ReactorPrototype.html
        machine['consumption_raw'] = convert_value_to_base_units(machine['consumption'])
        if not 'type' in machine['heat_buffer'].keys():
            machine['heat_buffer']['type'] = 'heat'
        machine['heat_buffer']['specific_heat_raw'] = convert_value_to_base_units(machine['heat_buffer']['specific_heat'])
    
    for machine in data['solar-panel'].values(): #https://lua-api.factorio.com/latest/prototypes/SolarPanelPrototype.html
        machine['production_raw'] = convert_value_to_base_units(machine['production'])
    
    for machine in data['lab'].values(): #https://lua-api.factorio.com/latest/prototypes/LabPrototype.html
        machine['energy_usage_raw'] = convert_value_to_base_units(machine['energy_usage'])
    
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
            resource['category'] = 'basic-solid'

    for cata in data['resource-category'].values():
        cata['resource_list'] = []
        for resource in data['resource'].values():
            if cata['name'] == resource['category']:
                cata['resource_list'].append(resource)

def vectorize_recipes(data: dict, RELEVENT_FLUID_TEMPERATURES: dict, COST_MODE: str) -> None:
    """
    Adds a base_inputs and vector component to each recipe. 
    The vector component represents how the recipe function.
    While the base_inputs are values stored for catalyst cost calculations.

    Parameters
    ----------
    data:
        Entire data.raw. https://wiki.factorio.com/Data.raw
    RELEVENT_FLUID_TEMPERATURES:
        Dict with keys of fluid names and values of a dict mapping temperatures to energy densities.
    COST_MODE:
        What cost mode is being used. https://lua-api.factorio.com/latest/concepts.html#DifficultySettings
    """
    logging.debug("Beginning the vectorization of recipes.")
    for recipe in data['recipe'].values(): #https://lua-api.factorio.com/latest/prototypes/RecipePrototype.html
        changes = CompressedVector()

        if COST_MODE in recipe.keys(): #https://lua-api.factorio.com/latest/prototypes/RecipePrototype.html#normal or https://lua-api.factorio.com/latest/prototypes/RecipePrototype.html#expensive
            recipe_definition = recipe[COST_MODE]
        else: #https://lua-api.factorio.com/latest/prototypes/RecipePrototype.html#ingredients
            recipe_definition = recipe


        if 'results' in recipe_definition.keys(): #https://lua-api.factorio.com/latest/prototypes/RecipePrototype.html#results both item and fluid have similar structure
            for result in recipe_definition['results']: 
                fixed_name = result['name']

                if 'temperature' in result.keys(): #https://lua-api.factorio.com/latest/types/FluidProductPrototype.html#temperature
                    fluid = data['fluid'][result['name']]
                    fixed_name += '@'+str(result['temperature'])
                    if not result['name'] in RELEVENT_FLUID_TEMPERATURES.keys():
                        RELEVENT_FLUID_TEMPERATURES[result['name']] = {}
                    if not result['temperature'] in RELEVENT_FLUID_TEMPERATURES[result['name']].keys():
                        RELEVENT_FLUID_TEMPERATURES[result['name']][result['temperature']] = Fraction(result['temperature'] - fluid['default_temperature']).limit_denominator() * convert_value_to_base_units(fluid['heat_capacity']) #https://lua-api.factorio.com/latest/prototypes/FluidPrototype.html#heat_capacity

                changes[fixed_name] = average_result_amount(result)
        else: #https://lua-api.factorio.com/latest/prototypes/RecipePrototype.html#result
            changes[recipe_definition['result']] = Fraction(recipe_definition['result_count'] if 'result_count' in recipe_definition.keys() else 1).limit_denominator()  #https://lua-api.factorio.com/latest/prototypes/RecipePrototype.html#result_count
            
        base_inputs = CompressedVector()

        for ingred in recipe_definition['ingredients']: #https://lua-api.factorio.com/latest/types/IngredientPrototype.html both item and fluid have similar structure
            fixed_name = ingred['name']

            if 'minimum_temperature' in ingred.keys():
                fixed_name += '@'+str(ingred['minimum_temperature'])+'-'+str(ingred['maximum_temperature'])

            changes.key_addition(ingred['name'], -1 * Fraction(ingred['amount']).limit_denominator())
            base_inputs.key_addition(ingred['name'], -1 * Fraction(ingred['amount']).limit_denominator())

        recipe['base_inputs'] = base_inputs #store singular run inputs for catalyst calculations.

        recipe['vector'] = (Fraction(1) / Fraction(recipe['energy_required']).limit_denominator()) * changes

def vectorize_resources(data: dict, RELEVENT_FLUID_TEMPERATURES: dict) -> None:
    """
    Adds a base_inputs and vector component to each resource. 
    The vector component represents how the mining of the resource acts.
    While the base_inputs are values stored for catalyst cost calculations.

    Parameters
    ----------
    data:
        Entire data.raw. https://wiki.factorio.com/Data.raw
    RELEVENT_FLUID_TEMPERATURES:
        Dict with keys of fluid names and values of a dict mapping temperatures to energy densities.
    """
    logging.debug("Beginning the vectorization of resources.")
    for resource in data['resource'].values(): #https://lua-api.factorio.com/latest/prototypes/ResourceEntityPrototype.html
        changes = CompressedVector()

        mining_definition = resource['minable']

        if 'results' in mining_definition.keys(): #https://lua-api.factorio.com/latest/types/MinableProperties.html#results
            for result in mining_definition['results']: 
                fixed_name = result['name']

                if 'temperature' in result.keys(): #https://lua-api.factorio.com/latest/types/FluidProductPrototype.html#temperature
                    fluid = data['fluid'][result['name']]
                    fixed_name += '@'+str(result['temperature'])
                    if not result['name'] in RELEVENT_FLUID_TEMPERATURES.keys():
                        RELEVENT_FLUID_TEMPERATURES[result['name']] = {}
                    if not result['temperature'] in RELEVENT_FLUID_TEMPERATURES[result['name']].keys():
                        RELEVENT_FLUID_TEMPERATURES[result['name']][result['temperature']] = Fraction(result['temperature'] - fluid['default_temperature']).limit_denominator() * convert_value_to_base_units(fluid['heat_capacity']) #https://lua-api.factorio.com/latest/prototypes/FluidPrototype.html#heat_capacity

                changes[fixed_name] = average_result_amount(result)

        else: #https://lua-api.factorio.com/latest/types/MinableProperties.html#result
            changes[mining_definition['result']] = Fraction(mining_definition['count']).limit_denominator() #https://lua-api.factorio.com/latest/types/MinableProperties.html#count
            
        base_inputs = CompressedVector()

        if 'required_fluid' in mining_definition: #https://lua-api.factorio.com/latest/types/MinableProperties.html#required_fluid
            changes.key_addition(mining_definition['required_fluid'], -1 * Fraction(mining_definition['fluid_amount']) / 10) #https://lua-api.factorio.com/latest/types/MinableProperties.html#fluid_amount
            base_inputs.key_addition(mining_definition['required_fluid'], -1 * Fraction(mining_definition['fluid_amount']) / 10)
            #don't really have a citation for the divided by 10. Current theory in the discord is that ores are batched 10 at a time.

        resource['base_inputs'] = base_inputs #store singular run inputs for catalyst calculations.

        resource['vector'] = (Fraction(1) / Fraction(mining_definition['mining_time']).limit_denominator()) * changes

def vectorize_technologies(data: dict, COST_MODE: str) -> None:
    """
    Adds a vector component to each technology. 
    The vector component represents how labs researching a technology function.

    Parameters
    ----------
    data:
        Entire data.raw. https://wiki.factorio.com/Data.raw
    COST_MODE:
        What cost mode is being used. https://lua-api.factorio.com/latest/concepts.html#DifficultySettings
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
            fixed_name = ingred['name']

            if 'minimum_temperature' in ingred.keys():
                fixed_name += '@'+str(ingred['minimum_temperature'])+'-'+str(ingred['maximum_temperature'])

            changes[ingred['name']] = -1 * Fraction(ingred['amount']).limit_denominator()
        
        if 'count' in cost_definition['unit'].keys(): #https://lua-api.factorio.com/latest/types/TechnologyUnit.html#count
            changes = Fraction(cost_definition['unit']['count']).limit_denominator() * changes
        else: #https://lua-api.factorio.com/latest/types/TechnologyUnit.html#count_formula
            logging.warning("Formulaic technological counts aren't fully supported yet, only the first instance is done. TODO")
            digit_match = re.search(r'\d+', technology['name'])
            if digit_match is None:
                logging.error("Unable to find a digit to calculate formulaic technology count with. Defaulting to 10.")
                digit = 10
            else:
                digit = int(digit_match.group())
            cost_definition['unit']['count'] = evaluate_formulaic_count(cost_definition['unit']['count_formula'], digit)
            changes = Fraction(cost_definition['unit']['count']).limit_denominator() * changes

        technology['base_inputs'] = CompressedVector({c: v for c, v in changes.items() if v < 0}) #store inputs for lab matching later. TODO. Do we need?
        
        changes[technology['name']+RESEARCH_SPECIAL_STRING] = Fraction(1)  #The result of a technology vector is the researched technology. Enabled recipies are calculated as limits of these results.

        technology['vector'] = changes * (1 / (Fraction(cost_definition['unit']['time']).limit_denominator() * Fraction(cost_definition['unit']['count']).limit_denominator()))

def link_modules(data: dict) -> None:
    """
    Adds the allowed_module component to each recipe and resource mining type representing which modules can be used in a machine running said operation.
    Also populates MODULE_REFERENCE

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
        recipe['allowed_modules'] = []
    for resource in data['resource'].values(): #https://lua-api.factorio.com/latest/prototypes/ResourceEntityPrototype.html
        resource['allowed_modules'] = []
    for tech in data['technology'].values():
        tech['allowed_modules'] = []
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
        
        #TODO: Underdefined for labs?
        for tech in data['technology'].values():
            tech['allowed_modules'].append(module)

        #https://lua-api.factorio.com/latest/types/Effect.html
        module['effect_vector'] = np.array([Fraction(module['effect'][effect]['bonus']).limit_denominator() if effect in module['effect'].keys() else Fraction(0) for effect in MODULE_EFFECTS])

        #MODULE_REFERENCE[module['name']] = module

def set_defaults_and_normalize(data: dict, COST_MODE: str) -> None:
    """
    Sets the defaults of various optional elements that are called later. Including:
    recipe's energy_required (default: .5)
    recipe's result_count (default: 1)
    resource's minable count (default: 1)
    machine's energy_source effectivity (default: 1)

    Normalizes terms across machines:
    furnace's, rocket-silo's, and assembling-machines's 'crafting_speed' is added to 'speed' term
    mining-drill's 'mining_speed' is added to 'speed' term
    lab's 'researching_speed' is added to 'speed' term

    Normalizes terms across recipe-likes:
    recipe's 'energy_required' is added to 'time_multiplier' term
    resources's 'minable' 'mining-time' is added to 'time_multiplier' term
    technology's 'unit' 'time' is added to 'time_multiplier' term

    Normalizes allowed_effects by adding it to crafting machines, mining drills, and labs if not already present.
    https://lua-api.factorio.com/latest/prototypes/CraftingMachinePrototype.html#allowed_effects
    https://lua-api.factorio.com/latest/prototypes/MiningDrillPrototype.html#allowed_effects
    https://lua-api.factorio.com/latest/prototypes/LabPrototype.html#allowed_effects

    Normalizes module_specification -> module_slots by adding it to crafting machines, mining drills, and labs if not already present.
    https://lua-api.factorio.com/latest/prototypes/CraftingMachinePrototype.html#module_specification
    https://lua-api.factorio.com/latest/prototypes/MiningDrillPrototype.html#module_specification
    https://lua-api.factorio.com/latest/prototypes/LabPrototype.html#module_specification

    Normalizes rocket_launch_products, if just rocket_launch_products is not given but rocket_launch_product is we put it into an array.

    Fixes IngredientPrototypes to always be a dict. https://lua-api.factorio.com/latest/types/IngredientPrototype.html
    Fixes ProductPrototypes to always be a dict. https://lua-api.factorio.com/latest/types/ItemProductPrototype.html
    Fixes IngredientPrototypes to always include their type. https://lua-api.factorio.com/latest/types/ItemIngredientPrototype.html#type
    Fixes ProductPrototypes to always include their type. https://lua-api.factorio.com/latest/types/ItemProductPrototype.html#type

    Parameters
    ----------
    data:
        Entire data.raw. https://wiki.factorio.com/Data.raw
    COST_MODE:
        What cost mode is being used. https://lua-api.factorio.com/latest/concepts.html#DifficultySettings
    """
    for recipe in data['recipe'].values():
        if not 'energy_required' in recipe.keys(): #https://lua-api.factorio.com/latest/prototypes/RecipePrototype.html#energy_required
            recipe['energy_required'] = Fraction(1, 2)
        if ('result' in recipe.keys()) and (not 'result_count' in recipe.keys()): #https://lua-api.factorio.com/latest/prototypes/RecipePrototype.html#result_count
            recipe['result_count'] = Fraction(1)
    for resource in data['resource'].values():
        if 'result' in resource['minable'] and not 'count' in resource['minable']:
            resource['minable']['count'] = Fraction(1) #set the default: https://lua-api.factorio.com/latest/types/MinableProperties.html#count
    for ref in ['boiler', 'burner-generator', 'offshore-pump', 'reactor', 'generator', 'furnace', 'mining-drill', 'solar-panel', 'rocket-silo', 'assembling-machine']:
        for machine in data[ref].values():
            if ('energy_source' in machine.keys()) and (not 'effectivity' in machine['energy_source'].keys()):
                machine['energy_source']['effectivity'] = Fraction(1)

    for ref in ['furnace', 'rocket-silo', 'assembling-machine']:
        for machine in data[ref].values():
            machine['speed'] = Fraction(machine['crafting_speed']).limit_denominator()
    for machine in data['mining-drill'].values():
        machine['speed'] = Fraction(machine['mining_speed']).limit_denominator()
    for machine in data['lab'].values():
        machine['speed'] = Fraction(machine['researching_speed']).limit_denominator()

    for recipe in data['recipe'].values():
        if COST_MODE in recipe.keys() and 'energy_required' in recipe[COST_MODE]:
            recipe['time_multiplier'] = Fraction(recipe[COST_MODE]['energy_required']).limit_denominator()
        else:
            recipe['time_multiplier'] = Fraction(recipe['energy_required']).limit_denominator()
    for resource in data['resource'].values():
        resource['time_multiplier'] = Fraction(resource['minable']['mining_time']).limit_denominator()
    for technology in data['technology'].values():
        if COST_MODE in technology.keys():
            technology['time_multiplier'] = Fraction(technology[COST_MODE]['unit']['time']).limit_denominator()
        else:
            technology['time_multiplier'] = Fraction(technology['unit']['time']).limit_denominator()

    for ref in ['furnace', 'rocket-silo', 'assembling-machine']:
        for machine in data[ref].values():
            if not 'allowed_effects' in machine.keys():
                machine['allowed_effects'] = []
    for lab in data['lab'].values():
        if not 'allowed_effects' in lab.keys():
            lab['allowed_effects'] = MODULE_EFFECTS
    for machine in data['mining-drill'].values():
        if not 'allowed_effects' in machine.keys():
            machine['allowed_effects'] = MODULE_EFFECTS

    for ref in ['furnace', 'rocket-silo', 'assembling-machine']:
        for machine in data[ref].values():
            if not 'module_specification' in machine.keys():
                machine['module_specification'] = {}
            if not 'module_slots' in machine['module_specification'].keys():
                machine['module_specification']['module_slots'] = 0
    for lab in data['lab'].values():
        if not 'module_specification' in lab.keys():
            lab['module_specification'] = {}
        if not 'module_slots' in lab['module_specification'].keys():
            lab['module_specification']['module_slots'] = 0
    for machine in data['mining-drill'].values():
        if not 'module_specification' in machine.keys():
            machine['module_specification'] = {}
        if not 'module_slots' in machine['module_specification'].keys():
            machine['module_specification']['module_slots'] = 0
    
    for item in data['item'].values():
        if not 'rocket_launch_products' in item.keys() and 'rocket_launch_product' in item.keys():
            item['rocket_launch_products'] = [item['rocket_launch_product']]

    def set_to_dicts(l):
        for i in range(len(l)):
            if not isinstance(l[i], dict):
                l[i] = {'name': l[i][0], 'amount': l[i][1]}
            if not 'type' in l[i].keys():
                l[i]['type'] = 'solid'
    for resource in data['resource'].values():
        if 'results' in resource['minable'].keys():
            set_to_dicts(resource['minable']['results'])
    for recipe in data['recipe'].values():
        if COST_MODE in recipe.keys():
            recipe_definition = recipe[COST_MODE]
        else:
            recipe_definition = recipe
        if 'results' in recipe_definition.keys():
            set_to_dicts(recipe_definition['results'])
        if 'ingredients' in recipe_definition.keys():
            set_to_dicts(recipe_definition['ingredients'])
    for item in data['item'].values():
        if 'rocket_launch_products' in item.keys():
            set_to_dicts(item['rocket_launch_products'])
    for technology in data['technology'].values():
        if COST_MODE in technology.keys():
            technology_definition = technology[COST_MODE]
        else:
            technology_definition = technology
        set_to_dicts(technology_definition['unit']['ingredients'])

    for building_type in ['boiler', 'burner-generator', 'offshore-pump', 'reactor', 'generator', 'furnace', 'mining-drill', 'solar-panel', 'rocket-silo', 'assembling-machine', 'lab', 'beacon']:
        for machine in data[building_type].values():
            if not 'tile_width' in machine.keys():
                machine['tile_width'] = math.ceil(abs(machine['collision_box'][1][0] - machine['collision_box'][0][0]))
            if not 'tile_height' in machine.keys():
                machine['tile_height'] = math.ceil(abs(machine['collision_box'][1][1] - machine['collision_box'][0][1]))

def complete_premanagement(data: dict, RELEVENT_FLUID_TEMPERATURES: dict, COST_MODE: str) -> TechnologyTree:
    """
    Does all the premanagment steps on data.raw in an appropriate order.

    Parameters
    ----------
    data:
        Entire data.raw. https://wiki.factorio.com/Data.raw
    RELEVENT_FLUID_TEMPERATURES:
        Dict with keys of fluid names and values of a dict mapping temperatures to energy densities.
    COST_MODE:
        What cost mode is being used. https://lua-api.factorio.com/latest/concepts.html#DifficultySettings
    """
    logging.debug("Beginning the premanagement of the game data.raw object.")
    set_defaults_and_normalize(data, COST_MODE)
    standardize_power(data)
    prep_resources(data)
    vectorize_recipes(data, RELEVENT_FLUID_TEMPERATURES, COST_MODE)
    vectorize_resources(data, RELEVENT_FLUID_TEMPERATURES)
    vectorize_technologies(data, COST_MODE)
    link_modules(data)
    return link_techs(data, COST_MODE)

def average_result_amount(result: dict) -> Fraction:
    """
    Calculates the average result amount from a item or fluid product prototype.

    Parameters
    ----------
    result:
        Item or Fluid product prototype.
    
    Returns
    -------
    Average amount of result produced.
    """
    if 'amount' in result.keys(): #https://lua-api.factorio.com/latest/types/FluidProductPrototype.html#amount works the same for items
        amount = Fraction(result['amount']).limit_denominator()
    else: #https://lua-api.factorio.com/latest/types/FluidProductPrototype.html#amount_min works the same for items
        amount = Fraction(1, 2)*Fraction(result['amount_max']+result['amount_min']).limit_denominator()

    if 'probability' in result.keys(): #https://lua-api.factorio.com/latest/types/FluidProductPrototype.html#probability works the same for items
        amount *= Fraction(result['probability']).limit_denominator()
    return amount
