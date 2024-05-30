from globalsandimports import *
from constructs import *
from lpproblems import *

def compute_transportation_densities(data: dict, pricing_model: CompressedVector) -> list[tuple[str, str, Fraction | float]]:
    """
    Calculates the transporation densities of all relevent items given a pricing model.

    Parameters
    ----------
    data:
        The entire data.raw
    pricing_model:
        Pricing model obtained of calculated factory. Should be stripped of all non-relevent or unwanted items.

    Returns
    -------
    CompressedVector of the various labeled transportation densities.
    """
    output: list[tuple[str, str, float]] = []
    for ident, value in pricing_model.items():
        if RESEARCH_SPECIAL_STRING in ident:
            continue #research has no density
        item_cata = [cata for cata in ITEM_SUB_PROTOTYPES if ident in data[cata].keys()]
        item_cata = item_cata[0] if len(item_cata)>0 else None
        if not item_cata is None:
            output.append((ident, "belt", value))
            output.append((ident, "inserter", value))
            for wagon in data['cargo-wagon'].values():
                output.append((ident, wagon['name']+" cargo-wagon", value * data[item_cata][ident]['stack_size'] * wagon['inventory_size']))
        elif ident in data['fluid'].keys():
            fluid_transport_density_helper(ident, value, output, data)
        elif '@' in ident:
            if ident.split('@')[0] in data['fluid'].keys():
                fluid_transport_density_helper(ident.split('@')[0], value, output, data)
            else:
                logging.error("Unknown object with temperature: "+ident)
        elif ident in ["electric", "heat"]:
            logging.debug("Found object "+ident+" with no transport methods. Skipping.")
        else:
            logging.error("Object without known type: "+ident)

    return output # type: ignore

def fluid_transport_density_helper(ident: str, value: float, output: list[tuple[str, str, float]], data: dict):
    """
    Helper function for compute_transportation_density for fluids
    """
    container_size, container_stacks = find_fluid_container_size(data, data['fluid'][ident])
    if not container_size is None and not container_stacks is None:
        output.append((ident, "belt", value * container_size))
        output.append((ident, "inserter", value * container_size))
        for wagon in data['cargo-wagon'].values():
            output.append((ident, wagon['name']+" cargo-wagon", value * container_stacks * wagon['inventory_size'] * container_size))
    output.append((ident, "pipe", value * PIPE_EXPECTED_SPEED))
    for wagon in data['fluid-wagon'].values():
        output.append((ident, wagon['name']+" fluid-wagon", value * wagon['capacity'] * (wagon['tank_count'] if 'tank_count' in wagon.keys() else 3)))

def find_fluid_container_size(data: dict, fluid: dict) -> tuple[int, int] | tuple[None, None]:
    """
    Finds the fluid container size and stack size.
    """
    assert 'name' in fluid.keys(), fluid
    direct_ancestors_list: set[str] = set()
    direct_predecessors_list: set[str] = set()
    relevent_recipes: set[str] = set()
    for recipe in data['recipe'].values():
        if fluid['name'] in recipe['vector'].keys():
            if recipe['vector'][fluid['name']] > 0:
                for ident, delta in recipe['vector'].items():
                    if ident!=fluid['name'] and delta < 0:
                        direct_ancestors_list.add(ident)
                        relevent_recipes.add(recipe['name'])
            elif recipe['vector'][fluid['name']] > 0:
                for ident, delta in recipe['vector'].items():
                    if ident!=fluid['name'] and delta > 0:
                        direct_predecessors_list.add(ident)
                        relevent_recipes.add(recipe['name'])

    intersect = direct_ancestors_list.intersection(direct_predecessors_list)
    containers: list[tuple[int, int]] = []
    for item in intersect:
        paired_value: int | None = None
        for recipe_name in relevent_recipes:
            recipe = data['recipe'][recipe_name]
            if item in recipe['vector'].keys():
                if not paired_value is None and not int(abs(recipe['vector'][fluid['name']]/recipe['vector'][item]))==paired_value: #false match
                    paired_value = None 
                    break
                paired_value = int(abs(recipe['vector'][fluid['name']]/recipe['vector'][item]))
        if not paired_value is None:
            containers.append((paired_value, data['item'][item]['stack_size']))
        
    assert len(containers)<2, "Found multiple containers, something is wrong!"
    if len(containers)==0:
        return None, None
    else:
        return containers[0]
