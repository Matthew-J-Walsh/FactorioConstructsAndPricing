from globalsandimports import *
from constructs import *
from lpproblems import *


def compute_complex_construct_efficiency(construct: ComplexConstruct, initial_pricing_model: np.ndarray, initial_pricing_keys: list[int], 
                                         output_pricing_model: np.ndarray, nnzfluxes: list[int], known_technologies: TechnologicalLimitation, 
                                         targets: dict[int, float]) -> float | None:
    """
    Computes the efficiency of a complex construct producing a target given a pricing model.

    Parameters
    ----------
    construct:
        ComplexConstruct to be evaluated.
    initial_pricing_model:
        Pricing model used to obtain optimal factory.
    initial_pricing_keys:
        Location of accuratly priced
    output_pricing_model:
        Pricing model obtained from optimal factory.
    nnzfluxes:
        Which items from the pricing model are guarenteed to be accurately priced.
    known_technologies:
        TechnologicalLimitation for building optimal factory.
    targets:
        Dict of output targets and their relative output.
    
    Returns
    -------
    Decimal Efficiency
    """
    A, c, N1, N0, Recovery = construct.compile(initial_pricing_model, initial_pricing_keys, known_technologies)
    Acsr = A.tocsr()
    valid_rows = list(set(targets.keys()).union(set(nnzfluxes)))
    valid_rows.sort() #do we really need?
    masked_A = sparse.coo_matrix(Acsr[np.array(valid_rows),:])
    
    primal_diluted, dual = DUAL_LP_SOLVERS[0](masked_A, np.array(targets.values()), c - Acsr @ output_pricing_model)
    if primal_diluted is None or dual is None:
        logging.error("Unable to solve initial construct efficiency problem.")
        return None
    
    primal, dual = DUAL_LP_SOLVERS[0](sparse.coo_matrix(sparse.vstack(sparse.coo_array(Acsr @ output_pricing_model - c), masked_A)), 
                                                                                  np.concatenate(np.array([np.dot(Acsr @ output_pricing_model - c, primal_diluted)]), 
                                                                                                 np.array(targets.values())), 
                                                                                  np.ones(A.shape[1]), g=primal_diluted)
    if primal is None or dual is None:
        logging.error("Unable to solve secondary construct efficiency problem.")
        return None
    
    return np.dot(Acsr @ output_pricing_model, primal) /  np.dot(c, primal)

def compute_transportation_densities(data: dict, pricing_model: CompressedVector) -> None:
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
    output = CompressedVector()
    for ident, value in pricing_model.items():
        if ident in data['item'].keys():
            output.update({ident+" via "+"belt": value})
            output.update({ident+" via "+"inserter": value})
            for wagon in data['cargo-wagon'].values():
                output.update({ident+" via "+wagon['name']+" cargo-wagon": value * data['item'][ident]['stack_size'] * wagon['inventory_size']})
        elif ident in data['fluid'].keys():
            container_size, container_stacks = find_fluid_container_size(data, data['fluid'][ident])
            if not container_size is None:
                output.update({ident+" via "+"belt": value * container_size})
                output.update({ident+" via "+"inserter": value * container_size})
                for wagon in data['cargo-wagon'].values():
                    output.update({ident+" via "+wagon['name']+" cargo-wagon": value * container_stacks * wagon['inventory_size'] * container_size})
            output.update({ident+" via "+"pipe": value * EXPECTED_PIPE_FLOW_RATE})
            for wagon in data['fluid-wagon'].values():
                output.update({ident+" via "+wagon['name']+" fluid-wagon": value * wagon['capacity'] * (wagon['tank_count'] if 'tank_count' in wagon.keys() else 3)})
        else:
            logging.error("Unknown item without type: "+ident)

def find_fluid_container_size(data: dict, fluid: dict) -> tuple[int, int] | tuple[None, None]:
    """
    Finds the fluid container size and stack size.
    """
    direct_ancestors_list: set[str] = set()
    direct_predecessors_list: set[str] = set()
    relevent_recipes: set[str] = set()
    for recipe in data['recipe']:
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
