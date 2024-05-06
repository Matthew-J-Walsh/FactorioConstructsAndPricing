from globalsandimports import *
from constructs import *
from lpproblems import *


def compute_complex_construct_efficiency(construct: ComplexConstruct, initial_pricing_model: np.ndarray, initial_pricing_keys: list[int], 
                                         output_pricing_model: np.ndarray, zfluxes: list[int], known_technologies: TechnologicalLimitation, 
                                         targets: dict[int, float], scale: float) -> float | None:
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
    zfluxes:
        Which items from the pricing model are not guarenteed to be accurately priced.
    known_technologies:
        TechnologicalLimitation for building optimal factory.
    targets:
        Dict of output targets and their relative output.
    scale:
        Scale difference between output pricing model and actual
    
    Returns
    -------
    Decimal Efficiency
    """
    #logging.info("========================================")
    #logging.info(construct)
    #logging.info(known_technologies)
    #logging.info(initial_pricing_keys)
    A, c, N1, N0, Recovery = construct.compile(initial_pricing_model, initial_pricing_keys, known_technologies, False)
    Acsr = A.tocsr()
    valid_rows = list(set(targets.keys()).union(set(zfluxes)))
    valid_rows.sort() #do we really need? This is a disaster
    valid_rows = np.array(valid_rows)
    col_values = np.zeros(valid_rows.shape)
    for k, v in targets.items():
        logging.info(np.where(valid_rows == k)[0][0])
        col_values[np.where(valid_rows == k)[0][0]] = v
    masked_Acsr = Acsr[valid_rows,:]
    masked_A = sparse.coo_matrix(masked_Acsr)
    #logging.info(targets)
    #logging.info(zfluxes)
    #logging.info(valid_rows)
    #logging.info(col_values)
    #logging.info(A.shape)
    #logging.info(c.shape)
    #logging.info(output_pricing_model.shape)
    #logging.info(type(masked_A))
    #logging.info(masked_A.shape)
    #logging.info(masked_A)
    #logging.info(type(col_values))
    #logging.info(col_values.shape)
    #logging.info(col_values)
    #logging.info(type(c - (Acsr.T @ output_pricing_model)))
    #logging.info((c - (Acsr.T @ output_pricing_model)).shape)
    #logging.info((c - (Acsr.T @ output_pricing_model)))

    cost_vec = (c - (Acsr.T @ output_pricing_model)).copy()

    primal_diluted, dual = BEST_LP_SOLVER(masked_A, col_values, cost_vec)
    if primal_diluted is None or dual is None:
        logging.error("Unable to solve initial construct efficiency problem.")
        return None
    
    assert linear_transform_is_gt(masked_A, primal_diluted, col_values).all()
    
    #logging.info(primal_diluted)
    #logging.info(dual)
    
    #logging.info((cost_vec[None, :]).shape)
    #logging.info(masked_Acsr.shape)
    modified_A: sparse.coo_matrix = sparse.vstack([cost_vec[None, :], masked_Acsr], format="coo") # type: ignore
    #logging.info(A)
    #logging.info(modified_A)
    modified_B = np.concatenate([np.array([np.dot(cost_vec, primal_diluted)]), col_values])
    #logging.info(col_values)
    #logging.info(modified_B)
    #logging.info(type(modified_A))
    #logging.info(modified_A.shape)
    #logging.info(modified_B.shape)
    #logging.info(np.ones(A.shape[1]).shape)
    #logging.info(primal_diluted.shape)
    assert linear_transform_is_gt(modified_A, primal_diluted, modified_B).all()
    
    primal, dual = BEST_LP_SOLVER(
                                  modified_A, # type: ignore
                                  modified_B, np.ones(A.shape[1]), g=primal_diluted)
    if primal is None or dual is None:
        raise RuntimeError("Alegedly no primal found but we have one.")
    
    #logging.info(primal)
    #logging.info(dual)
    assert linear_transform_is_gt(masked_A, primal, col_values).all()
    #logging.info(np.dot(primal_diluted, cost_vec))
    #logging.info(np.dot(primal, cost_vec))
    #assert np.isclose(np.dot(primal_diluted, cost_vec), np.dot(primal, cost_vec), rtol=SOLVER_TOLERANCES['rtol'], atol=SOLVER_TOLERANCES['atol']), str(np.dot(primal_diluted, cost_vec))+", "+str(np.dot(primal, cost_vec))
    #logging.info(output_pricing_model)
    #logging.info(Acsr.T @ output_pricing_model)
    #logging.info(primal)
    #logging.info(np.multiply(Acsr.T @ output_pricing_model, primal))
    #logging.info(np.dot(Acsr.T @ output_pricing_model, primal))
    #logging.info(np.dot(c, primal))
    
    return np.dot(Acsr.T @ output_pricing_model, primal) /  np.dot(c, primal) / scale
    #return np.dot((Acsr / c[:, None]).tocsr().T @ output_pricing_model, primal) / scale # type: ignore

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
    output.append((ident, "pipe", value * EXPECTED_PIPE_FLOW_RATE))
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
