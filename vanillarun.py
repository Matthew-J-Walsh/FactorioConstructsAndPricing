import logging

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                    level=logging.INFO, 
                    handlers=[
                        logging.FileHandler(filename="logfiles\\vanillarun.log", mode='w'),
                    ])
logging.getLogger().addHandler(logging.StreamHandler())
logger = logging.getLogger()
from tools import *

def vanilla_main(optimization_mode: dict | str = 'standard'):
    print("Starting run.")
    gamefiles_filename = 'vanilla-rawdata.json'
    output_file = "RunResultsSave.xlsx"
    vanilla = FactorioInstance(gamefiles_filename)
    print("Instance built.")

    if isinstance(optimization_mode, dict):
        uncompiled_cost_function = hybrid_cost_function(optimization_mode, vanilla)
    elif optimization_mode in ['standard', 'basic', 'simple', 'baseline', 'dual']:
        uncompiled_cost_function = standard_cost_function
    elif optimization_mode in ['spatial', 'ore space', 'tiles', 'mining', 'mining tiles', 'resource space']:
        uncompiled_cost_function = lambda pricing_vector, construct, lookup_indicies: spatial_cost_function(vanilla.spatial_pricing, construct, lookup_indicies)
    elif optimization_mode in ['ore', 'ore count', 'raw', 'raw resource', 'resources', 'resource count']:
        uncompiled_cost_function = lambda pricing_vector, construct, lookup_indicies: ore_cost_function(vanilla.raw_ore_pricing, construct, lookup_indicies)
    else:
        raise ValueError(optimization_mode)
    
    logging.info("Ore mode info")
    logging.info(vanilla.raw_ore_pricing)

    nuclear_targets = [
    "heat from nuclear-reactor via uranium-fuel-cell", 
    "water to steam@500 in heat-exchanger with heat", 
    "electric from steam-turbine via steam@500",
    "uranium-fuel-cell in assembling-machine-1", 
    "uranium-fuel-cell in assembling-machine-2", 
    "uranium-fuel-cell in assembling-machine-3", 
    ("kovarex-enrichment-process in centrifuge", False), 
    ("uranium-processing in centrifuge", False), 
    ("uranium-ore in electric-mining-drill", False)]

    nuclear_construct = vanilla.bind_complex_constructs(nuclear_targets)
    vanilla.add_post_analysis(nuclear_construct, {vanilla.reference_list.index("electric"): 1e10})
    vanilla.add_post_analysis("solar-panel", {vanilla.reference_list.index("electric"): 1e10})
    vanilla.add_post_analysis("electric from steam-engine via steam@165", {vanilla.reference_list.index("electric"): 1e10})

    logging.info("=============================================")
    logging.info(str(list(enumerate(vanilla.reference_list))))
    logging.info(vanilla.catalyst_list)

    logging.info("=============================================")
    for construct in vanilla.uncompiled_constructs:
        logging.info(repr(construct)+"\n")

    logging.info("=============================================")
    logging.info(len([cc for cc in vanilla.complex_constructs if not 'with module setup' in cc.ident]))
    logging.info(vanilla.complex_constructs)

    logging.info("=============================================")
    tech_level = vanilla.technological_limitation_from_specification(fully_automated=["automation-science-pack"])
    logging.info(tech_level)

    #logging.info("=============================================")
    #for cc in vanilla.complex_constructs:
    #    if cc.ident=="utility-science-pack in assembling-machine-3 with electric":
    #        assert isinstance(cc.subconstructs[0], CompiledConstruct)
    #        logging.info(cc.subconstructs[0].lookup_table)

    logging.info("=============================================")
    vanilla_chain = FactorioFactoryChain(vanilla, uncompiled_cost_function)

    starting_pricing = {}
    starting_pricing['electric'] = .000001
    starting_pricing['steam@165'] = .0001
    starting_pricing['coal'] = 2
    starting_pricing['iron-ore'] = 2
    starting_pricing['copper-ore'] = 2
    starting_pricing['stone'] = 2
    starting_pricing['iron-plate'] = 3.2 + starting_pricing['iron-ore']
    starting_pricing['copper-plate'] = 3.2 + starting_pricing['copper-ore']
    starting_pricing['iron-gear-wheel'] = .5 + 2 * starting_pricing['iron-plate']
    starting_pricing['copper-cable'] = .5 + starting_pricing['copper-plate'] / 2
    starting_pricing['electronic-circuit'] = .5 + starting_pricing['iron-plate'] + 3 * starting_pricing['copper-cable']
    starting_pricing['stone-furnace'] = .5 + 5 * starting_pricing['stone']
    starting_pricing['assembling-machine-1'] = .5 + 3 * starting_pricing['electronic-circuit'] + 5 * starting_pricing['iron-gear-wheel'] + 9 * starting_pricing['iron-plate']
    starting_pricing['pipe'] = .5 + starting_pricing['iron-plate']
    starting_pricing['pipe-to-ground'] = .5 + 10 * starting_pricing['pipe'] + 5 * starting_pricing['iron-plate']
    starting_pricing['offshore-pump'] = .5 + 2 * starting_pricing['electronic-circuit'] + starting_pricing['iron-gear-wheel'] + starting_pricing['pipe']
    starting_pricing['steam-engine'] = .5 + 8 * starting_pricing['iron-gear-wheel'] + 5 * starting_pricing['pipe'] + 10 * starting_pricing['iron-plate']
    starting_pricing['boiler'] = .5 + starting_pricing['stone-furnace'] + 4 * starting_pricing['pipe']
    starting_pricing['iron-stick'] = .5 + starting_pricing['iron-plate']
    starting_pricing['steel-plate'] = 16 + 5 * starting_pricing['iron-plate']
    starting_pricing['medium-electric-pole'] = .5 + 4 * starting_pricing['iron-stick'] + 2 * starting_pricing['steel-plate'] + 2 * starting_pricing['copper-plate']
    starting_pricing['inserter'] = .5 + 4 * starting_pricing['iron-stick'] + 2 * starting_pricing['steel-plate'] + 2 * starting_pricing['copper-plate']
    starting_pricing['medium-electric-pole'] = .5 + starting_pricing['electronic-circuit'] + starting_pricing['iron-gear-wheel'] + 2 * starting_pricing['iron-plate']
    starting_pricing['transport-belt'] = .5 + starting_pricing['iron-gear-wheel'] + 2 * starting_pricing['iron-plate']
    starting_pricing['splitter'] = 1 + 5 * starting_pricing['electronic-circuit'] + 5 * starting_pricing['iron-plate'] + 4 * starting_pricing['transport-belt']
    starting_pricing['underground-belt'] = 1 + 10 * starting_pricing['iron-plate'] + 5 * starting_pricing['transport-belt']
    starting_pricing['electric-mining-drill'] = 2 + 3 * starting_pricing['electronic-circuit'] + 5 * starting_pricing['iron-gear-wheel'] + 10 * starting_pricing['iron-plate']
    starting_pricing['lab'] = 2 + 10 * starting_pricing['electronic-circuit'] + 10 * starting_pricing['iron-gear-wheel'] + 4 * starting_pricing['transport-belt']

    starting_pricing = CompressedVector({k: Fraction(v).limit_denominator() for k, v in starting_pricing.items()})
    vanilla_chain.initial_pricing(starting_pricing, tech_level)

    print("Starting factories.")

    logging.info("=============================================")
    logger.setLevel(logging.DEBUG)
    vanilla_chain.add("all tech")
    logging.info(set(vanilla_chain.chain[-1].targets.keys()).difference(set(starting_pricing.keys())))

    logging.info("=============================================")
    vanilla_chain.add("all materials")
    logging.info(set(vanilla_chain.chain[-1].targets.keys()).difference(set(starting_pricing.keys())))
    logging.info(vanilla_chain.chain[-1].optimal_pricing_model)

    logging.info("=============================================")
    vanilla_chain.add("all tech")
    logging.info(len(vanilla_chain.chain)-1)

    logging.info("=============================================")
    vanilla_chain.add("all materials")
    logging.info(len(vanilla_chain.chain)-1)
    logging.info(set(vanilla_chain.chain[4].targets.keys()).difference(set(vanilla_chain.chain[2].targets.keys())))
    logging.info(set(vanilla_chain.chain[4].optimal_pricing_model.keys()).difference(set(vanilla_chain.chain[2].optimal_pricing_model.keys())))
    logging.info(vanilla_chain.chain[-1].optimal_pricing_model)

    logging.info("=============================================")
    vanilla_chain.add("all materials")
    logging.info(len(vanilla_chain.chain)-1)
    logging.info(set(vanilla_chain.chain[5].targets.keys()).difference(set(vanilla_chain.chain[4].targets.keys())))
    logging.info(set(vanilla_chain.chain[5].optimal_pricing_model.keys()).difference(set(vanilla_chain.chain[4].optimal_pricing_model.keys())))
    logging.info(vanilla_chain.chain[-1].optimal_pricing_model)

    logging.info("=============================================")
    vanilla_chain.add("all materials")
    logging.info(len(vanilla_chain.chain)-1)
    logging.info(set(vanilla_chain.chain[6].targets.keys()).difference(set(vanilla_chain.chain[5].targets.keys())))
    logging.info(set(vanilla_chain.chain[6].optimal_pricing_model.keys()).difference(set(vanilla_chain.chain[5].optimal_pricing_model.keys())))
    logging.info(vanilla_chain.chain[-1].optimal_pricing_model)

    logging.info("=============================================")
    vanilla_chain.add("all materials")
    logging.info(len(vanilla_chain.chain)-1)
    logging.info(set(vanilla_chain.chain[7].targets.keys()).difference(set(vanilla_chain.chain[6].targets.keys())))
    logging.info(set(vanilla_chain.chain[7].optimal_pricing_model.keys()).difference(set(vanilla_chain.chain[6].optimal_pricing_model.keys())))
    logging.info(vanilla_chain.chain[-1].optimal_pricing_model)
    
    logging.info("=============================================")
    vanilla_chain.add("all tech")
    logging.info(len(vanilla_chain.chain)-1)

    logging.info("=============================================")
    vanilla_chain.add("all materials")
    logging.info(len(vanilla_chain.chain)-1)
    logging.info(set(vanilla_chain.chain[9].targets.keys()).difference(set(vanilla_chain.chain[7].targets.keys())))
    logging.info(set(vanilla_chain.chain[9].optimal_pricing_model.keys()).difference(set(vanilla_chain.chain[7].optimal_pricing_model.keys())))
    logging.info(vanilla_chain.chain[-1].optimal_pricing_model)

    logging.info("=============================================")
    vanilla_chain.add("all materials")
    logging.info(len(vanilla_chain.chain)-1)
    logging.info(set(vanilla_chain.chain[10].targets.keys()).difference(set(vanilla_chain.chain[9].targets.keys())))
    logging.info(set(vanilla_chain.chain[10].optimal_pricing_model.keys()).difference(set(vanilla_chain.chain[9].optimal_pricing_model.keys())))
    logging.info(vanilla_chain.chain[-1].optimal_pricing_model)

    logging.info("=============================================")
    vanilla_chain.add("all tech")
    logging.info(len(vanilla_chain.chain)-1)

    logging.info("=============================================")
    vanilla_chain.add("all materials")
    logging.info(len(vanilla_chain.chain)-1)
    logging.info("=============================================")
    vanilla_chain.add("all materials")
    logging.info(len(vanilla_chain.chain)-1)
    logging.info("=============================================")
    vanilla_chain.add("all materials")
    logging.info(len(vanilla_chain.chain)-1)

    logging.info("=============================================")
    vanilla_chain.add("all tech")
    logging.info(len(vanilla_chain.chain)-1)
    logging.info("=============================================")
    vanilla_chain.add("all materials")
    logging.info(len(vanilla_chain.chain)-1)

    print("Retarget and computing.")

    vanilla_chain.compute_all()

    print("Dumping to Excel.")

    logging.info("=============================================")
    for cc in vanilla.complex_constructs:
        if cc.ident in ["steel-plate in electric-furnace with electric", "automation-science-pack in assembling-machine-3 with electric", "logistic-science-pack in assembling-machine-3 with electric"]:
            assert isinstance(cc.subconstructs[0], CompiledConstruct)
            p0_j = np.zeros(len(vanilla.reference_list), dtype=np.longdouble)
            for k, v in vanilla_chain.chain[-3].full_optimal_pricing_model.items():
                p0_j[vanilla.reference_list.index(k)] = v
            cost_function = lambda construct, lookup_indicies: uncompiled_cost_function(p0_j, construct, lookup_indicies)
            p_j = np.zeros(len(vanilla.reference_list), dtype=np.longdouble)
            for k, v in vanilla_chain.chain[-2].full_optimal_pricing_model.items():
                assert not np.isnan(v), k
                p_j[vanilla.reference_list.index(k)] = v
            priced_indices =  np.array([vanilla.reference_list.index(k) for k in vanilla_chain.chain[-3].optimal_pricing_model.keys()])
            logging.info(cc.subconstructs[0].efficency_dump(cost_function, priced_indices, p_j, vanilla_chain.chain[-2].known_technologies))

    vanilla_chain.dump_to_excel(output_file)

    print("Finished.")


def main():
    vanilla_main()

if __name__=="__main__":
    main()











