import logging

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                    level=logging.INFO, 
                    handlers=[
                        logging.FileHandler(filename="logfiles\\vanillarun.log", mode='w'),
                    ])
#logging.getLogger().addHandler(logging.StreamHandler())
logger = logging.getLogger()
from tools import *

def vanilla_main(optimization_mode: dict | str = 'standard'):
    print("Starting run.")
    gamefiles_filename = 'vanilla-rawdata.json'
    output_file = "RunResultsSave.xlsx"
    vanilla = FactorioInstance(gamefiles_filename)
    #if not force_rebuild and os.path.isfile(instance_filename): #False:
    #    vanilla = FactorioInstance.load(instance_filename)
    #    print("Instance loaded.")
    #else:
    #    vanilla = FactorioInstance(gamefiles_filename)
    #    vanilla.save(instance_filename)
    #    print("Instance built and saved.")

    if isinstance(optimization_mode, dict):
        uncompiled_cost_function: CostFunction = hybrid_cost_function(optimization_mode, vanilla)
    elif optimization_mode in ['standard', 'basic', 'simple', 'baseline', 'dual']:
        uncompiled_cost_function: CostFunction = standard_cost_function
    elif optimization_mode in ['spatial', 'ore space', 'tiles', 'mining', 'mining tiles', 'resource space']:
        uncompiled_cost_function: CostFunction = spatial_cost_function
    elif optimization_mode in ['ore', 'ore count', 'raw', 'raw resource', 'resources', 'resource count']:
        uncompiled_cost_function: CostFunction = ore_cost_function
    elif optimization_mode in ['space', 'space platform']:
        uncompiled_cost_function: CostFunction = space_cost_function
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
    vanilla.bind_surface_construct("Nauvis")

    logging.info("=============================================")
    logging.info(str(list(enumerate(vanilla.reference_list))))
    logging.info(vanilla.catalyst_list)

    logging.info("=============================================")
    for construct in vanilla._uncompiled_constructs:
        logging.info(repr(construct)+"\n")

    logging.info("=============================================")
    logging.info(len([cc for cc in vanilla._complex_constructs if not 'with module setup' in cc.ident]))
    logging.info(vanilla._complex_constructs)

    logging.info("=============================================")
    logging.info(vanilla.research_modifiers)

    logging.info("=============================================")
    tech_level = vanilla.technological_limitation_from_specification(fully_automated=["automation-science-pack"])
    logging.info(tech_level)

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

    def curated():
        vanilla_chain.initial_pricing(starting_pricing, tech_level)
        print("Starting factories.")

        logger.setLevel(logging.DEBUG)
        logging.info("=============================================")
        vanilla_chain.add("all tech")
        logging.info("=============================================")
        vanilla_chain.add("all materials")
        logging.info("=============================================")
        vanilla_chain.add("all tech")
        logging.info("=============================================")
        vanilla_chain.add("all materials")
        logging.info("=============================================")
        vanilla_chain.add("all materials")
        logging.info("=============================================")
        vanilla_chain.add("all materials")
        logging.info("=============================================")
        vanilla_chain.add("all materials")
        logging.info("=============================================")
        vanilla_chain.add("all tech")
        logging.info("=============================================")
        vanilla_chain.add("all materials")
        logging.info("=============================================")
        vanilla_chain.add("all materials")
        logging.info("=============================================")
        vanilla_chain.add("all tech")
        logging.info("=============================================")
        vanilla_chain.add("all materials")
        logging.info("=============================================")
        vanilla_chain.add("all materials")
        logging.info("=============================================")
        vanilla_chain.add("all materials")
        logging.info("=============================================")
        vanilla_chain.add("all tech")
        logging.info("=============================================")
        vanilla_chain.add("all materials")
    
    def uncurated():
        vanilla_chain.initial_pricing(starting_pricing, tech_level)
        print("Autocompleting factories.")
        vanilla_chain.complete()

    def blank_run():
        #tech_level = vanilla.technological_limitation_from_specification()
        vanilla_chain.initial_pricing(CompressedVector(), tech_level)
        print("Autocompleting factories from nothing.")
        vanilla_chain.complete()

    curated()
    #uncurated()
    #blank_run()

    print("Computing.")

    vanilla_chain.compute_all()

    print("Retarget and computing.")

    vanilla_chain.retarget_all()

    print("Dumping to Excel.")

    vanilla_chain.dump_to_excel(output_file)

    print("Finished.")


def main():
    vanilla_main()

if __name__=="__main__":
    main()











