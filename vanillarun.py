import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
from tools import *

filename = 'C:\\Users\\OptimusPrimeLord\\Documents\\FactorioCalc\\FactorioConstructsAndPricing\\vanilla-rawdata.json'
vanilla = FactorioInstance(filename)

logging.info("=============================================")
logging.info(vanilla.reference_list)
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

logging.info("=============================================")
vanilla_chain = FactorioFactoryChain(vanilla)

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

logging.info("=============================================")
logger.setLevel(logging.DEBUG)
vanilla_chain.add("all tech")
logging.info(set(vanilla_chain.chain[-1].targets.keys()).difference(set(starting_pricing.keys())))

logging.info("=============================================")
vanilla_chain.add("all materials")
logging.info(set(vanilla_chain.chain[-1].targets.keys()).difference(set(starting_pricing.keys())))
logging.info(vanilla_chain.chain[-1].optimal_pricing_model)

try:
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
except:
    pass
















