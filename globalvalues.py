import numpy as np

RELEVENT_FLUID_TEMPERATURES = {} #keys are fluid names, values are a dict with keys of temperature and values of energy density
COST_MODE = 'normal' #can be set to 'expensive' for the other recipes
MACHINE_CATEGORIES = ['assembling-machine', 'rocket-silo', 'boiler', 'burner-generator', 'furnace', 'generator', 'mining-drill', 'offshore-pump', 'reactor', 'solar-panel']
MODULE_EFFECTS = ['consumption', 'speed', 'productivity']
MODULE_EFFECT_MINIMUMS = {'consumption': .2, 'speed': .2, 'productivity': 1}
MODULE_EFFECT_MINIMUMS_NUMPY = np.array([1 - MODULE_EFFECT_MINIMUMS[eff] for eff in MODULE_EFFECTS])
MODULE_REFERENCE = {} #Keys are names and values are Module class instances form linearconstructs.py
DEFAULT_TARGET_RESEARCH_TIME = 10000