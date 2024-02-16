import numpy as np

RELEVENT_FLUID_TEMPERATURES = {} #keys are fluid names, values are a dict with keys of temperature and values of energy density
COST_MODE = 'normal' #can be set to 'expensive' for the other recipes
MACHINE_CATEGORIES = ['assembling-machine', 'rocket-silo', 'boiler', 'burner-generator', 'furnace', 'generator', 'mining-drill', 'offshore-pump', 'reactor', 'solar-panel']
MODULE_EFFECTS = ['consumption', 'speed', 'productivity']
MODULE_EFFECT_MINIMUMS = {'consumption': .2, 'speed': .2, 'productivity': 1}
MODULE_EFFECT_MINIMUMS_NUMPY = np.array([1 - MODULE_EFFECT_MINIMUMS[eff] for eff in MODULE_EFFECTS])
FALSE_CATALYST_METHODS = [['fill', 'empty'], ['barrel']] #This is a list of terms that can be found in a catalyst name that indicate it might link to itself accidently. Classic example is fluid -> barrel -> fluid. Currently set to require 'fill' or 'empty' and 'barrel'
FALSE_CATALYST_LINKS = ['empty-barrel'] #nothing in this list is allowed to be used as a link for something being a catalyst