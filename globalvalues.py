import numpy as np
from fractions import Fraction

MODULE_EFFECTS = ['consumption', 'speed', 'productivity'] 
MODULE_EFFECT_MINIMUMS = {'consumption': Fraction(1, 5), 'speed': Fraction(1, 5), 'productivity': Fraction(1)} 
MODULE_EFFECT_MINIMUMS_NUMPY = np.array([1 - MODULE_EFFECT_MINIMUMS[eff] for eff in MODULE_EFFECTS])
DEBUG_SOLVERS = False #Should be set to True for debugging of solvers, will cause solvers that give infesible results to throw errors instead of being ignored.