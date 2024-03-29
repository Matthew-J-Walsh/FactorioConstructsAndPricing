import numpy as np
import scipy as sp
from scipy import linalg
import itertools
import copy #TODO do i need?
import typing
import re
import logging
from fractions import Fraction
from numbers import Real
from typing import TypeVar, Callable, Hashable, Iterable, Any, Optional, Generator

logging.basicConfig(level=logging.DEBUG)

MODULE_EFFECTS = ['consumption', 'speed', 'productivity'] 
MODULE_EFFECT_MINIMUMS = {'consumption': Fraction(1, 5), 'speed': Fraction(1, 5), 'productivity': Fraction(1)} 
MODULE_EFFECT_MINIMUMS_NUMPY = np.array([1 - MODULE_EFFECT_MINIMUMS[eff] for eff in MODULE_EFFECTS])
DEBUG_SOLVERS = False #Should be set to True for debugging of solvers, will cause solvers that give infesible results to throw errors instead of being ignored.