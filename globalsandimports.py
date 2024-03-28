import numpy as np
import scipy as sp
from scipy import linalg
from scipy import sparse
from scipy import optimize
import itertools
import copy
import typing
import re
import logging
import time
import math
from fractions import Fraction
import json
import Levenshtein
import numexpr

import pyscipopt as scip
import highspy
import pulp as pl
import pandas as pd

from numbers import Real
from typing import TypeVar, Callable, Hashable, Iterable, Any, Optional, Generator

MODULE_EFFECTS = ['consumption', 'speed', 'productivity'] 
MODULE_EFFECT_MINIMUMS = {'consumption': Fraction(1, 5), 'speed': Fraction(1, 5), 'productivity': Fraction(1)} 
MODULE_EFFECT_MINIMUMS_NUMPY = np.array([1 - MODULE_EFFECT_MINIMUMS[eff] for eff in MODULE_EFFECTS])
DEBUG_SOLVERS = False #Should be set to True for debugging of solvers, will cause solvers that give infesible results to throw errors instead of being ignored.
DEBUG_BLOCK_MODULES = False #Should modules be removed from pricing to speed up debugging of non-module related issues.
DEBUG_BLOCK_BEACONS = False #Should modules be removed from pricing to speed up debugging of non-module related issues.
DEBUG_TIME_DUAL_PROBLEM = False #Should the standard problem be timed when solving a dual problem.
DEBUG_TIME_ITERATIVE_PROBLEM = False #Should the standard problem be timed when solving an iterative problem.
DEBUG_REFERENCE_LIST = [] #copy of reference list for debugging difficult construct problems.
WARNING_LIST = [] #List of items that have had warnings thrown about them. Won't throw the same item twice.
BENCHMARKING_MODE = False #Should the different lp solvers be benchmarked against eachother.
BENCHMARKING_TIMES = {}

#Prototype Lists
ITEM_SUB_PROTOTYPES = ['item', 'ammo', 'capsule', 'gun', 'item-with-entity-data', 'item-with-label', 'item-with-inventory', 'blueprint-book', 'item-with-tags', 'selection-tool',
                       'blueprint', 'copy-paste-tool', 'deconstruction-item', 'upgrade-item', 'module', 'rail-planner', 'spidertron-remote', 'tool', 'armor', 'mining-tool', 'repair-tool']




