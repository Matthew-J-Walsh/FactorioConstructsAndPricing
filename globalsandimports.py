import numpy as np
import scipy as sp
from scipy import linalg
from scipy import sparse
from scipy import optimize
import itertools
import copy
import re
import logging
import time
import math
from fractions import Fraction
import json
import Levenshtein
import numexpr
import pandas as pd

import pyscipopt as scip
import highspy
import pulp as pl

import typing
from numbers import Real, Number
from typing import Tuple, TypeVar, Callable, Hashable, Iterable, Collection, Any, Optional, Generator, Protocol

class CallableSolver(Protocol):
    def __call__(self, A: sparse.csr_matrix, b: np.ndarray, c: np.ndarray | None = None, g: np.ndarray | None = None) -> np.ndarray | None:
        return None

class CallableSparseSolver(Protocol):
    def __call__(self, A: sparse.csr_matrix, b: np.ndarray, c: np.ndarray, g: np.ndarray | None = None, ginv: np.ndarray | None = None) -> Tuple[np.ndarray | None, np.ndarray | None]:
        return None, None

class CallableDenseSolver(Protocol):
    def __call__(self, A: np.ndarray, b: np.ndarray, c: np.ndarray, g: np.ndarray | None = None, ginv: np.ndarray | None = None) -> Tuple[np.ndarray | None, np.ndarray | None]:
        return None, None

BEST_LP_SOLVER: CallableDenseSolver

ALL_MODULE_EFFECTS = ['consumption', 'speed', 'productivity', 'pollution'] 
ACTIVE_MODULE_EFFECTS = ['consumption', 'speed', 'productivity'] 
MODULE_EFFECT_MINIMUMS = {'consumption': Fraction(1, 5), 'speed': Fraction(1, 5), 'productivity': Fraction(1)} 
def multilienar_effect_ordering():
    """
    Creates an effect ordering, list of tuples, for the standard effect multilinear form.
    123 12 13 23 1 2 3 0
    1234 123 124 134 234 12 13 14 23 24 34 1 2 3 4 0
    """
    l = len(ACTIVE_MODULE_EFFECTS)
    ordering: list[set[int]] = []
    for i in range(l+1):
        for c in itertools.combinations(range(l), i):
            ordering.append(set(c))
    return ordering
MODULE_EFFECT_ORDERING = multilienar_effect_ordering()
MODULE_EFFECT_MINIMUMS_NUMPY = np.array([1 - MODULE_EFFECT_MINIMUMS[eff] for eff in ACTIVE_MODULE_EFFECTS])

DEBUG_BLOCK_MODULES: bool = False #Should modules be removed from pricing to speed up debugging of non-module related issues.
DEBUG_BLOCK_BEACONS: bool = False #Should modules be removed from pricing to speed up debugging of non-module related issues.

OUTPUT_WARNING_LIST = [] #List of items that have had warnings thrown about them. Won't throw the same item twice.

SOLVER_TOLERANCES = {'rtol': 1e-4, 'atol': 1e-6}
BASELINE_RETAINMENT_VALUE = 1e-2

#How item's per second a single logistical mobility tool can handle from a construct, solids need both a belt and an inserter, liquids need just pipes.
#A value of 1 on belts means that for every 15 items a second we need 1 yellow transport belt.
#Inserter upgrades dont work yet ;.; and neither do more advanced versions (fast inserter etc.)... even if they are more efficient.
LOGISTICAL_COST_MULTIPLIERS: dict = {"transport-belt": .1, "inserter": 1, "pipe": .5}
PIPE_EXPECTED_SPEED = 200 #Fluid flow rate through pipe guess.

#Prototype Lists
ITEM_SUB_PROTOTYPES = ['item', 'ammo', 'capsule', 'gun', 'item-with-entity-data', 'item-with-label', 'item-with-inventory', 'blueprint-book', 'item-with-tags', 'selection-tool',
                       'blueprint', 'copy-paste-tool', 'deconstruction-item', 'upgrade-item', 'module', 'rail-planner', 'spidertron-remote', 'tool', 'armor', 'mining-tool', 'repair-tool']
#Special identifier for research names so that they don't overlap with other items.
RESEARCH_SPECIAL_STRING = "=research"

