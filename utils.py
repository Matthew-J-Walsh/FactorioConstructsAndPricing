from __future__ import annotations

from globalsandimports import *

T = TypeVar('T')

class CompressedVector(dict):
    """CompressedVector's are dicts where the values are all numbers and the values represent some dimension

    They also have hashes for easier search through collections
    """
    hash_value: int | None

    def __init__(self, input = None) -> None: # type: ignore
        if input is None:
            super().__init__()
        else:
            super().__init__(input) # type: ignore
        self.hash_value = None

    def __add__(self, other: CompressedVector) -> CompressedVector:
        self.hash_value = None
        new_cv = CompressedVector()
        for k, v in self.items():
            new_cv.key_addition(k, v)
        for k, v in other.items():
            new_cv.key_addition(k, v)
        return new_cv

    def __mul__(self, multiplier: Number) -> CompressedVector:
        self.hash_value = None
        dn = CompressedVector()
        for k, v in self.items():
            dn[k] = multiplier * v
        return dn
    
    def __rmul__(self, multiplier: Number) -> CompressedVector:
        self.hash_value = None
        return self.__mul__(multiplier)
    
    def key_addition(self, key, value) -> None:
        """
        Single line adding key or adding to value in key.
        """
        self.hash_value = None
        if key in self.keys():
            self[key] += value
        else:
            self[key] = value
    
    def __hash__(self) -> int: # type: ignore
        if self.hash_value is None:
            self.hash_value = hash(tuple(sorted(self.items())))
        return self.hash_value
    
    def __eq__(self, other: object) -> bool:
        assert isinstance(other, dict)
        if hash(self)!=hash(other):
            return False
        if set(self.keys())!=set(other.keys()):
            return False
        for k in self.keys():
            if other[k]!=self[k]:
                return False
        return True

    def __lt__(self, other: CompressedVector) -> bool:
        return hash(self) < hash(other)

def count_via_lambda(l: list[T] | dict[Any, T], func: Callable[[T], bool] = lambda x: True) -> int:
    """Counting helper for lists or dicts, counts how many elements in the list return true when passed to function func

    Parameters
    ----------
    l : list[T] | dict[Any, T]
        The list or dict
    func : _type_, optional
        A function with 1 list element as input, by default lambdax:True

    Returns
    -------
    int
        Number of instances where func returned Truthy on list elements
    """
    if isinstance(l, list):
        return sum([1 if func(e) else 0 for e in l])
    else: #isinstance(l, dict):
        return sum([1 if func(e) else 0 for e in l.values()])

class TechnologyTree:
    """
    A representation of the technology tree for more effective technology limitations.
    We only need to store the 'caps' of the tree.
    We want to be able to, given a cap researches of a technological limitation, add two tech limits, subtract them, 
    and most importantly determine if all researches imply another tech limit (>=).
    The fastest way to do this will be to provide a function that determines if a research is greater than, less than, equal to, or skew to another research.

    Let A and B be researches.

    A > B iff B is a prerequisite of A.

    A = B iff they are the same research.

    A < B iff A is a prerequisite of B.

    A X B (X = 'skew') iff None of the above 3 apply.
    """
    reference_map: dict[str, int]
    inverse_map: dict[int, str]
    comparison_table: np.ndarray

    def __init__(self, technology_data: list[dict]) -> None:
        self.reference_map = {}
        self.inverse_map = {}
        self.comparison_table = np.full((len(technology_data), len(technology_data)), 13, dtype=float) #random number to check for at end to make sure we filled it
        for i, tech in enumerate(technology_data):
            self.reference_map.update({tech['name']: i})
            self.inverse_map.update({i: tech['name']})
            tech.update({'tech_tree_identifier': i})
            for j, other_tech in enumerate(technology_data):
                if tech['name']==other_tech['name']:
                    self.comparison_table[i, j] = 0
                elif other_tech in tech['all_prereq']:
                    self.comparison_table[i, j] = 1
                elif tech in other_tech['all_prereq']:
                    self.comparison_table[i, j] = -1
                else:
                    self.comparison_table[i, j] = float('nan')
        assert not (self.comparison_table==13).any()

    def compare(self, rA: int, rB: int) -> int:
        """
        A > B = 1 iff B is a prerequisite of A
        A = B = 0 iff they are the same research
        A < B = -1 iff A is a prerequisite of B
        A X B = nan (X = 'skew') iff None of the above 3 apply.
        """
        return self.comparison_table[rA, rB]

    def simplify(self, sets_of_references: Collection[Collection[int]]) -> frozenset[frozenset[int]]:
        """
        Given a list of possible lists research references simplify it to the minimal set of minimal sets of researches.
        """
        subsets = set()
        for sublist in sets_of_references:
            subset: set[int] = set()

            for research in sublist:
                covered = False
                cleaved = []
                for other_res in subset:
                    if self.compare(other_res, research) >= 0:
                        covered = True
                    if self.compare(research, other_res) >= 0:
                        cleaved.append(other_res)
                
                if not covered:
                    subset.add(research)
                    for cleave in cleaved:
                        subset.remove(cleave)
                else:
                    assert len(cleaved)==0

            subsets.add(frozenset(subset))

        return frozenset(subsets)

    def reference(self, sets_of_researches: Collection[Collection[str]]) -> frozenset[frozenset[int]]:
        """
        Given a list of possible lists researche names simplify it to the minimal set of minimal sets of researches.
        """
        sets_of_references: list[list[int]] = [[self.reference_map[research_str] for research_str in sublist] for sublist in sets_of_researches]
        return self.simplify(sets_of_references)
    
    def techlimit_ge(self, A: TechnologicalLimitation, B: TechnologicalLimitation) -> bool:
        for Asubsets in A.canonical_form:
            for Bsubsets in B.canonical_form:
                for Bresearch in Bsubsets:
                    covered = False

                    for Aresearch in Asubsets:
                        if self.compare(Aresearch, Bresearch)>=0:
                            covered = True
                            break

                    if not covered:
                        return False

        return True

class TechnologicalLimitation:
    """
    Shortened to a 'limit' elsewhere in the program. Represents the technologies that must be researched in order
    to unlock the specific object (recipe, machine, module, etc.).

    Members
    -------
    canonical_form:
        List of sets making up the canonical form.
    """
    tree: TechnologyTree
    canonical_form: frozenset[frozenset[int]]

    def __init__(self, tree: TechnologyTree, sets_of_researches: Collection[Collection[str]] = [], sets_of_references: Collection[Collection[int]] = []) -> None:
        """
        Parameters
        ----------
        """
        assert len(sets_of_researches)==0 or len(sets_of_references)==0, "Can't take both input types"
        self.tree = tree
        if len(sets_of_references)==0:
            self.canonical_form = self.tree.reference(sets_of_researches)
        else:
            self.canonical_form = self.tree.simplify(sets_of_references)
        
    def __repr__(self) -> str:
        return str(self.canonical_form)

    def __add__(self, other: TechnologicalLimitation) -> TechnologicalLimitation:
        """
        Addition betwen two TechnologicalLimitations is an AND operation.
        """
        assert self.tree == other.tree
        if len(self.canonical_form)==0:
            return other
        if len(other.canonical_form)==0:
            return self
        added = set()
        for s1 in self.canonical_form:
            for s2 in other.canonical_form:
                added.add(s1.union(s2))
        return TechnologicalLimitation(self.tree, sets_of_references=added)

    def __sub__(self, other: Collection[str]) -> list[frozenset[str]]:
        """
        Returns a list of possible additions to a set of researches that will result in the completion of limitation.
        """
        raise NotImplementedError()
        return [frozenset(nodes.difference(other)) for nodes in self.canonical_form]
        
    def __ge__(self, other: TechnologicalLimitation) -> bool: #only permitted and only needed comparison, "Does having this unlocked mean that other is unlocked?"
        assert self.tree == other.tree
        return self.tree.techlimit_ge(self, other)

class TechnologicalLimitationOld:
    """
    Shortened to a 'limit' elsewhere in the program. Represents the technologies that must be researched in order
    to unlock the specific object (recipe, machine, module, etc.).

    Members
    -------
    canonical_form:
        List of sets making up the canonical form.
    """
    canonical_form: frozenset[frozenset[str]]

    def __init__(self, list_of_nodes: Collection[Collection[str]] = []) -> None:
        """
        Parameters
        ----------
        list_of_nodes:
            Collection of iterable over nodes to make up this limitation.
        """
        temp_cf = set()
        for nodes in list_of_nodes:
            temp_cf.add(frozenset(nodes))
        self.canonical_form = frozenset(temp_cf)
        
    def __repr__(self) -> str:
        return str(self.canonical_form)
        #return ", or\n\t".join([" and ".join(nodes) for nodes in self.canonical_form])

    def __add__(self, other: TechnologicalLimitationOld) -> TechnologicalLimitationOld:
        """
        Addition betwen two TechnologicalLimitations is an AND operation.
        """
        if len(self.canonical_form)==0:
            return other
        if len(other.canonical_form)==0:
            return self
        added = set()
        for s1 in self.canonical_form:
            for s2 in other.canonical_form:
                added.add(s1.union(s2))
        return TechnologicalLimitationOld(added)

    def __sub__(self, other: Collection[str]) -> list[frozenset[str]]:
        """
        Returns a list of possible additions to a set of researches that will result in the completion of limitation.
        """
        return [frozenset(nodes.difference(other)) for nodes in self.canonical_form]
        
    #def __eq__(self, other: TechnologicalLimitation) -> bool:
    #    return self.canonical_form == other.canonical_form
        
    #def __ne__(self, other: TechnologicalLimitation) -> bool:
    #    return self.canonical_form != other.canonical_form

    #def __lt__(self, other: TechnologicalLimitation) -> bool:
    #    return self <= other and self != other
        
    #def __le__(self, other: TechnologicalLimitation) -> bool:
    #    return other >= self
        
    #def __gt__(self, other: TechnologicalLimitation) -> bool:
    #    return self >= other and self != other
        
    def __ge__(self, other: TechnologicalLimitationOld) -> bool: #only permitted and only needed comparison, "Does having this unlocked mean that other is unlocked?"
        if len(other.canonical_form)==0 and len(self.canonical_form)!=0:
            return True
        if len(self.canonical_form)==0 and len(other.canonical_form)!=0:
            return False
        for nodes_self in self.canonical_form:
            if not any([nodes_self >= nodes_other for nodes_other in other.canonical_form]):
                return False
        return True

def list_union(l1: list[T], l2: list[T]) -> list[T]:
    """
    Union operation between lists. Used for when we cant freeze dicts into a set.

    Parameters
    ----------
    l1:
        First list.
    l2:
        Second list.
    
    Returns
    -------
    Union of lists.
    """
    l3 = []
    for v in l1:
        l3.append(v)
    for v in l2:
        if not v in l3:
            l3.append(v)
    return l3

power_letters = {'k': 10**3, 'K': 10**3, 'M': 10**6, 'G': 10**9, 'T': 10**12, 'P': 10**15, 'E': 10**18, 'Z': 10**21, 'Y': 10**24}
def convert_value_to_base_units(string: str) -> Fraction:
    """
    Converts various values (power, energy, etc.) into their expanded form.

    Parameters
    ----------
    string:
        String in some sort of Factorio base unit.
    
    Returns
    -------
    Value in Fraction form.
    """
    try:
        value = Fraction(re.findall(r'[\d.]+', string)[0]).limit_denominator()
        value*= power_letters[re.findall(r'[k,K,M,G,T,P,E,Z,Y]', string)[0]]
        return value
    except:
        raise ValueError(string)

def technological_limitation_from_specification(instance, COST_MODE: str, fully_automated: list[str] = [], extra_technologies: list[str] = [], extra_recipes: list[str] = []) -> TechnologicalLimitation:
    """
    Generates a TechnologicalLimitation from a specification. Works as a more user friendly way of getting useful TechnologicalLimitations.

    Parameters
    ----------
    data:
        Entire data.raw. https://wiki.factorio.com/Data.raw
    COST_MODE:
        What cost mode is being used. https://lua-api.factorio.com/latest/concepts.html#DifficultySettings
    fully_automated:
        List of fully automated science packs.
    extra_technologies:
        List of additional unlocked technologies.
    extra_recipes:
        List of additionally unlocked recipes.
    
    Returns
    -------
    Specified TechnologicalLimitations
    """
    tech_obj = set()
    
    assert len(fully_automated)+len(extra_technologies)+len(extra_recipes) > 0, "Trying to find an empty tech limit. Likely some error."
    for pack in fully_automated:
        assert pack in instance.data_raw['tool'].keys() #https://lua-api.factorio.com/latest/prototypes/ToolPrototype.html
    for tech in extra_technologies:
        assert tech in instance.data_raw['technology'].keys() #https://lua-api.factorio.com/latest/prototypes/TechnologyPrototype.html
    for recipe in extra_recipes:
        assert recipe in instance.data_raw['recipe'].keys() #https://lua-api.factorio.com/latest/prototypes/RecipeCategory.html

    for tech in instance.data_raw['technology'].values(): #https://lua-api.factorio.com/latest/prototypes/TechnologyPrototype.html
        if COST_MODE in tech.keys():
            unit = tech[COST_MODE]['unit']
        else:
            unit = tech['unit'] #https://lua-api.factorio.com/latest/prototypes/TechnologyPrototype.html#unit
        if all([tool_name in fully_automated for tool_name in [ingred['name'] for ingred in unit['ingredients']]]):
            tech_obj.add(tech['name'])
    
    for tech in extra_technologies:
        tech_obj = tech_obj.union(next(iter(instance.data_raw['technology'][tech]['limit'].canonical_form)))
        tech_obj.add(tech)
    
    for recipe in extra_recipes:
        tech_obj = tech_obj.union(instance.data_raw['recipe'][recipe]['limit'])
    
    return TechnologicalLimitation(instance.tech_tree, [tech_obj])

def evaluate_formulaic_count(expression: str, level: int) -> int:
    """
    Evaluates the count for a formulaic expression for a level.

    Parameters
    ----------
    expression:
        Expression for the formulaic count. https://lua-api.factorio.com/latest/types/TechnologyUnit.html#count_formula
    level:
        What level to calculate at.
    
    Returns
    -------
    Count value.
    """
    fixed_expression = expression.replace("l", str(level)).replace("L", str(level)).replace("^", "**")
    fixed_expression = re.sub(r'(\d)\(', r'\1*(', fixed_expression)
    value = numexpr.evaluate(fixed_expression).item()
    if value <= 0:
        raise ValueError("Found a negative count. PANIC.")
    return value

def linear_transform_is_gt(A: np.ndarray | sparse.coo_matrix | sparse.csr_matrix, x: np.ndarray, b:np.ndarray, rel_tol=1e-5) -> np.ndarray:
    """
    Determines of a linear transformation, A, onto a contravector, x, is greater than or equal to b.
    Sometimes the tolerance has to be larger for values with huge i/o (such as electricity).

    Parameters
    ----------
    A:
        Linear transformation
    x:
        contravector assumed to be >=0
    b:
        contravector
    """
    A = A.astype(np.longdouble)
    x = x.astype(np.longdouble)
    b = b.astype(np.longdouble)

    if isinstance(A, np.ndarray):
        Ap = A.copy()
        Ap[Ap < 0] = 0
        An = A.copy()
        An[An > 0] = 0
    elif isinstance(A, sparse.coo_matrix):
        Ap = sparse.coo_matrix(([A.data[k] for k in range(A.nnz) if A.data[k] > 0],
                                ([A.row[k] for k in range(A.nnz) if A.data[k] > 0],
                                 [A.col[k] for k in range(A.nnz) if A.data[k] > 0])),
                               shape=A.shape, dtype=np.longdouble)
        An = sparse.coo_matrix(([A.data[k] for k in range(A.nnz) if A.data[k] < 0],
                                ([A.row[k] for k in range(A.nnz) if A.data[k] < 0],
                                 [A.col[k] for k in range(A.nnz) if A.data[k] < 0])),
                               shape=A.shape, dtype=np.longdouble)
    else: #isinstance(A, sparse.csr_matrix):
        Ap = A.copy()
        Ap[Ap < 0] = 0
        An = A.copy()
        An[An > 0] = 0
    
    Lhp = Ap @ x
    Lhn = An @ x

    true_tol = 1/2 * (np.abs(Lhp) + np.abs(Lhn)) * rel_tol
    Aax = A @ x
    return np.logical_or(Aax - b >= -1 * true_tol, np.logical_or(Aax >= b, np.isclose(Aax, b, rtol=SOLVER_TOLERANCES['rtol'], atol=SOLVER_TOLERANCES['atol'])))

def linear_transform_is_close(A: np.ndarray | sparse.coo_matrix | sparse.csr_matrix, x: np.ndarray, b:np.ndarray, rel_tol=1e-5) -> np.ndarray:
    """
    Determines of a linear transformation, A, onto a contravector, x, is equal to b.
    Sometimes the tolerance has to be larger for values with huge i/o (such as electricity).

    Parameters
    ----------
    A:
        Linear transformation
    x:
        contravector assumed to be >=0
    b:
        contravector
    """
    A = A.astype(np.longdouble)
    x = x.astype(np.longdouble)
    b = b.astype(np.longdouble)

    if isinstance(A, np.ndarray):
        Ap = A.copy()
        Ap[Ap < 0] = 0
        An = A.copy()
        An[An > 0] = 0
    elif isinstance(A, sparse.coo_matrix):
        Ap = sparse.coo_matrix(([A.data[k] for k in range(A.nnz) if A.data[k] > 0],
                                ([A.row[k] for k in range(A.nnz) if A.data[k] > 0],
                                 [A.col[k] for k in range(A.nnz) if A.data[k] > 0])),
                               shape=A.shape, dtype=np.longdouble)
        An = sparse.coo_matrix(([A.data[k] for k in range(A.nnz) if A.data[k] < 0],
                                ([A.row[k] for k in range(A.nnz) if A.data[k] < 0],
                                 [A.col[k] for k in range(A.nnz) if A.data[k] < 0])),
                               shape=A.shape, dtype=np.longdouble)
    else: #isinstance(A, sparse.csr_matrix):
        Ap = A.copy()
        Ap[Ap < 0] = 0
        An = A.copy()
        An[An > 0] = 0
    
    Lhp = Ap @ x
    Lhn = An @ x

    true_tol = 1/2 * (np.abs(Lhp) + np.abs(Lhn)) * rel_tol
    Aax = A @ x
    return np.logical_or(np.abs(Aax - b) <=  true_tol, np.isclose(Aax, b, rtol=SOLVER_TOLERANCES['rtol'], atol=SOLVER_TOLERANCES['atol']))

def find_zeros(R_j_i: np.ndarray, s_i: np.ndarray) -> list[int]:
    """
    Given a factory finds which items have zero throughtput

    Parameters
    ----------
    s_i:
        CompressedVector of the requested factory

    Returns
    -------
    List of indexes of reference_list that have zero throughput
    """
    R_j_i = R_j_i.copy()
    R_j_i[R_j_i < 0] = 0
    return np.where(np.isclose(R_j_i @ s_i, 0, rtol=SOLVER_TOLERANCES['rtol'], atol=SOLVER_TOLERANCES['atol']))[0].tolist()

def vectors_orthant(v: np.ndarray | sparse.sparray | list) -> Hashable:
    """
    Determines the orthant a vector is in.

    Parameters
    ----------
    v:
        Some vector (1 dimensional array).
    
    Returns
    -------
    The orthant that vector is in.
    """
    if isinstance(v, np.ndarray):
        assert len(v.shape)==1
        return tuple(np.sign(v))
    if isinstance(v, sparse.coo_array) or isinstance(v, sparse.coo_matrix):
        return tuple(v.tocsr().sign().todense()[0]) # type: ignore
    if isinstance(v, sparse.csr_array) or isinstance(v, sparse.csr_matrix):
        return tuple(v.sign().toarray()[0]) # type: ignore
    if isinstance(v, list):
        return tuple(np.sign(np.array(v)))

def pareto_frontier(l: list[sparse.coo_array]) -> np.ndarray:
    """
    Returns a mask on l of just the elements in the pareto frontier.
    """
    if len(l)==0:
        return np.array([])
    revelent_points = np.vstack([p.data for p in l])
    mask = np.full(revelent_points.shape[0], True, dtype=bool)
    for i in range(revelent_points.shape[0]):
        if mask[i]:
            #if len(l)>5:
            #    print(len(l))#10
            #    print(mask.shape)#(10,)
            #    print(l[0].shape)#(397, 1)
            #    print(revelent_points.shape)#(10, 2)
            #    print(revelent_points[i, :].shape)#(2,)
            #    print((revelent_points > revelent_points[i, :]).shape)#(10,2)
            #    print(np.any(revelent_points > revelent_points[i, :], axis=1).shape)#(10,)
            mask = np.logical_and(mask, np.any(revelent_points > revelent_points[i, :], axis=1))
            mask[i] = True #mask[i] will compute to false in last line because its not strictly greater than every point in self.
    
    return np.where(mask)[0]

def beacon_setups_old(building: dict, beacon: dict) -> list[tuple[int, Fraction]]:
    """
    Determines the possible optimal beacon setups.

    Parameters
    ----------
    building:
        Building being buffed by the beacon.
    beacon:
        Beacon buffing the building.
    
    Returns
    -------
    list_of_setups:
        list of tuples with beacons hitting each building and beacons/building in the crystal.
    """
    try:
        M_plus = max(building['tile_width'], building['tile_height'])
        M_minus = min(building['tile_width'], building['tile_height'])
    except:
        raise ValueError(building['name'])
    B_plus = max(beacon['tile_width'], beacon['tile_height'])
    B_minus = min(beacon['tile_width'], beacon['tile_height'])
    E_plus = int(beacon['supply_area_distance'])*2+B_plus
    E_minus = int(beacon['supply_area_distance'])*2+B_minus

    setups = []
    #surrounded buildings: same direction
    surrounded_buildings_same_direction_side_A = math.floor((E_plus - B_plus - 2 + M_plus)*1.0/B_minus)
    surrounded_buildings_same_direction_side_B = math.floor((E_minus - B_minus - 2 + M_minus)*1.0/B_minus)
    setups.append((4+2*surrounded_buildings_same_direction_side_A+2*surrounded_buildings_same_direction_side_B,
                   -1*Fraction(2+surrounded_buildings_same_direction_side_A+surrounded_buildings_same_direction_side_B)))
    #surrounded buildings: opposite direction
    surrounded_buildings_opp_direction_side_A = math.floor((E_plus - B_plus - 2 + M_minus)*1.0/B_minus)
    surrounded_buildings_opp_direction_side_B = math.floor((E_minus - B_minus - 2 + M_plus)*1.0/B_minus)
    setups.append((4+2*surrounded_buildings_opp_direction_side_A+2*surrounded_buildings_opp_direction_side_B,
                   -1*Fraction(2+surrounded_buildings_opp_direction_side_A+surrounded_buildings_opp_direction_side_B)))
    #optimized rows: beacons long way
    setups.append((2*math.ceil((1+math.ceil((E_plus-1)*1.0/M_minus))*1.0/math.ceil(B_plus*1.0/M_minus)),
                   -1*Fraction(1, math.ceil(B_plus*1.0/M_minus))))
    #optimized rows: beacons short way
    setups.append((2*math.ceil((1+math.ceil((E_minus-1)*1.0/M_minus))*1.0/math.ceil(B_minus*1.0/M_minus)),
                   -1*Fraction(1, math.ceil(B_minus*1.0/M_minus))))
    
    mask = [True]*4
    for i in range(4): #ew
        for j in range(4):
            if i!=j:
                if (setups[i][0] >= setups[j][0] and setups[i][1] > setups[j][1]) or (setups[i][0] > setups[j][0] and setups[i][1] >= setups[j][1]):
                    mask[j] = False
    filt_setups = []
    for i in range(4):
        if mask[i]:
            filt_setups.append(setups[i])

    return list(set(filt_setups))

def sparse_addition_broadcasting(A, b):
    """
    Adds sparse vector b to A. Retains A's format.
    https://stackoverflow.com/questions/30741461/elementwise-addition-of-sparse-scipy-matrix-vector-with-broadcasting
    """
    An = A.tolil()
    tp = b.tocoo()
    for i, v in zip(tp.col, tp.data):
        An[:, i] = sparse.coo_matrix(An[:, i].A + v)
    return An.asformat(A.getformat())




