from __future__ import annotations

from globalsandimports import *



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
    
    def norm(self) -> CompressedVector:
        """Returns a version of self with unit length 1.

        Returns
        -------
        CompressedVector
            A version of self with unit length 1
        """
        scale = 1#float(1000000 / np.linalg.norm([float(v) for v in self.values()]))
        return CompressedVector({k: v * scale for k, v in self.items()})


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
    """A representation of the technology tree for more effective technology limitations.
    So we only need to store the 'caps' of the tree.
    We want to be able to, given a cap researches of a technological limitation, add two tech limits, subtract them, 
    and most importantly determine if all researches imply another tech limit (>=).
    The fastest way to do this is to provide a function that determines if a research is greater than, less than, equal to, or skew to another research.

    Let A and B be researches:

    A > B iff B is a prerequisite of A.

    A = B iff they are the same research.

    A < B iff A is a prerequisite of B.

    A X B (X = 'skew') iff None of the above 3 apply.
    """
    reference_map: dict[str, int]
    inverse_map: dict[int, str]
    comparison_table: np.ndarray

    def __init__(self, technology_data: Collection[dict]) -> None:
        """
        Parameters
        ----------
        technology_data : Collection[dict]
            All values of data.raw['technology']
        """        
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
        """Compares two technologies.

        Parameters
        ----------
        rA : int
            First technology id
        rB : int
            Second technology id

        Returns
        -------
        int
            A > B = 1 iff B is a prerequisite of A

            A = B = 0 iff they are the same research

            A < B = -1 iff A is a prerequisite of B

            A X B = nan (X = 'skew') iff None of the above 3 apply.
        """
        return self.comparison_table[rA, rB]

    def simplify(self, sets_of_references: Collection[Collection[int]]) -> frozenset[frozenset[int]]:
        """Simplifies a set of set of research indicies

        Parameters
        ----------
        sets_of_references : Collection[Collection[int]]
            Set of set of research indicies

        Returns
        -------
        frozenset[frozenset[int]]
            Simplifed version
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
        """Simplified a set of set of research names

        Parameters
        ----------
        sets_of_researches : Collection[Collection[str]]
            Set of set of research names

        Returns
        -------
        frozenset[frozenset[int]]
            Simplified version
        """
        sets_of_references: list[list[int]] = [[self.reference_map[research_str] for research_str in sublist] for sublist in sets_of_researches]
        return self.simplify(sets_of_references)
    
    def techlimit_ge(self, A: TechnologicalLimitation, B: TechnologicalLimitation) -> bool:
        """Determines if B will always have a research subset if A has a researched subset.

        Parameters
        ----------
        A : TechnologicalLimitation
            First research state
        B : TechnologicalLimitation
            Second research state

        Returns
        -------
        bool
            If B will always be researched if A has been
        """
        if len(A.canonical_form)==0 and len(B.canonical_form)!=0: #special case when comparing if nothing is greater than something
            return False
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
    """Shortened to a 'limit' elsewhere in the program. Represents the technologies that must be researched in order
    to unlock the specific object (recipe, machine, module, etc.).

    Members
    -------
    tree : TechnologyTree
        Tree that this limit is a part of
    canonical_form : frozenset[frozenset[int]]
        Canonical form of the research
    """
    tree: TechnologyTree
    canonical_form: frozenset[frozenset[int]]

    def __init__(self, tree: TechnologyTree, sets_of_researches: Collection[Collection[str]] = [], sets_of_references: Collection[Collection[int]] = []) -> None:
        """
        Parameters
        ----------
        tree : TechnologyTree
            Tree to base this limit on
        sets_of_researches : Collection[Collection[str]], optional
            A set of possible sets of research names, by default []
        sets_of_references : Collection[Collection[int]], optional
            A set of possible sets of research ids, by default []
        """        
        assert len(sets_of_researches)==0 or len(sets_of_references)==0, "Can't take both input types"
        self.tree = tree
        if len(sets_of_references)==0:
            self.canonical_form = self.tree.reference(sets_of_researches)
        else:
            self.canonical_form = self.tree.simplify(sets_of_references)
        
    def __repr__(self) -> str:
        if len(self.canonical_form)==0:
            return "No-tech"
        out = ""
        for outer in self.canonical_form:
            out += "Technology set of:"
            for inner in outer:
                out += "\n\t"+self.tree.inverse_map[inner]
                
        return out

    def __add__(self, other: TechnologicalLimitation) -> TechnologicalLimitation:
        """Addition betwen two TechnologicalLimitations, an AND operation between disjunctive normal forms

        Parameters
        ----------
        other : TechnologicalLimitation
            Research status to add to self

        Returns
        -------
        TechnologicalLimitation
            Result of the addition
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

    def __sub__(self, other: TechnologicalLimitation) -> TechnologicalLimitation:
        """Possible additions other to research this research status

        Parameters
        ----------
        other : TechnologicalLimitation
            Research being added to

        Returns
        -------
        TechnologicalLimitation
            Possible additions
        """
        raise NotImplementedError("TODO")
        return [frozenset(nodes.difference(other)) for nodes in self.canonical_form]
        
    def __ge__(self, other: TechnologicalLimitation) -> bool: #, "Does having this unlocked mean that other is unlocked?"
        """Does having self unlocked mean that other is unlocked,
        the only permitted and only needed comparison

        Parameters
        ----------
        other : TechnologicalLimitation
            What is being compared to

        Returns
        -------
        bool
            If having self means that other has been unlocked
        """        
        assert self.tree == other.tree
        return self.tree.techlimit_ge(self, other)

T = TypeVar('T')
def list_union(l1: list[T], l2: list[T]) -> list[T]:
    """Union operation between lists, Used for when T is mutable

    Parameters
    ----------
    l1 : list[T]
        First list
    l2 : list[T]
        Second list

    Returns
    -------
    list[T]
        Union of lists
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
    """Converts various values (power, energy, etc.) into their expanded form (all zeros included)

    Parameters
    ----------
    string : str
        String in some sort of Factorio base unit

    Returns
    -------
    Fraction
        Value in Fraction form

    Raises
    ------
    ValueError
        String cannot be understood
    """
    try:
        value = Fraction(re.findall(r'[\d.]+', string)[0]).limit_denominator()
        value*= power_letters[re.findall(r'[k,K,M,G,T,P,E,Z,Y]', string)[0]]
        return value
    except:
        raise ValueError(string)

def technological_limitation_from_specification(instance, fully_automated: list[str] = [], extra_technologies: list[str] = [], extra_recipes: list[str] = []) -> TechnologicalLimitation:
    """Generates a TechnologicalLimitation from a specification

    Parameters
    ----------
    instance : FactorioInstance
        Instance for which the tech limit should be made
    fully_automated : list[str], optional
        List of fully automated science packs, by default []
    extra_technologies : list[str], optional
        List of additional unlocked technologies, by default []
    extra_recipes : list[str], optional
        List of additionally unlocked recipes, by default []

    Returns
    -------
    TechnologicalLimitation
        Specified research status
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
        if instance.COST_MODE in tech.keys():
            unit = tech[instance.COST_MODE]['unit']
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
    """Evaluates the count for a formulaic expression at a level

    Parameters
    ----------
    expression : str
        Expression for the formulaic count https://lua-api.factorio.com/latest/types/TechnologyUnit.html#count_formula
    level : int
        What level to calculate at

    Returns
    -------
    int
        Count value

    Raises
    ------
    ValueError
        If count is found to be less than zero
    """
    fixed_expression = expression.replace("l", str(level)).replace("L", str(level)).replace("^", "**")
    fixed_expression = re.sub(r'(\d)\(', r'\1*(', fixed_expression)
    value = numexpr.evaluate(fixed_expression).item()
    if value <= 0:
        raise ValueError("Found a negative count. PANIC.")
    return value

def linear_transform_is_gt(A: np.ndarray | sparse.coo_matrix | sparse.csr_matrix, x: np.ndarray, b:np.ndarray, rel_tol=1e-5) -> np.ndarray:
    """Determines of a linear transformation, A, used on contravector x, is approximately greater than or equal to contravector b

    Parameters
    ----------
    A : np.ndarray | sparse.coo_matrix | sparse.csr_matrix
        Linear Transform
    x : np.ndarray
        First contravector
    b : np.ndarray
        Second contravector
    rel_tol : _type_, optional
        tolerance value, by default 1e-5

    Returns
    -------
    np.ndarray
        Mask of where A @ x ~>= b
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
    """Determines of a linear transformation, A, used on contravector x, is approximately equal to contravector b

    Parameters
    ----------
    A : np.ndarray | sparse.coo_matrix | sparse.csr_matrix
        Linear Transform
    x : np.ndarray
        First contravector
    b : np.ndarray
        Second contravector
    rel_tol : _type_, optional
        tolerance value, by default 1e-5

    Returns
    -------
    np.ndarray
        Mask of where A @ x ~== b
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
    """Given a linear transformation used on a contravector, find which output indicies have zero throughput (not zero due to subtraction)

    Parameters
    ----------
    R_j_i : np.ndarray
        Linear transformation
    s_i : np.ndarray
        Contravector

    Returns
    -------
    list[int]
        List of indexes that have zero throughput
    """
    R_j_i = R_j_i.copy()
    R_j_i[R_j_i < 0] = 0
    return np.where(np.isclose(R_j_i @ s_i, 0, rtol=SOLVER_TOLERANCES['rtol'], atol=SOLVER_TOLERANCES['atol']))[0].tolist()

def vectors_orthant(v: np.ndarray | sparse.sparray | list) -> Hashable:
    """Determines the orthant a vector is in

    Parameters
    ----------
    v : np.ndarray | sparse.sparray | list
        The vector

    Returns
    -------
    Hashable
        The orthant that vector is in

    Raises
    ------
    ValueError
        If v cannot be identified
    """
    if isinstance(v, np.ndarray):
        assert len(v.shape)==1
        return tuple(np.sign(v))
    elif isinstance(v, sparse.coo_array) or isinstance(v, sparse.coo_matrix):
        return tuple(v.tocsr().sign().todense()[0]) # type: ignore
    elif isinstance(v, sparse.csr_array) or isinstance(v, sparse.csr_matrix):
        return tuple(v.sign().toarray()[0]) # type: ignore
    elif isinstance(v, list):
        return tuple(np.sign(np.array(v)))
    else:
        raise ValueError("Unknown type: "+str(type(v)))

def pareto_frontier(l: list[sparse.coo_array]) -> np.ndarray:
    """Finds the pareto frontier of coo_arrays

    Parameters
    ----------
    l : list[sparse.coo_array]
        List of vectors

    Returns
    -------
    np.ndarray
        Indicies of the list that are pareto frontier elements
    """
    raise DeprecationWarning("Pareto frontier not updated")
    if len(l)==0:
        return np.array([])
    revelent_points = np.vstack([p.data for p in l])
    mask = np.full(revelent_points.shape[0], True, dtype=bool)
    for i in range(revelent_points.shape[0]):
        if mask[i]:
            mask = np.logical_and(mask, np.any(revelent_points > revelent_points[i, :], axis=1))
            mask[i] = True #mask[i] will compute to false in last line because its not strictly greater than every point in self.
    
    return np.where(mask)[0]



