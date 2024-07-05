from __future__ import annotations

from globalsandimports import *

if TYPE_CHECKING:
    from tools import FactorioInstance


class CompressedVector(dict):
    """CompressedVector's are dicts where the values are all numbers and the values represent some dimension

    They also have hashes for easier search through collections
    """
    _hash_value: int | None

    def __init__(self, input = None) -> None: # type: ignore
        if input is None:
            super().__init__()
        else:
            super().__init__(input) # type: ignore
        self._hash_value = None

    def __add__(self, other: CompressedVector) -> CompressedVector:
        self._hash_value = None
        new_cv = CompressedVector()
        for k, v in self.items():
            new_cv.key_addition(k, v)
        for k, v in other.items():
            new_cv.key_addition(k, v)
        return new_cv

    def __mul__(self, multiplier: Number) -> CompressedVector:
        self._hash_value = None
        dn = CompressedVector()
        for k, v in self.items():
            dn[k] = multiplier * v
        return dn
    
    def __rmul__(self, multiplier: Number) -> CompressedVector:
        self._hash_value = None
        return self.__mul__(multiplier)
    
    def key_addition(self, key, value) -> None:
        """
        Single line adding key or adding to value in key.
        """
        self._hash_value = None
        if key in self.keys():
            self[key] += value
        else:
            self[key] = value
    
    def __hash__(self) -> int: # type: ignore
        if self._hash_value is None:
            self._hash_value = hash(tuple(sorted(self.items())))
        return self._hash_value
    
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
    _reference_map: dict[str, int]
    _inverse_map: dict[int, str]
    _comparison_table: np.ndarray

    def __init__(self, technology_data: Collection[dict]) -> None:
        """
        Parameters
        ----------
        technology_data : Collection[dict]
            All values of data.raw['technology']
        """        
        self._reference_map = {}
        self._inverse_map = {}
        self._comparison_table = np.full((len(technology_data), len(technology_data)), 13, dtype=float) #random number to check for at end to make sure we filled it
        for i, tech in enumerate(technology_data):
            self._reference_map.update({tech['name']: i})
            self._inverse_map.update({i: tech['name']})
            tech.update({'tech_tree_identifier': i})
            for j, other_tech in enumerate(technology_data):
                if tech['name']==other_tech['name']:
                    self._comparison_table[i, j] = 0
                elif other_tech in tech['all_prereq']:
                    self._comparison_table[i, j] = 1
                elif tech in other_tech['all_prereq']:
                    self._comparison_table[i, j] = -1
                else:
                    self._comparison_table[i, j] = float('nan')
        assert not (self._comparison_table==13).any()

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
        return self._comparison_table[rA, rB]

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
        sets_of_references: list[list[int]] = [[self._reference_map[research_str] for research_str in sublist] for sublist in sets_of_researches]
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
    _tree : TechnologyTree
        Tree that this limit is a part of
    canonical_form : frozenset[frozenset[int]]
        Canonical form of the research
    """
    _tree: TechnologyTree
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
        self._tree = tree
        if len(sets_of_references)==0:
            self.canonical_form = self._tree.reference(sets_of_researches)
        else:
            self.canonical_form = self._tree.simplify(sets_of_references)
        
    def __repr__(self) -> str:
        if len(self.canonical_form)==0:
            return "No-tech"
        out = ""
        for outer in self.canonical_form:
            out += "Technology set of:"
            for inner in outer:
                out += "\n\t"+self._tree._inverse_map[inner]
                
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
        assert self._tree == other._tree
        if len(self.canonical_form)==0:
            return other
        if len(other.canonical_form)==0:
            return self
        added = set()
        for s1 in self.canonical_form:
            for s2 in other.canonical_form:
                added.add(s1.union(s2))
        return TechnologicalLimitation(self._tree, sets_of_references=added)

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
        assert self._tree == other._tree
        return self._tree.techlimit_ge(self, other)
    
    @property
    def tech_coverage(self) -> TechnologicalLimitation:
        """Returns self

        Returns
        -------
        TechnologicalLimitation
            Self
        """
        return self

class ColumnTable:
    """Container for columns of a LP problem

    Members
    -------
    column : np.ndarray
        Columns of the problem
    costs : np.ndarray
        Costs of the problem
    true_costs : np.ndarray
        True costs of the columns
    ident : np.ndarray
        Identifiers of the columns
    """
    columns: np.ndarray
    costs: np.ndarray
    true_costs: np.ndarray
    idents: np.ndarray[CompressedVector, Any]
    _valid_rows: np.ndarray | None

    def __init__(self, columns: np.ndarray, costs: np.ndarray, true_costs: np.ndarray, idents: np.ndarray[CompressedVector, Any]):
        """
        Parameters
        ----------
        column : np.ndarray
            Columns of the problem
        costs : np.ndarray
            Costs of the problem
        true_costs : np.ndarray
            True costs of the columns
        ident : np.ndarray
            Identifiers of the columns
        """
        self.columns = columns
        self.costs = costs
        self.true_costs = true_costs
        self.idents = idents
        self._valid_rows = None

    @staticmethod
    def empty(size: int) -> ColumnTable:
        """Makes an empty column table

        Parameters
        ----------
        size : int
            Column size

        Returns
        -------
        ColumnTable
            Column table with correct size without any values
        """        
        return ColumnTable(np.zeros((size, 0)), np.zeros(0), np.zeros((size, 0)), np.zeros(0, dtype=CompressedVector))
    
    @staticmethod
    def sum(all_tables: Sequence[ColumnTable], size_backup: int | None = None) -> ColumnTable:
        """Adds a set of column tables together

        Parameters
        ----------
        all_tables : Sequence[ColumnTable]
            ColumnTables to add
        size_backup : int | None, optional
            Backup column size to use 
            should be given if all_tables could be empty, by default None

        Returns
        -------
        ColumnTable
            Resulting tables

        Raises
        ------
        ValueError
            _description_
        """        
        if len(all_tables)==0:
            if size_backup is None:
                raise ValueError("Empty Column Table and no size backup")
            return ColumnTable.empty(size_backup)
        return ColumnTable(np.concatenate([tab.columns for tab in all_tables], axis=1), np.concatenate([tab.costs for tab in all_tables]), 
                           np.concatenate([tab.true_costs for tab in all_tables], axis=1), np.concatenate([tab.idents for tab in all_tables]))
        #s: ColumnTable = all_tables[0]
        #for i in range(1, len(all_tables)):
        #    s = s + all_tables[i]
        #return s
    
    @property
    def valid_rows(self) -> np.ndarray:
        """Rows with positive output
        """        
        if self._valid_rows is None:
            self._valid_rows = (self.columns > 0).sum(axis=1) > 0
        return self._valid_rows
    
    @property
    def sorted(self) -> ColumnTable:
        """A sorted version of self for easier equality filtering
        """        
        ident_hashes = np.array([hash(ide) for ide in self.idents])
        sort_list = ident_hashes.argsort()

        return ColumnTable(self.columns[:, sort_list], self.costs[sort_list], self.true_costs[:, sort_list], self.idents[sort_list])

    def stabilize_row(self, row: int, direction: int) -> ColumnTable:
        """Stabilizes a row in a direction

        Parameters
        ----------
        row : int
            Row to stabilize
        direction : int
            Direction to stabilize

        Returns
        -------
        ColumnTable
            _description_

        Raises
        ------
        ValueError
            _description_
        """        
        if self.columns.shape[0]==0:
            return self

        if direction>0:
            violating_columns = np.where(self.columns[:, row] < 0)[0]
            unviolating_columns = np.where(self.columns[:, row] > 0)[0]
        elif direction<0:
            violating_columns = np.where(self.columns[:, row] > 0)[0]
            unviolating_columns = np.where(self.columns[:, row] < 0)[0]
        else:
            raise ValueError("TODO: Remove me. Error: direction? "+str(direction))
        
        retained_columns: ColumnTable = self.mask(unviolating_columns)

        fixed_columns: list[ColumnTable] = []
        for vcol, ucol in itertools.product(violating_columns, unviolating_columns):
            scale_factor = self.columns[vcol, row] / self.columns[ucol, row]
            ncolumn = self.columns[ucol] - scale_factor * self.columns[vcol]
            ncost = self.costs[ucol] - scale_factor * self.costs[vcol]
            ntrue_cost = self.true_costs[ucol] - scale_factor * self.true_costs[vcol]
            nident = self.idents[ucol] + -1 * scale_factor * self.idents[vcol]
            fixed_columns.append(ColumnTable(ncolumn.reshape(1, -1), np.array([ncost]), ntrue_cost.reshape(1, -1), np.array([nident])))
            #TODO: remove asserts
            assert fixed_columns[-1].columns.shape[1] == self.columns.shape[1]
            assert fixed_columns[-1].true_costs.shape[1] == self.true_costs.shape[1]
        
        result = ColumnTable.sum([retained_columns]+fixed_columns)

        assert (result.columns[:, row]!=0).sum()==0

        return result

    def __add__(self, other: ColumnTable) -> ColumnTable:
        """Adds two tables together

        Parameters
        ----------
        other : ColumnTable
            Table to add

        Returns
        -------
        ColumnTable
            Added tables
        """
        return ColumnTable(np.concatenate((self.columns, other.columns), axis=1), np.concatenate((self.costs, other.costs)), 
                           np.concatenate((self.true_costs, other.true_costs), axis=1), np.concatenate((self.idents, other.idents)))

    def mask(self, mask: np.ndarray) -> ColumnTable:
        """Returns a new ColumnTable with masked columns removed

        Parameters
        ----------
        mask : np.ndarray
            Masking array

        Returns
        -------
        ColumnTable
            New ColumnTable
        """
        return ColumnTable(self.columns[:, mask], self.costs[mask], self.true_costs[:, mask], self.idents[mask])
    
    def shadow_attachment(self, other: ColumnTable) -> ColumnTable:
        """Attaches another ColumnTable onto this one.
        The costs of the second ColumnTable will be transfered to an added row of true_cost.
        Works best when doing column operations with ManualConstructs

        Parameters
        ----------
        other : ColumnTable
            Special elements to add

        Returns
        -------
        ColumnTable
            Combined table with added true_cost row
        """        
        vector = np.concatenate((self.columns, other.columns), axis=1)
        cost = np.concatenate((self.costs, np.zeros_like(other.costs)))
        true_cost = np.concatenate((np.concatenate((self.true_costs, other.true_costs), axis=1), np.concatenate((np.zeros(self.true_costs.shape[1]), other.costs)).reshape(1, -1)), axis=0)
        assert true_cost.shape[1]==vector.shape[1]
        assert (true_cost.shape[0]-1)==vector.shape[0]
        ident = np.concatenate((self.idents, other.idents))
        return ColumnTable(vector, cost, true_cost, ident)
    
    @property
    def reduced(self) -> ColumnTable:
        """Removes columns that cannot be used because their >0 rows cannot be made, makes a copy
        """        
        logging.debug("Beginning reduction of "+str(self.columns.shape[1])+" constructs with "+str(np.count_nonzero(self.valid_rows))+" counted outputs.")
        size: int = self.columns.shape[1]
        last_size: int = self.columns.shape[1]+1
        reduced: ColumnTable = self
        while last_size!=size:
            last_size = reduced.columns.shape[1]
            reduced = reduced.mask(np.logical_not(np.asarray((reduced.columns[~reduced.valid_rows, :] < 0).sum(axis=0)).flatten()))
            size = reduced.columns.shape[1]
            logging.debug("Reduced to "+str(size)+" constructs with "+str(np.count_nonzero(reduced.valid_rows))+" counted outputs.")
    
        return reduced
    
    def find_zeros(self, cv: np.ndarray) -> list[int]:
        """Given a contravector, find which output indicies have zero throughput (not zero due to subtraction)

        Parameters
        ----------
        cv : np.ndarray
            Contravector

        Returns
        -------
        list[int]
            List of indexes that have zero throughput
        """        
        positive = self.columns.copy()
        positive[positive < 0] = 0
        return np.where(np.isclose(positive @ cv, 0, rtol=SOLVER_TOLERANCES['rtol'], atol=SOLVER_TOLERANCES['atol']))[0].tolist()
    
class ResearchTable:
    """Table of research depenedent values

    Members
    -------
    _limits : list[TechnologicalLimitation]
        Technological Levels
    _values : list[Any]
        Research dependent values
    """    
    _limits: list[TechnologicalLimitation]
    _values: list[Any]

    def __init__(self):
        """Empty init for ordering
        """        
        self._limits = []
        self._values = []
    
    def add(self, limit: TechnologicalLimitation, value: Any):
        """Adds an value to the table

        Parameters
        ----------
        limit : TechnologicalLimitation
            Tech level to add at
        element : Any
            Value to add
        """        
        if len(self._limits)!=0:
            for i in range(len(self._limits)):
                if not limit >= self._limits[i]:
                    self._limits.insert(i, limit)
                    self._values.insert(i, value)
                    return
        self._limits.append(limit)
        self._values.append(value)

    def value(self, known_technologies: TechnologicalLimitation) -> Any:
        """Evalutes the sum of bonuses up to a tech level

        Parameters
        ----------
        known_technologies : TechnologicalLimitation
            Tech level to use

        Returns
        -------
        Any
            Sum of bonuses
        """        
        return sum([value for limit, value in zip(self._limits, self._values) if known_technologies >= limit])
    
    def max(self, known_technologies: TechnologicalLimitation) -> Any:
        """Evaluates the maximum (ordered) value given the tech level

        Parameters
        ----------
        known_technologies : TechnologicalLimitation
            Tech level to use

        Returns
        -------
        Any
            Maximum tech level's associated value
        """        
        for i in range(len(self._limits)):
            if not known_technologies >= self._limits[i]:
                return self._values[i-1]
        return self._values[-1]
    
    def __iter__(self) -> Iterator[tuple[TechnologicalLimitation, Any]]:
        for limit in self._limits:
            yield limit, self.value(limit)

    def __repr__(self) -> str:
        return repr(self._limits)+"\n"+repr(self._values)



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

def technological_limitation_from_specification(instance: FactorioInstance, fully_automated: list[str] = [], extra_technologies: list[str] = [], extra_recipes: list[str] = []) -> TechnologicalLimitation:
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
    
    #assert len(fully_automated)+len(extra_technologies)+len(extra_recipes) > 0, "Trying to find an empty tech limit. Likely some error."
    for pack in fully_automated:
        assert pack in instance._data_raw['tool'].keys() #https://lua-api.factorio.com/latest/prototypes/ToolPrototype.html
    for tech in extra_technologies:
        assert tech in instance._data_raw['technology'].keys() #https://lua-api.factorio.com/latest/prototypes/TechnologyPrototype.html
    for recipe in extra_recipes:
        assert recipe in instance._data_raw['recipe'].keys() #https://lua-api.factorio.com/latest/prototypes/RecipeCategory.html

    for tech in instance._data_raw['technology'].values(): #https://lua-api.factorio.com/latest/prototypes/TechnologyPrototype.html
        if instance.COST_MODE in tech.keys():
            unit = tech[instance.COST_MODE]['unit']
        else:
            unit = tech['unit'] #https://lua-api.factorio.com/latest/prototypes/TechnologyPrototype.html#unit
        if all([tool_name in fully_automated for tool_name in [ingred['name'] for ingred in unit['ingredients']]]):
            tech_obj.add(tech['name'])
    
    for tech in extra_technologies:
        tech_obj = tech_obj.union(next(iter(instance._data_raw['technology'][tech]['limit'].canonical_form)))
        tech_obj.add(tech)
    
    for recipe in extra_recipes:
        tech_obj = tech_obj.union(instance._data_raw['recipe'][recipe]['limit'])
    
    return TechnologicalLimitation(instance._tech_tree, [tech_obj])

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



