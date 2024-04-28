import pytest

from globalsandimports import *
from utils import *
from tools import *
from lpproblems import *

def test_compressed_vectors():
    compA = CompressedVector({"A": 1, "B": 2, "C": 3, "D": 5, "E": 7})
    compB = CompressedVector({"A": 2, "E": 1, "C": -1, "F": 2, "G": -4})
    added = CompressedVector({"A": 3, "B": 2, "C": 2, "D": 5, "E": 8, "F": 2, "G": -4})
    added_via_func = compA + compB
    print(added_via_func)
    print(added)
    assert added==added_via_func
    multiA = CompressedVector({"A": 2, "B": 4, "C": 6, "D": 10, "E": 14})
    multiA_via_func = compA * Fraction(2)
    assert multiA==multiA_via_func


@pytest.fixture
def tech_limit_testing_limits():
    base_limits = [TechnologicalLimitation([[tech]]) for tech in "A,B,C,D,E,F,G,H,I,J,K,L".split(",")]+[TechnologicalLimitation()]
    advanced_limits = [base_limits[0]+base_limits[1]+base_limits[3],
                       base_limits[2]+base_limits[3],
                       base_limits[4]+base_limits[2]+base_limits[6]+base_limits[11],
                       base_limits[0]+base_limits[10],
                       base_limits[6]+base_limits[4]+base_limits[3],
                       base_limits[0]+base_limits[9]+base_limits[9]]
    return base_limits, advanced_limits

def test_tech_limit(tech_limit_testing_limits):
    base_limits, advanced_limits = tech_limit_testing_limits
    assert advanced_limits[0] >= base_limits[0]
    assert advanced_limits[0] >= base_limits[0]
    assert advanced_limits[2] >= base_limits[11]
    assert advanced_limits[2] >= base_limits[11]
    assert advanced_limits[5] >= base_limits[9]
    assert advanced_limits[5] >= base_limits[9]
    assert advanced_limits[1] >= advanced_limits[1] 
    assert advanced_limits[1] >= advanced_limits[1]
    for bl in base_limits[:-1]:
        assert bl >= base_limits[-1]


def test_list_union():
    l1 = ["A", "B", "C", "D"]
    l2 = ["E", "F", "A", "B", "G"]
    actual = ["A", "B", "C", "D", "E", "F", "G"]
    via_func = list_union(l1, l2)

    assert set(actual)==set(via_func)
    

def test_convert_value_to_base_units():
    values = ["112k", "132K", "12M", "1.4M", "3.22E", "4Z"]
    real_values = [112000, 132000, 12000000, 1400000, 3220000000000000000, 4000000000000000000000]
    via_func = [convert_value_to_base_units(val) for val in values]

    assert all([rv==vf for rv, vf in zip(real_values, via_func)]), via_func


def test_vanilla_instance():
    """
    ident: str
    drain: CompressedVector
    deltas: CompressedVector
    effect_effects: dict[str, list[str]]
    allowed_modules: list[tuple[str, bool, bool]]
    internal_module_limit: int
    base_inputs: CompressedVector
    cost: CompressedVector
    limit: TechnologicalLimitation
    base_productivity: Fraction
    universal_reference_list: list[str]
    catalyzing_deltas: list[str]
    """
    filename = 'vanilla-rawdata.json'
    vanilla = FactorioInstance(filename)
    for construct in vanilla.uncompiled_constructs:
        assert isinstance(construct.ident, str), construct.ident
        assert isinstance(construct.drain, CompressedVector), construct.ident
        for k, v in construct.drain.items():
            assert isinstance(k, str), construct.ident
            assert isinstance(v, Fraction), construct.ident
        assert isinstance(construct.deltas, CompressedVector), construct.ident
        for k, v in construct.deltas.items():
            assert isinstance(k, str), construct.ident
            assert isinstance(v, Fraction), construct.ident
        assert isinstance(construct.effect_effects, dict), construct.ident
        for k, v in construct.effect_effects.items():
            assert isinstance(k, str), construct.ident
            assert isinstance(v, list), construct.ident
            for e in v:
                assert isinstance(e, str), construct.ident
                assert e in construct.deltas.keys(), construct.ident
        assert isinstance(construct.allowed_modules, list), construct.ident
        for am in construct.allowed_modules:
            assert isinstance(am[0], str), construct.ident
            assert isinstance(am[1], bool), construct.ident
            assert isinstance(am[2], bool), construct.ident
        assert isinstance(construct.internal_module_limit, int), construct.ident
        assert isinstance(construct.base_inputs, CompressedVector), construct.ident
        for k, v in construct.base_inputs.items():
            assert isinstance(k, str), construct.ident
            assert isinstance(v, Fraction), construct.ident
        assert isinstance(construct.cost, CompressedVector), construct.ident
        for k, v in construct.cost.items():
            assert isinstance(k, str), construct.ident
            assert isinstance(v, Fraction), construct.ident
        assert isinstance(construct.limit, TechnologicalLimitation), construct.ident
        assert isinstance(construct.base_productivity, Fraction), construct.ident
    
    vanilla.compile()
    assert isinstance(vanilla.compiled, ComplexConstruct)
    all_tech = vanilla.technological_limitation_from_specification(fully_automated=[k for k in vanilla.data_raw['tool'].keys()])
    for _ in range(25):
        p_j = np.random.rand(len(vanilla.reference_list))
        p = CompressedVector()
        for j in range(len(vanilla.reference_list)):
            if np.random.rand() >= .02:
                p[j] = p_j[j]
        vanilla.compiled.compile(p_j, list(range(len(vanilla.reference_list))), all_tech)
        A, c, N1, N0, R = vanilla.compiled.reduce(p_j, list(range(len(vanilla.reference_list))), all_tech)
        assert isinstance(A, sparse.spmatrix)
        assert A.shape[1]==c.shape[0]
        assert A.shape[0]==len(vanilla.reference_list)
        assert isinstance(c, np.ndarray)
        assert isinstance(N1, sparse.spmatrix)
        #assert isinstance(N0, sparse.sparray)

def test_solvers():
    successful = False
    for _ in range(100):
        m = np.random.randint(5, 15)
        n = np.random.randint(5, 15)
        A_dense = np.random.rand(m, n)
        rows, cols = np.nonzero(A_dense)
        data = A_dense[rows, cols]
        A = sparse.coo_matrix((data, (rows, cols)), shape=A_dense.shape)
        b = np.random.rand(m)
        c = np.random.rand(n)
        results = []
        for solver in PRIMARY_LP_SOLVERS+BACKUP_LP_SOLVERS:
            results.append(solver(A, b, c))
            assert results[-1] is None or linear_transform_is_close(A, results[-1], b).all()
        for i, j in itertools.combinations(range(len(results)), 2):
            assert (results[i]==None)==(results[j]==None), str(i)+" "+str(j)
            if not results[i] is None:
                successful = True
                assert np.max(np.abs(results[i]-results[j]))<1e-5, results[i]-results[j]
    #assert successful
                
def test_vectors_orthant():
    v1 = [1,2,-3,4]
    v2 = np.array([5,6,-7,24])
    v3 = sparse.csr_array([1,0,-2,1])
    v4 = [-1,2,3,4]
    v5 = np.array([5,6,37,-24])
    v6 = sparse.csr_array([-1,0,3,4])
    assert vectors_orthant(v1)==vectors_orthant(v2)
    assert vectors_orthant(v1)==vectors_orthant(v3)
    assert vectors_orthant(v1)!=vectors_orthant(v4)
    assert vectors_orthant(v1)!=vectors_orthant(v5)
    assert vectors_orthant(v1)!=vectors_orthant(v6)
    assert vectors_orthant(v4)!=vectors_orthant(v5)
    assert vectors_orthant(v4)==vectors_orthant(v6)
    assert vectors_orthant(v5)!=vectors_orthant(v6)

def test_pareto_frontier():
    return