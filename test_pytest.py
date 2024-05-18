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
    vanilla.compile()

def test_solvers():
    successful = False
    for _ in range(100):
        m = np.random.randint(5, 15)
        n = np.random.randint(5, 15)
        A_dense = np.random.rand(m, n)
        rows, cols = np.nonzero(A_dense)
        data = A_dense[rows, cols]
        A = sparse.csr_matrix((data, (rows, cols)), shape=A_dense.shape)
        b = np.random.rand(m)
        c = np.random.rand(n)
        results = []
        for solver in [BEST_LP_SOLVER]:
            results.append(solver(A, b, c))
            assert results[-1] is None or linear_transform_is_close(A, results[-1], b).all()
        for i, j in itertools.combinations(range(len(results)), 2):
            assert (results[i]==None)==(results[j]==None), str(i)+" "+str(j)
            if not results[i] is None:
                successful = True
                assert np.max(np.abs(results[i]-results[j]))<1e-5, results[i]-results[j]
    #assert successful
                
"""def test_vectors_orthant():
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
    assert vectors_orthant(v5)!=vectors_orthant(v6)"""

def test_pareto_frontier():
    return


def test_lookup_tables_sameas():
    return # i dont want to talk about it
    filename = 'vanilla-rawdata.json'
    vanilla = FactorioInstance(filename)
    old_compiled = vanilla.compile()
    def unravel_complex_construct_to_columns_vectors(c: ComplexConstruct) -> sparse.coo_matrix:
        return sparse.coo_matrix(sparse.hstack([
                    sparse.hstack([sparse.hstack([lc.vector for lc in orth_set]) for orth_set in sc.subconstructs]) if isinstance(sc, ModulatedConstruct) 
                    else (unravel_complex_construct_to_columns_vectors(sc)) 
                    for sc in c.subconstructs]))
    def unravel_complex_construct_to_columns_costs(c: ComplexConstruct) -> sparse.coo_matrix:
        return sparse.coo_matrix(sparse.hstack([
                    sparse.hstack([sparse.hstack([lc.cost for lc in orth_set]) for orth_set in sc.subconstructs]) if isinstance(sc, ModulatedConstruct) 
                    else (unravel_complex_construct_to_columns_costs(sc)) 
                    for sc in c.subconstructs]))
    def unravel_complex_construct_names(c: ComplexConstruct) -> list[str]:
        sl = [[[lc.ident for lc in orth_set] for orth_set in sc.subconstructs] if isinstance(sc, ModulatedConstruct) 
            else [unravel_complex_construct_names(sc)] 
            for sc in c.subconstructs]
        fl = []
        for sl1 in sl:
            for sl2 in sl1:
                for e in sl2:
                    fl.append(e)
        return fl # type: ignore
    old_full_columns, old_full_costs, old_full_names = sparse.csr_matrix(unravel_complex_construct_to_columns_vectors(old_compiled).T), \
                                                                                        sparse.csr_matrix(unravel_complex_construct_to_columns_costs(old_compiled).T), \
                                                                                        unravel_complex_construct_names(old_compiled)
    assert old_full_columns.shape[0]==old_full_costs.shape[0], str(old_full_columns.shape[0])+" vs "+str(old_full_costs.shape[0])
    assert old_full_columns.shape[1]==len(vanilla.reference_list), str(old_full_columns.shape[1])+" vs "+str(len(vanilla.reference_list))
    assert old_full_costs.shape[1]==len(vanilla.reference_list), str(old_full_costs.shape[1])+" vs "+str(len(vanilla.reference_list))

    new_compiled = vanilla.compiled_constructs
    def generate_columns_from_compiled_construct_vectors(c: CompiledConstruct) -> sparse.csr_matrix:
        return c.lookup_table.effect_transform @ c.effect_transform
    def generate_columns_from_compiled_construct_costs(c: CompiledConstruct) -> sparse.csr_matrix:
        logging.info(c.ident)
        logging.info(c.lookup_table.cost_transform)
        logging.info(c.lookup_table.cost_transform.shape)
        logging.info(c.base_cost_vector)
        logging.info(c.base_cost_vector.shape)
        logging.info(sparse_addition_broadcasting(c.lookup_table.cost_transform, c.base_cost_vector.T))
        logging.info(sparse_addition_broadcasting(c.lookup_table.cost_transform, c.base_cost_vector.T).getformat())
        return sparse_addition_broadcasting(c.lookup_table.cost_transform, c.base_cost_vector.T) # type: ignore
    def generate_names_from_compiled_construct(c: CompiledConstruct) -> list[str]:
        return [c._generate_vector(i)[2] for i in range(c.lookup_table.effect_table.shape[0])]
    new_full_columns = sparse.csr_matrix(sparse.vstack([generate_columns_from_compiled_construct_vectors(cc) for cc in new_compiled]))
    new_full_costs = sparse.csr_matrix(sparse.vstack([generate_columns_from_compiled_construct_costs(cc) for cc in new_compiled]))
    new_full_names = sum([generate_names_from_compiled_construct(cc) for cc in new_compiled], [])
    logging.info(new_full_columns.shape)
    logging.info(new_full_costs.shape)
    assert new_full_columns.shape[0]==new_full_costs.shape[0], str(new_full_columns.shape[0])+" vs "+str(new_full_costs.shape[0])
    assert new_full_columns.shape[1]==len(vanilla.reference_list), str(new_full_columns.shape[1])+" vs "+str(len(vanilla.reference_list))
    assert new_full_costs.shape[1]==len(vanilla.reference_list), str(new_full_costs.shape[1])+" vs "+str(len(vanilla.reference_list))
    assert new_full_columns.shape[0]==old_full_columns.shape[0], str(new_full_columns.shape[0])+" vs "+str(old_full_columns.shape[0])

    assert len(new_full_names)==new_full_columns.shape[0], str(len(new_full_names))+" vs "+str(new_full_columns.shape[0])
    logging.info(old_full_names)
    for name in new_full_names:
        assert name in old_full_names, name

    for name in new_full_names:
        assert (old_full_columns[old_full_names.index(name)]!=new_full_columns[new_full_names.index(name)]).nnz==0, name+"\n---------------\n"+str(old_full_columns[old_full_names.index(name)])+"\n---------------\n"+str(new_full_columns[new_full_names.index(name)])
        assert (old_full_costs[old_full_names.index(name)]!=new_full_costs[new_full_names.index(name)]).nnz==0, name+"\n---------------\n"+str(old_full_costs[old_full_names.index(name)])+"\n---------------\n"+str(new_full_costs[new_full_names.index(name)])
