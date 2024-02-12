import json
import logging
logging.basicConfig(level=logging.DEBUG)
from utils import *
from generators import *
from constructs import *
from depressionsolvers import *

data = None

with open('vanilla-rawdata.txt') as f:
    data = json.load(f)

complete_premanagement(data)
all_constructs = generate_all_constructs(data)

all_families, reference_list = generate_all_construct_families(all_constructs)

method_to_use = "simplex"

testing_constructs = [UncompiledConstruct("Constuct A", {'p': -1}, {'a': 1, 'p': -10}, 
                                          {'speed': ['a'], 'productivity': ['a'], 'consumption': ['p']}, [(m,1) for m in data['module'].values()],
                                          base_inputs={'p': -11}, cost={'c': 1}, limit=TechnologicalLimitation([[]])),
                      UncompiledConstruct("Constuct B", {'p': -3}, {'b': 1, 'p': -50}, 
                                          {'speed': ['b'], 'productivity': ['b'], 'consumption': ['p']}, [(m,1) for m in data['module'].values()],
                                          base_inputs={'p': -7}, cost={'c': 1}, limit=TechnologicalLimitation([[]])),
                      UncompiledConstruct("Constuct C", {'p': -5}, {'c': 1, 'p': -100, 'a': -3, 'b': -5}, 
                                          {'speed': ['a', 'b', 'c'], 'productivity': ['c'], 'consumption': ['p']}, [(m,1) for m in data['module'].values()],
                                          base_inputs={'p': -7}, cost={'c': 1}, limit=TechnologicalLimitation([[]])),
                      UncompiledConstruct("Constuct P", {}, {'p': 100}, 
                                          {'speed': ['b'], 'productivity': ['b'], 'consumption': ['p']}, [(m,1) for m in data['module'].values()],
                                          base_inputs={}, cost={'c': 3}, limit=TechnologicalLimitation([[]])),
                      UncompiledConstruct("Constuct bA", {'p': -2}, {'a': 1, 'p': -15}, 
                                          {'speed': ['a'], 'productivity': ['a'], 'consumption': ['p']}, [(m,1) for m in data['module'].values()],
                                          base_inputs={'p': -11}, cost={'c': 1}, limit=TechnologicalLimitation([[]])),
                      UncompiledConstruct("Constuct bB", {'p': -3}, {'b': 1, 'p': -50}, 
                                          {'speed': ['b'], 'productivity': ['b'], 'consumption': ['p']}, [(m,1) for m in data['module'].values()],
                                          base_inputs={'p': -7}, cost={'c': 2}, limit=TechnologicalLimitation([[]])),
                      UncompiledConstruct("Constuct bC", {'p': -5}, {'c': 1, 'p': -100, 'a': -4, 'b': -6}, 
                                          {'speed': ['a', 'b', 'c'], 'productivity': ['c'], 'consumption': ['p']}, [(m,1) for m in data['module'].values()],
                                          base_inputs={'p': -7}, cost={'c': 1, 'a': 1}, limit=TechnologicalLimitation([[]])),
                      UncompiledConstruct("Constuct bP", {}, {'p': 88}, 
                                          {'speed': ['b'], 'productivity': ['b'], 'consumption': ['p']}, [(m,1) for m in data['module'].values()],
                                          base_inputs={}, cost={'c': 3}, limit=TechnologicalLimitation([[]])),]
#testing_family, testing_references = generate_all_construct_families(testing_constructs)
testing_family, testing_references = generate_all_construct_families_linear(testing_constructs)
starting_pricing_model = {'a': 1, 'b': 3, 'c': 30, 'p': .08}
logging.info(testing_references)
#testing_R_i, testing_phi_i, testing_thetas, testing_s_i, testing_p_i = optimize_for_outputs_via_reference_model(testing_family, testing_references, 
#                                                                                                                {'c': 1}, TechnologicalLimitation([[]]), starting_pricing_model, method_to_use)
testing_R_i_j, testing_s_i, testing_p_i = optimize_for_outputs_via_reference_model(testing_family, testing_references, 
                                                                    {'c': 1}, TechnologicalLimitation([[]]), starting_pricing_model, method_to_use)

testing_u_j = np.zeros(len(testing_references))
for k, v in {'c': 1}.items():
    testing_u_j[testing_references.index(k)] = v
p0_i = np.zeros(len(testing_references))
for k, v in starting_pricing_model.items():
    p0_i[testing_references.index(k)] = v
logging.info(p0_i)
logging.info([construct.cost.to_dense() for construct in sum([f.get_constructs(TechnologicalLimitation([[]])) for f in testing_family], [])])

testing_C_i_j = concatenate_sparse_tensors([construct.cost for construct in sum([f.get_constructs(TechnologicalLimitation([[]])) for f in testing_family], [])], 0)
print(testing_C_i_j)

print(testing_C_i_j.to_dense() @ p0_i)
testing_p_i_inf = calculate_pricing_model_via_prebuilt(testing_R_i_j, testing_C_i_j, testing_s_i, testing_u_j, testing_references.index('c'), method_to_use, universal_reference_list=testing_references)

#testing_p_i_iterative = iterative_pricing_model(testing_R_i_j, testing_s_i, testing_u_j, np.ones(), 100, method_to_use, universal_reference_list=testing_references)

print(testing_p_i)
print(testing_p_i_inf)
#print(testing_p_i_iterative)