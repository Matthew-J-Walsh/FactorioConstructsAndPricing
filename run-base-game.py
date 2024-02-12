import json
import logging
logging.basicConfig(level=logging.INFO)
from utils import *
from generators import *
from constructs import *
from solver import *

data = None

with open('vanilla-rawdata.txt') as f:
    data = json.load(f)

complete_premanagement(data)
all_constructs = generate_all_constructs(data)

all_families, reference_list = generate_all_construct_families(all_constructs)

out = ""
for e in reference_list:
    out += e + ","
print("=/=/=/=/=/==\=\=\=\=\=")
print(out)
print("=/=/=/=/=/==\=\=\=\=\=")

method_to_use = "ipopt"

tech_limit = tech_objection_via_spec({"fully_automated": ['automation-science-pack']}, data)
#goals = {'automation-science-pack': 1}
goals = {'automation-science-pack': 1, 'logistic-science-pack': 1}#, 'chemical-science-pack': 1, 'production-science-pack': 1, 'utility-science-pack': 1
pricing_model = {}
for k in reference_list:
    pricing_model.update({k: 1})
active_R_i_b, active_phi_i_b, active_thetas_b, active_s_i_b, active_p_i_b = optimize_for_outputs_via_reference_model(all_families, reference_list, goals, tech_limit, pricing_model, method_to_use)
active_R = np.concatenate([np.array([r.apply(theta)[0]]) for r, theta in zip(active_R_i_b, active_thetas_b)], axis=0)
print(len(active_p_i_b))
active_u_j = np.zeros(len(reference_list))
for k, v in goals.items():
    active_u_j[reference_list.index(k)] = v
prebuilt_p_i, prebuilt_R = calculate_pricing_model_via_prebuilt(active_R_i_b, active_u_j, method_to_use, universal_reference_list=reference_list)

print("=+-=+-=+-=+-=+-=+-=+-=Full results=-+=-+=-+=-+=-+=-+=-+=-+=")
print("active_p_i_b:")
print(active_p_i_b)
print("prebuilt_p_i:")
print(prebuilt_p_i)
print()
print(np.sum(active_u_j))
print()
for i in range(prebuilt_p_i.shape[0]):
    if not np.isclose(prebuilt_p_i[i], 0):
        print(active_p_i_b[i])
        print(prebuilt_p_i[i])

print(np.sum(active_R,axis=0))
print(np.sum(prebuilt_R,axis=0))
print(active_u_j)
for i in range(active_u_j.shape[0]):
    if active_u_j[i]==1:
        print(i)
        print(np.sum(active_R,axis=0)[i])
        print(np.sum(prebuilt_R,axis=0)[i])