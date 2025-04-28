from pyomo.environ import *

"""
Parameterisation:

This is what we kinda want our data to look like if we took this approach

Note: I made up this data
"""

import numpy as np
import pandas as pd

# Problem setup
H = 24  # Number of hours
I_E = ["G1", "G2", "G3"]  #Energy generators
I_S = ["S1", "S2"]  #Storage systems
I_G = ["G1", "G3"]  #Plants in geographic region g (we define region as DK1, DK2, norway, sweden etc.)
S_g = ["L1", "L2"]  #Lines connected to g
np.random.seed(42)  #For reproducibility -> as long as we use the same seed

# Parameters
ll = {"L1": 0.02, "L2": 0.03} #Fixed loss on line thru export, will change this
q_line = {"L1": 500, "L2": 600} 
u = 50  #Utility of served demand -> We HAVE TO ASSUME THAT THIS IS FIXED
c = {"G1": 20, "G2": 25, "G3": 30, "S1": 10, "S2": 12, "L1": 5, "L2": 6}  #Cost coefficients
FOM = {"G1": 100, "G2": 120, "G3": 140, "L1": 50, "L2": 60}  #Fixed operating costs
q = {"G1": 300, "G2": 250, "G3": 400, "S1": 200, "S2": 150, "L1": 500, "L2": 600}  #Capacity
gamma = pd.DataFrame(np.random.uniform(0.8, 1.2, (H, len(I_E))), columns=I_E)  #Scaling factors
L_g = np.random.randint(500, 700, H)  #Maximum demand per hour
eta = {"S1": 0.9, "S2": 0.85}  #Efficiency of storage
S_init = {"S1": 100, "S2": 80}  #Initial storage
S_max = {"S1": 400, "S2": 350}  #Maximum storage capacity

#Variables
# D_h: Random demand
D_h = np.random.randint(400, 600, H)

#Made decision variables random till we hava daaaaaata
E = pd.DataFrame(np.random.uniform(0, q["G1"], (H, len(I_E))), columns=I_E)
Y_d = pd.DataFrame(np.random.uniform(0, q["S1"], (H, len(I_S))), columns=I_S)
Y_C = pd.DataFrame(np.random.uniform(0, q["S1"], (H, len(I_S))), columns=I_S)
D_g = np.random.uniform(300, 600, H)
x = pd.DataFrame(np.random.uniform(0, q["L1"], (H, len(S_g))), columns=S_g)
S = pd.DataFrame(np.random.uniform(0, S_max["S1"], (H, len(I_S))), columns=I_S)


data = {
    "Parameters": {
        "u": u,
        "c": c,
        "FOM": FOM,
        "q": q,
        "gamma": gamma,
        "L_g": L_g,
        "eta": eta,
        "S_init": S_init,
        "S_max": S_max,
    },
    "Variables": {
        "D_h": D_h,
        "E": E,
        "Y_d": Y_d,
        "Y_C": Y_C,
        "D_g": D_g,
        "x": x,
        "S": S,
    },
}

#Testing Rows
print("Demand (D_h):", D_h[:5])
print("\nGeneration (E):")
print(E.head())
print("\nStorage Discharge (Y_d):")
print(Y_d.head())
print("\nStorage Charge (Y_C):")
print(Y_C.head())

"""
Model:

I altered the original model such that it is easier to solve. Here, we have 2.5 constraints.
"""


model = ConcreteModel()

# Sets
model.H = RangeSet(1, H)  # Time steps
model.I_E = Set(initialize=I_E)  #Generators (overall)
model.I_S = Set(initialize=I_S)  #Generators which have storage capacity
model.I_G = Set(initialize=I_G)  #Generators in a certain region (has overlap with I_S)
model.S_g = Set(initialize=S_g)  #Subset for Line flows across geographic regions

# Variables
model.E = Var(model.I_E, model.H, within=NonNegativeReals)
model.Yd = Var(model.I_S, model.H, within=NonNegativeReals)
model.YC = Var(model.I_S, model.H, within=NonNegativeReals)
model.Dg = Var(model.H, within=NonNegativeReals)
model.x_import = Var(model.S_g, model.H, within=NonNegativeReals)  # Imported power
model.x_export = Var(model.S_g, model.H, within=NonNegativeReals)  # Exported power
model.S = Var(model.I_S, model.H, within=NonNegativeReals)

# Objective function
def objective_rule(model):
    W = sum(
        u * model.Dg[h] -
        sum(c[i] * model.E[i, h] + FOM[i] * q[i] for i in model.I_E) -
        sum(c[i] * (model.Yd[i, h] + model.YC[i, h]) for i in model.I_S) -
        sum(FOM[l] * q[l] + c[l] * model.x_export[l, h] for l in model.S_g)
        for h in model.H
    )
    return W

model.obj = Objective(rule=objective_rule, sense=maximize)

def demand_served_with_export(model, h, ll):
    """
    Demand served constraint constraint
    """
    generation_sum = sum(model.E[i, h] for i in model.I_E)
    flow_sum = sum((1 - ll[l]) * model.x_import[l, h] - model.x_export[l, h] for l in model.S_g)
    return model.Dg[h] == generation_sum + flow_sum

model.demand_served = Constraint(model.H,ll, rule=demand_served_with_export)

def import_capacity_rule(model, l, h):
    return model.x_import[l, h] <= q_line[l]

model.import_capacity = Constraint(model.S_g, model.H, rule=import_capacity_rule)

def export_capacity_rule(model, l, h):
    return model.x_export[l, h] <= q_line[l]

model.export_capacity = Constraint(model.S_g, model.H, rule=export_capacity_rule)

def energy_balance_rule(model, h):
    """
    Energy Balance constraint (1c)
    """
    return sum(model.E[i, h] for i in model.I_E) + \
           sum(model.Yd[i, h] for i in model.I_S) == \
           model.Dg[h] + sum(model.YC[i, h] for i in model.I_S)

model.energy_balance = Constraint(model.H, rule=energy_balance_rule)

def storage_dynamics_rule(model, i, h):
    """
    Storage Dynamics constraint (1g)
    """
    if h == 1:
        return model.S[i, h] == S_init[i] + sqrt(eta[i]) * model.YC[i, h] - model.YC[i, h] / sqrt(eta[i])
    else:
        return model.S[i, h] == model.S[i, h - 1] + sqrt(eta[i]) * model.YC[i, h] - model.YC[i, h] / sqrt(eta[i])

model.storage_dynamics = Constraint(model.I_S, model.H, rule=storage_dynamics_rule)

#Got to add more capacity

from pyomo.opt import SolverFactory

# Solve the model
from pyomo.opt import SolverFactory

import sys

solvername='glpk'

solverpath_folder='C:\\glpk\\w64' #does not need to be directly on c drive

solverpath_exe='C:\\glpk\\w64\\glpsol' #does not need to be directly on c drive

sys.path.append(solverpath_folder)

solver=SolverFactory(solvername,executable=solverpath_exe)

results = solver.solve(model, tee=True)

# Display results
model.display()
