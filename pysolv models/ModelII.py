from pyomo.environ import *

# Model Initialization
model = ConcreteModel()

# Sets
model.H = RangeSet(1, 24)  # 24 hours (time periods)
model.I_E = Set(initialize=["G1", "G2", "G3", "G4", "G5"])  # All generators
model.S_g = Set(initialize=["L1", "L2"])  # Transmission lines
model.I_S = Set(initialize=["Storage1", "Storage2"])  # Storage systems
model.Regions = Set(initialize=["DK1", "DK2"])  # Regions

# Subset of generators for each region
region_generators = {
    "DK1": ["G1", "G2", "G3"],
    "DK2": ["G4", "G5"]
}
model.I_G = Set(model.Regions, initialize=region_generators)

# Parameters -> ARBITRARY!
u = 0.5  # Utility weight
c_i = {"G1": 20, "G2": 25, "G3": 30, "G4": 35, "G5": 40}  # Generation cost
FOM_i = {"G1": 5, "G2": 6, "G3": 7, "G4": 8, "G5": 9}  # Fixed operating costs
q = {"G1": 300, "G2": 250, "G3": 400, "G4": 200, "G5": 350, "L1": 500, "L2": 600}  # Capacities
gamma = {  # Availability factors
    "G1": [1.0], "G2": [0.9], "G3": [0.8],
    "G4": [0.7], "G5": [0.6]
}
ll = {"L1": 0.02, "L2": 0.03}  # Line loss factors
eta = {"Storage1": 0.9, "Storage2": 0.85}  # Storage efficiency
max_storage = {"Storage1": 500, "Storage2": 400}  # Max storage capacity
L_g = [500] * 24  # Maximum demand per time period

# Variables
model.Dg = Var(model.H, within=NonNegativeReals)  # Demand served
model.E = Var(model.I_E, model.H, within=NonNegativeReals)  # Energy generated
model.x_import = Var(model.S_g, model.H, within=NonNegativeReals)  # Imported power
model.x_export = Var(model.S_g, model.H, within=NonNegativeReals)  # Exported power
model.Yc = Var(model.I_S, model.H, within=NonNegativeReals)  # Energy charged into storage
model.Yd = Var(model.I_S, model.H, within=NonNegativeReals)  # Energy discharged from storage
model.S = Var(model.I_S, model.H, within=NonNegativeReals)  # State of charge for storage

# Objective Function
def objective_function(model):
    generation_cost = sum(c_i[i] * model.E[i, h] + FOM_i[i] for i in model.I_E for h in model.H)
    line_costs = sum(model.x_import[l, h] + model.x_export[l, h] for l in model.S_g for h in model.H)
    return u * sum(model.Dg[h] for h in model.H) - (generation_cost + line_costs)

model.obj = Objective(rule=objective_function, sense=maximize)

# Constraints

# Constraint (1b): Demand served balance
def demand_served_rule(model, h):
    generation_sum = sum(model.E[i, h] for i in model.I_E)
    flow_sum = sum((1 - ll[l]) * model.x_import[l, h] - model.x_export[l, h] for l in model.S_g)
    return model.Dg[h] == generation_sum + flow_sum

model.demand_served = Constraint(model.H, rule=demand_served_rule)

# Constraint (1c): Total supply equals demand
def supply_equals_demand_rule(model, h):
    return sum(model.E[i, h] for i in model.I_E) + sum(model.Yd[s, h] for s in model.I_S) == \
           model.Dg[h] + sum(model.Yc[s, h] for s in model.I_S)

model.supply_equals_demand = Constraint(model.H, rule=supply_equals_demand_rule)

# Constraint (1d): Demand bounds
def demand_bounds_rule(model, h):
    return model.Dg[h] <= L_g[h-1]

model.demand_bounds = Constraint(model.H, rule=demand_bounds_rule)

# Constraint (1e): Line capacity constraints
def import_capacity_rule(model, l, h):
    return model.x_import[l, h] <= q[l]

model.import_capacity = Constraint(model.S_g, model.H, rule=import_capacity_rule)

def export_capacity_rule(model, l, h):
    return model.x_export[l, h] <= q[l]

model.export_capacity = Constraint(model.S_g, model.H, rule=export_capacity_rule)

# Constraint (1f): Generator capacity with availability
def generation_capacity_rule(model, i, h):
    return model.E[i, h] <= q[i] * gamma[i][h-1]

model.generation_capacity = Constraint(model.I_E, model.H, rule=generation_capacity_rule)

# Constraint (1g): Storage state of charge
def storage_state_rule(model, s, h):
    if h == 1:
        prev_charge = max_storage[s] / 2  # Initial state of charge (50%)
    else:
        prev_charge = model.S[s, h-1]
    return model.S[s, h] == prev_charge + eta[s] * model.Yc[s, h] - model.Yd[s, h] / eta[s]

model.storage_state = Constraint(model.I_S, model.H, rule=storage_state_rule)

# Constraint (1j): Storage bounds
def storage_bounds_rule(model, s, h):
    return model.S[s, h] <= max_storage[s]

model.storage_bounds = Constraint(model.I_S, model.H, rule=storage_bounds_rule)

# Region-specific constraint (example): Sum of generation in a region
def regional_generation_limit_rule(model, g, h):
    return sum(model.E[i, h] for i in model.I_G[g]) <= 800  # Example regional limit

model.regional_generation_limit = Constraint(model.Regions, model.H, rule=regional_generation_limit_rule)

import sys


#Set up solver
solvername='glpk'

solverpath_folder='C:\\glpk\\w64' #does not need to be directly on c drive

solverpath_exe='C:\\glpk\\w64\\glpsol' #does not need to be directly on c drive

sys.path.append(solverpath_folder)
solver=SolverFactory(solvername,executable=solverpath_exe)
solver.solve(model, tee="True")



if __name__ == "__main__":
    # Solve the model
    results = solver.solve(model, tee=True)

    # Display results
    print("\n=== Results ===")
    for h in model.H:
        print(f"Hour {h}: Demand Served = {model.Dg[h].value}")
        for i in model.I_E:
            print(f"    Generator {i} Output = {model.E[i, h].value}")
        for l in model.S_g:
            print(f"    Line {l} Import = {model.x_import[l, h].value}")
            print(f"    Line {l} Export = {model.x_export[l, h].value}")
        for s in model.I_S:
            print(f"    Storage {s} State = {model.S[s, h].value}")
