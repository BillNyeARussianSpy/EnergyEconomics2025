from pyomo.environ import ConcreteModel, Var, Param, Objective, Constraint, NonNegativeReals, RangeSet, sum_product

def create_model(db):
    """
    Creates our ConcreteModel used for modelling (duh)
    
    Args:
        db: pandas database of the relevant data
    """
    model = ConcreteModel()

    #Sets for hours, electricity generators and storage devices
    model.H = RangeSet(1, len(db['hours']))  # Set of hours
    model.I_E = RangeSet(1, len(db['generation_e']))  # Set of generators for electricity
    model.I_S = RangeSet(1, len(db['storage_devices']))  # Set of storage devices

    # Define decision variables
    model.E = Var(model.I_E, model.H, within=NonNegativeReals)  #Energy produced by generator i at hour h
    model.Y_d = Var(model.I_S, model.H, within=NonNegativeReals)  #Discharged electricity for storage device i at hour h
    model.Y_C = Var(model.I_S, model.H, within=NonNegativeReals)  #Charged electricity for storage device i at hour h
    model.D_g = Var(model.H, within=NonNegativeReals)  #Total electricity demand at hour h
    model.x = Var(db['transmission_lines'], model.H, within=NonNegativeReals)  #Flow across transmission lines
    model.S = Var(model.I_S, model.H, within=NonNegativeReals)  #Storage level for storage device i at hour h

    #Model parameters. Note 
    model.c = Param(model.I_S, initialize=db['marginal_costs'])  #Marginal charge/discharge costs
    model.FOM = Param(model.I_E, initialize=db['fixed_op_costs'])  #Fixed operational costs
    model.q = Param(model.I_E, initialize=db['generation_capacity'])  #Maximum generation capacity
    model.q_l = Param(db['transmission_lines'], initialize=db['line_capacity'])  #Transmission line capacity
    model.eta = Param(model.I_S, initialize=db['efficiency'])  #Roundtrip efficiency for storage devices
    model.L_g = Param(model.H, initialize=db['max_demand'])  #Max allowable demand per hour
    model.S_max = Param(model.I_S, initialize=db['storage_capacity'])  #Max storage capacity per device
    model.S0 = Param(model.I_S, initialize=db['initial_storage'])  #Initial storage levels

    # Objective Function (Welfare maximization)
    def objective_rule(model):
        # W = sum_h (u * D_h) - sum_{i∈I^E}(c_i * E_{i,h} + FOM_i * q_i) - sum_{i∈I^S}(c_i * (Y_d_{i,h} + Y_C_{i,h})) - sum_l (FOM_l * q_l + sum_h (c_l * x_{l,h}))
        return sum(model.D_g[h] for h in model.H) - sum(model.c[i] * model.E[i,h] + model.FOM[i] * model.q[i] for i in model.I_E for h in model.H) \
            - sum(model.c[i] * (model.Y_d[i,h] + model.Y_C[i,h]) for i in model.I_S for h in model.H) \
            - sum(model.FOM[l] * model.q_l[l] + sum(model.c[l] * model.x[l,h] for h in model.H) for l in db['transmission_lines'])
    
    model.obj = Objective(rule=objective_rule, sense=-1)  # Maximize W, hence -1 to maximize

    # Constraints
    def demand_constraint(model, h):
        # (1b) D^g_h = sum_{i∈I_g} E_{i,h} + sum_{l∈S_g} [(1-ll) * x_{l',h} - x_{l,h}]
        return model.D_g[h] == sum(model.E[i,h] for i in model.I_E) + sum(model.x[l,h] for l in db['transmission_lines'])

    model.demand_constraint = Constraint(model.H, rule=demand_constraint)

    def storage_balance(model, i, h):
        # (1h) S_{i,h} = S_{i,h-1} + sqrt(eta_i) * Y_C_{i,h} - (Y_d_{i,h} / sqrt(eta_i))
        if h == 1:
            return model.S[i,h] == model.S0[i] + (model.eta[i]**0.5) * model.Y_C[i,h] - (model.Y_d[i,h] / model.eta[i]**0.5)
        else:
            return model.S[i,h] == model.S[i,h-1] + (model.eta[i]**0.5) * model.Y_C[i,h] - (model.Y_d[i,h] / model.eta[i]**0.5)

    model.storage_balance = Constraint(model.I_S, model.H, rule=storage_balance)

    def demand_supply_balance(model, h):
        # (1c) sum_{i∈I^E} E_{i,h} sum_{i∈I^S} Y_d_{i,h} = D^g_h + sum_{i∈I^S} Y_C_{i,h}
        return sum(model.E[i,h] for i in model.I_E) + sum(model.Y_d[i,h] for i in model.I_S) == model.D_g[h] + sum(model.Y_C[i,h] for i in model.I_S)

    model.demand_supply_balance = Constraint(model.H, rule=demand_supply_balance)

    # Bounds Constraints (capacity and storage limits)
    model.D_g_bounds = Constraint(model.H, rule=lambda model, h: (0, model.L_g[h]))
    model.Y_d_bounds = Constraint(model.I_S, model.H, rule=lambda model, i, h: (0, model.q[i]))
    model.Y_C_bounds = Constraint(model.I_S, model.H, rule=lambda model, i, h: (0, model.q[i]))
    model.S_bounds = Constraint(model.I_S, model.H, rule=lambda model, i, h: (0, model.S_max[i]))
    model.S_terminal = Constraint(model.I_S, rule=lambda model, i: model.S[i,len(db['hours'])] == model.S0[i])

    return model
