import numpy as np
import cvxpy as cp

def get_contract_customtypes(thetas = [1,2,3],
                             TAU =3, n_ev=1, ALPHA_D = 11,
                             GAMMA=0.5, 
                             KAPPA = 0.05,
                             BAT_DEG=2*80
                             ):
    M = len(thetas)
    PI_M = np.ones(M) * (1/M)

    # VARS
    Y = cp.Variable(M, nonneg=True) # Pay (g)
    Z = cp.Variable(M, nonneg=True) # Energy (w)

    constraints = []
    
    for idx in range(M):
        theta_m = thetas[idx]
        
        if (idx == 0): # type-1
            constraints += [Y[idx] - ((Z[idx] * GAMMA)/(BAT_DEG * theta_m)) == 0]
            constraints += [Z[idx] >= 0]
            #constraints += [Y[idx] >= 0]

        elif (idx >= 1):
            constraints += [(Y[idx] - ((Z[idx] * GAMMA)/(BAT_DEG * theta_m))) == (Y[idx-1] - ((Z[idx-1] * GAMMA)/(BAT_DEG * theta_m)))]
            #constraints += [Y[idx] >= Y[idx-1]]
            #constraints += [Z[idx] <= (ALPHA_D * TAU)]
            constraints += [Z[idx] >= Z[idx-1]]

    constraints += [Z[-1] <= (ALPHA_D  * TAU)]          
            
    objective_func = 0
    for idx in range(M):
        objective_func += PI_M[idx] * n_ev * (KAPPA*(cp.log(Z[idx]+1)) - (Y[idx]))

    obj = cp.Maximize(objective_func)
    prob = cp.Problem(obj, constraints)

    prob.solve(verbose=False)
    if prob.status != 'optimal':
        raise Exception("Optimal contracts not found")
    return Y.value, Z.value

def get_contract(num_types, TAU =3, n_ev=5, ALPHA_D = 11, GAMMA=0.5, KAPPA = 0.05, BAT_DEG=2*80):   
    thetas = list(range(1,num_types+1))
    return get_contract_customtypes(thetas, TAU=TAU, n_ev=n_ev, ALPHA_D=ALPHA_D, GAMMA=GAMMA, KAPPA=KAPPA, BAT_DEG=BAT_DEG)


def get_contract_saidur(num_types, TAU =3, n_ev=5, ALPHA_D = 11, DAY_HRS=24, GAMMA=0.5, KAPPA = 0.05, V_BATT = 150, BAT_DEG=2*80):   
    M = num_types
    PI_M = np.ones(M) * (1/M)

    # VARS
    Y = cp.Variable(M, nonneg=True) # Pay (g)
    Z = cp.Variable(M, nonneg=True) # Energy (w)

    constraints = []
    
    for idx in range(M):
        theta_m = idx + 1
        
        if (idx == 0): # type-1
            constraints += [Y[idx] - ((Z[idx] * GAMMA)/(BAT_DEG * theta_m)) == 0]
            constraints += [Y[idx] >= 0]
            constraints += [Z[idx] >= 0]

        elif (idx >= 1):
            constraints += [(Y[idx] - ((Z[idx] * GAMMA)/(BAT_DEG * theta_m))) == (Y[idx-1] - ((Z[idx-1] * GAMMA)/(BAT_DEG * theta_m)))]
            constraints += [Y[idx] >= Y[idx-1]]
            constraints += [Z[idx] >= Z[idx-1]]
            constraints += [Z[idx] <= (ALPHA_D  * TAU)]          
            
    objective_func = np.zeros(M)

    for idx in range(M):
        objective_func += cp.sum((PI_M[idx]) * n_ev * (KAPPA*(cp.log(Z[idx]+1)) - (Y[idx])))

    obj = cp.Maximize(cp.sum(objective_func))
    prob = cp.Problem(obj, constraints)

    prob.solve(verbose=False)
    return Y.value, Z.value


####
def get_contract_non_tract(num_types, TAU=3, n_ev=5, ALPHA_D=11, DAY_HRS=24, GAMMA=0.5, KAPPA=0.05, V_BATT=150, BAT_DEG=2*80): 

    
    M = num_types
    PI_M = (np.ones(M)*(1/M))

    # VARS
    Y = cp.Variable(M, nonneg=True)
    Z = cp.Variable(M, nonneg=True)

    constraints = []

    for idx in range(M):
        theta_m = idx + 1
        constraints += [(Y[idx] - ((Z[idx] * GAMMA)/(BAT_DEG * theta_m))) >= 0]
        if(idx == 0):
            constraints += [Y[idx] >= 0]
            constraints += [Z[idx] >= 0]
        elif (idx >= 1):
            constraints += [Y[idx] >= Y[idx-1]]
            constraints += [Z[idx] >= Z[idx-1]]
            constraints += [Z[idx] <= (ALPHA_D * TAU)]

        for idx2 in range(M):
            theta_n = idx2 + 1
            if(idx2 != idx):
                constraints += [((Y[idx]) - ((Z[idx] * GAMMA)/(BAT_DEG * theta_m))) >= ((Y[idx2]) - ((Z[idx2] * GAMMA)/(BAT_DEG * theta_m)))]

    objective_func = np.zeros(M)

    for idx in range(M):
        objective_func += cp.sum((PI_M[idx]) * n_ev * (KAPPA*(cp.log(Z[idx]+1)) - (Y[idx])))

    obj = cp.Maximize(cp.sum(objective_func))
    prob = cp.Problem(obj, constraints)

    
    prob.solve(verbose=False, solver=cp.MOSEK)
    
    return Y.value, Z.value # (end-start)

def l_transpose(data):
    return list(map(list, zip(*data))) # Transpose 
