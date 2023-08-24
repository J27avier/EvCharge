import numpy as np
import cvxpy as cp

def general_contracts(thetas_i = [1/1.25, 1/1, 1/0.75],
                      thetas_j = [1/1.25, 1/1, 1/0.75],
                      c1 = 0.01,
                      c2 = 0.1,
                      kappa1 = 0.1,
                      kappa2 = 0.5,
                      alpha_d = 11,
                      psi = 0.49,
                      my_print = False,
                      monotonicity = True,
                      integer = False,
                      IR = "all",
                      IC = "neq",
                      ):
    I = len(thetas_i)
    J = len(thetas_j)
    #G = cp.Variable((I, J), nonneg=True)
    #W = cp.Variable((I), nonneg=True)
    G = cp.Variable((I, J))
    W = cp.Variable((I))
    L = cp.Variable((J))
    
    #if integer:
    #    L = cp.Variable((J), integer=True)
    #else:
    #    L = cp.Variable((J), nonneg=True)

    PI = np.ones((I,J)) / (I*J)
    constraints = []
    objective_func = 0

    # Objective function
    for i, theta_i in enumerate(thetas_i):
        for j, theta_j in enumerate(thetas_j):
            objective_func += PI[i,j] * (kappa1 * cp.log(W[i]+1) + kappa2 * cp.log(L[j]+1) - G[i,j])

    # Individual rationality
    for i, theta_i in enumerate(thetas_i):
        for j, theta_j in enumerate(thetas_j):
            if (IR == "fst" and i == 0 and j == 0):
                constraints += [G[i,j] - c1 * W[i] / theta_i - c2 * L[j] / theta_j == 0]
            elif(IR == "all"):
                constraints += [G[i,j] - c1 * W[i] / theta_i - c2 * L[j] / theta_j >= 0]

    # Incentive compatibility
    for i, theta_i in enumerate(thetas_i):
        for j, theta_j in enumerate(thetas_j):
            for i_p in range(I):
                for j_p in range(J):
                    if (IC == "neq" and (i_p != i or j_p != j)) or\
                       (IC == "ort") and ((i == i_p and j != j_p) or (i != i_p and j == j_p)) or\
                       (IC == "ort_d") and ((i == i_p and j > j_p) or (i > i_p and j == j_p)) or\
                       (IC == "all"):
                        constraints += [G[i,j] - c1 * W[i] / theta_i - c2 * L[j] / theta_j >= G[i_p,j_p] - c1 * W[i_p] / theta_i - c2 * L[j_p] / theta_j]
                    elif (IC == "ort_l") and ((i == i_p and j - j_p == 1) or (i - i_p == 1 and j ==j_p)):
                        constraints += [G[i,j] - c1 * W[i] / theta_i - c2 * L[j] / theta_j == G[i_p,j_p] - c1 * W[i_p] / theta_i - c2 * L[j_p] / theta_j]

    # Monotonicity
    if monotonicity:
        constraints += [0 <= W[0]]
        for i in range(I-1): 
            constraints += [W[i] <= W[i+1]]
        constraints += [0 <= L[0]]
        for j in range(J-1):
            constraints += [L[j] <= L[j+1]]
        for i in range(I):
            constraints += [0 <= G[i,0]]
            for j in range(J-1):
                constraints += [G[i,j] <= G[i,j+1]]
        for j in range(J):
            constraints += [0 <= G[0,j]]
            for i in range(I-1):
                constraints += [G[i,j] <= G[i+1,j]]
    

    # TODO: Ordering constraints
    obj = cp.Maximize(objective_func)
    prob = cp.Problem(obj, constraints)
    prob.solve(verbose=False)
    if prob.status != "optimal":
        raise Exception("Optimal contracts not found")
    if my_print: print("G")
    if my_print: print(G.value)
    if my_print: print("\nW")
    if my_print: print(W.value)
    if my_print: print("\nL")
    if my_print: print(L.value)
    return G.value, W.value, L.value

def time_var_contracts(thetas = [1/1.5, 1/1.25, 1/1,  1/0.75, 1/0.5],
                       laxes = [1,2,3,4,5],
                       c1 = 0.01,
                       c2 = 0.1,
                       kappa1 = 0.1,
                       kappa2 = 0.5,
                       alpha_d = 11,
                       psi = 0.49,
                       my_print = False,
                       u_vpp = "sep",
                       g_const = True,
                       lax_ic_const = True
                      ):

    if my_print: print(f"{thetas=}")
    if my_print: print(f"{laxes=}")

    M = len(thetas)
    L = len(laxes)
    G = cp.Variable((M, L), nonneg=True)
    W = cp.Variable((M, L), nonneg=True)

    PI = np.ones((M,L)) / (M*L)
    if my_print: print(f"{PI=}")

    constraints = []
    objective_func = 0

    # The loops are repetitive, but they are useful for printing everything in order

    # Objective
    if my_print: print("Maximize over G, W")
    for i, theta in enumerate(thetas):
        for j, lax in enumerate(laxes):
            if u_vpp == "mult":
                # From a VERY superficial analysis, it doesn't seem to make much of a difference
                objective_func += PI[i,j] * (kappa1 * cp.log(W[i,j]*lax + kappa2 * lax +1) - G[i,j])
                if my_print: print(f"PI[{i},{j}] * (kappa1 * cp.log(W[{i},{j}]*{lax} + kappa2 * {lax} + 1) - G[{i},{j}])")

            if u_vpp == "sep":
                objective_func += PI[i,j] * (kappa1 * cp.log(W[i,j]+1)  + kappa2 * cp.log(lax) - G[i,j])
                if my_print: print(f" + {PI[i,j]:.1f} * (kappa1 * cp.log(W[{i},{j}]+1) + kappa2 * cp.log({lax}+1) - G[{i},{j}])")
        if my_print: print("")

    # Constraints
    if my_print: print("such that:")

    if my_print: print("\nIndividual Rationality")
    for i, theta in enumerate(thetas):
        for j, lax in enumerate(laxes):
            # Individual rationality
            constraints += [G[i,j] - c1 * W[i,j] / theta  - c2 * lax >= 0]
            if my_print: print(f"G[{i}, {j}] - {c1} * W[{i}, {j}] / {theta} - {c2} * {lax} >= 0")
        if my_print: print("")

    if my_print: print("\nIncentive compatibility for degradation")
    for i, theta in enumerate(thetas):
        for j, lax in enumerate(laxes):
            # Incentive compatibiliy for degradation
            for p, y in enumerate(thetas):
                if y != theta:
                    constraints += [G[i,j] - c1 * W[i,j] / theta - c2 * lax >= G[p,j] - c1 * W[p,j] / theta - c2 * lax ]
                    if my_print: print(f"G[{i},{j}] - {c1} * W[{i},{j}] / {theta} - {c2} * {lax} >= G[{p},{j}] - {c1} * W[{p},{j}] / {theta} - {c2} * {lax} ")
        if my_print: print("")

    if lax_ic_const:
        if my_print: print("\nIncentive compatibility for laxity")
        for i, theta in enumerate(thetas):
            for j, lax in enumerate(laxes):
                # Incentive compatibility for time
                for q, z in enumerate(laxes):
                    if z < lax: # z < lax
                        constraints += [G[i,j] - c1 * W[i,j] / theta - c2 * lax >= G[i,q] - c1 * W[i,q] / theta - c2 * lax ]
                        if my_print: print(f"G[{i},{j}] - {c1} * W[{i},{j}] / {theta}- {c2} * {lax} >= G[{i},{q}] - {c1} * W[{i},{q}] / {theta} - {c2} * {lax} ")
            if my_print: print("")
            
            
    if my_print: print("\nEnergy ordering")
    for j, lax in enumerate(laxes):
        for i, theta in enumerate(thetas):
            # Energy ordering constraint
            if i == 0:
                constraints += [0 == W[i,j]]
                if my_print: print(f"0 == W[{i},{j}]")

            if i == M-1:
                constraints += [W[i,j] <= lax * psi * alpha_d]
                if my_print: print(f"W[{i},{j}] <= {lax} * {psi} * {alpha_d}")
            else:
                constraints += [W[i,j] <= W[i+1,j]]
                if my_print: print(f"W[{i},{j}] <= W[{i+1},{j}]")
        if my_print: print("")
            
    if g_const:
        if my_print: print("\nPayoff ordering by energy")
        for j, lax in enumerate(laxes):
            for i, theta in enumerate(thetas):
                # Payoff ordering constraint
                if i == 0:
                    constraints += [0 <= G[i,j]]
                    if my_print: print(f"0 <= G[{i},{j}]")

                if i < M-1:
                    constraints += [G[i,j] <= G[i+1,j]]
                    if my_print: print(f"G[{i},{j}] <= G[{i+1},{j}]")
            if my_print: print("")

        if my_print: print("\nPayoff ordering by laxity")
        for i, theta in enumerate(thetas):
            for j, lax in enumerate(laxes):
                # Laxity first constraints
                if j == 0:
                    constraints +=[0 <= G[i,j]]
                    if my_print: print(f"0 <= G[{i},{j}]")

                if j < L-1:
                    constraints += [G[i,j] <= G[i,j+1]]
                    if my_print: print(f"G[{i},{j}] <= G[{i},{j+1}]")
            if my_print: print("")

    obj = cp.Maximize(objective_func)
    prob = cp.Problem(obj, constraints)
    prob.solve(verbose=False)
    if prob.status != "optimal":
        raise Exception("Optimal contracts not found")
    if my_print: print("G")
    if my_print: print(G.value)
    if my_print: print("\nW")
    if my_print: print(W.value)
    return G.value, W.value


def time_var_contracts_typelax(thetas = [1/1.5, 1/1.25, 1/1,  1/0.75, 1/0.5],
                       laxes = [1,2,3,4,5],
                       c1 = 0.01,
                       c2 = 0.1,
                       kappa1 = 0.1,
                       kappa2 = 0.5,
                       alpha_d = 11,
                       psi = 0.49,
                       my_print = False,
                       u_vpp = "sep",
                       g_const = True,
                       lax_ic_const = True
                      ):

    if my_print: print(f"{thetas=}")
    if my_print: print(f"{laxes=}")

    M = len(thetas)
    L = len(laxes)
    G = cp.Variable((M, L), nonneg=True)
    W = cp.Variable((M, L), nonneg=True)

    PI = np.ones((M,L)) / (M*L)
    if my_print: print(f"{PI=}")

    constraints = []
    objective_func = 0

    # The loops are repetitive, but they are useful for printing everything in order

    # Objective
    if my_print: print("Maximize over G, W")
    for i, theta in enumerate(thetas):
        for j, lax in enumerate(laxes):
            if u_vpp == "mult":
                # From a VERY superficial analysis, it doesn't seem to make much of a difference
                objective_func += PI[i,j] * (kappa1 * cp.log(W[i,j]*lax + kappa2 * lax +1) - G[i,j])
                if my_print: print(f"PI[{i},{j}] * (kappa1 * cp.log(W[{i},{j}]*{lax} + kappa2 * {lax} + 1) - G[{i},{j}])")

            if u_vpp == "sep":
                objective_func += PI[i,j] * (kappa1 * cp.log(W[i,j]+1)  + kappa2 * cp.log(lax) - G[i,j])
                if my_print: print(f" + {PI[i,j]:.1f} * (kappa1 * cp.log(W[{i},{j}]+1) + kappa2 * cp.log({lax}+1) - G[{i},{j}])")
        if my_print: print("")

    # Constraints
    if my_print: print("such that:")

    if my_print: print("\nIndividual Rationality")
    for i, theta in enumerate(thetas):
        for j, lax in enumerate(laxes):
            # Individual rationality
            constraints += [G[i,j] - c1 * W[i,j] / theta  - c2 * lax / theta >= 0]
            if my_print: print(f"G[{i}, {j}] - {c1} * W[{i}, {j}] / {theta} - {c2} * {lax} >= 0")
        if my_print: print("")

    if my_print: print("\nIncentive compatibility for degradation")
    for i, theta in enumerate(thetas):
        for j, lax in enumerate(laxes):
            # Incentive compatibiliy for degradation
            for p, y in enumerate(thetas):
                if y != theta:
                    constraints += [G[i,j] - c1 * W[i,j] / theta - c2 * lax / theta >= G[p,j] - c1 * W[p,j] / theta - c2 * lax / theta ]
                    if my_print: print(f"G[{i},{j}] - {c1} * W[{i},{j}] / {theta} - {c2} * {lax} / {theta} >= G[{p},{j}] - {c1} * W[{p},{j}] / {theta} - {c2} * {lax} / {theta} ")
        if my_print: print("")

    if lax_ic_const:
        if my_print: print("\nIncentive compatibility for laxity")
        for i, theta in enumerate(thetas):
            for j, lax in enumerate(laxes):
                # Incentive compatibility for time
                for q, z in enumerate(laxes):
                    if z < lax: # z < lax
                        constraints += [G[i,j] - c1 * W[i,j] / theta - c2 * lax / theta >= G[i,q] - c1 * W[i,q] / theta - c2 * lax / theta ]
                        if my_print: print(f"G[{i},{j}] - {c1} * W[{i},{j}] / {theta} - {c2} * {lax} >= G[{i},{q}] - {c1} * W[{i},{q}] / {theta} - {c2} * {lax} / {theta}")
            if my_print: print("")
            
            
    if my_print: print("\nEnergy ordering")
    for j, lax in enumerate(laxes):
        for i, theta in enumerate(thetas):
            # Energy ordering constraint
            if i == 0:
                constraints += [0 == W[i,j]]
                if my_print: print(f"0 == W[{i},{j}]")

            if i == M-1:
                constraints += [W[i,j] <= lax * psi * alpha_d]
                if my_print: print(f"W[{i},{j}] <= {lax} * {psi} * {alpha_d}")
            else:
                constraints += [W[i,j] <= W[i+1,j]]
                if my_print: print(f"W[{i},{j}] <= W[{i+1},{j}]")
        if my_print: print("")
            
    if g_const:
        if my_print: print("\nPayoff ordering by energy")
        for j, lax in enumerate(laxes):
            for i, theta in enumerate(thetas):
                # Payoff ordering constraint
                if i == 0:
                    constraints += [0 <= G[i,j]]
                    if my_print: print(f"0 <= G[{i},{j}]")

                if i < M-1:
                    constraints += [G[i,j] <= G[i+1,j]]
                    if my_print: print(f"G[{i},{j}] <= G[{i+1},{j}]")
            if my_print: print("")

        if my_print: print("\nPayoff ordering by laxity")
        for i, theta in enumerate(thetas):
            for j, lax in enumerate(laxes):
                # Laxity first constraints
                if j == 0:
                    constraints +=[0 <= G[i,j]]
                    if my_print: print(f"0 <= G[{i},{j}]")

                if j < L-1:
                    constraints += [G[i,j] <= G[i,j+1]]
                    if my_print: print(f"G[{i},{j}] <= G[{i},{j+1}]")
            if my_print: print("")

    obj = cp.Maximize(objective_func)
    prob = cp.Problem(obj, constraints)
    prob.solve(verbose=False)
    if prob.status != "optimal":
        raise Exception("Optimal contracts not found")
    if my_print: print("G")
    if my_print: print(G.value)
    if my_print: print("\nW")
    if my_print: print(W.value)
    return G.value, W.value


def time_var_contracts_tract(thetas = [1/1.5, 1/1.25, 1/1,  1/0.75, 1/0.5],
                       laxes = [1,2,3,4,5],
                       c1 = 0.01,
                       c2 = 0.5,
                       kappa1 = 0.1,
                       kappa2 = 0.5,
                       alpha_d = 11,
                       psi = 0.49,
                       my_print = False,
                       #u_vpp = "sep",
                       #g_const = False,
                       #lax_ic_const = False
                      ):
    if my_print: print(f"{thetas=}")
    if my_print: print(f"{laxes=}")

    G_acc = []
    W_acc = []

    M = len(thetas)
    L = len(laxes)

    for lax in laxes:
        constraints = []
        objective_func = 0
        G = cp.Variable((M), nonneg=True)
        W = cp.Variable((M), nonneg=True)
        PI = np.ones((M)) / (M)
        if my_print: print(f"{PI=}")

        if my_print: print("Maximize over G, W")
        for i, theta in enumerate(thetas):
            objective_func += PI[i]*(kappa1*cp.log(W[i]+1) - G[i])
            if my_print: print(f" PI[{i}]*(kappa1 * cp.log(W[{i}]+1)- G[{i}])")

        
        # Individual rationality 0
        if my_print: print("\nIndividual rationality 0")
        constraints += [G[0] - c1 * W[0]  / thetas[0]- c2 * lax == 0] 
        if my_print: print(f"G[0] - c1 * thetas[0] * W[0] - c2 * lax == 0")

        # Binding incentive constraints
        if my_print: print("\nBinding incentive constraints")
        for i in range(1, M):
            constraints += [G[i] - c1 * W[i] / thetas[i]- c2 * lax == G[i-1] - c1 * W[i-1] / thetas[i]- c2 * lax] # c2*lax can be eliminated
            if my_print: print(f"G[{i}] - c1 * thetas[{i}] * W[{i}] - c2 * {lax} == G[{i-1}] - c1 * thetas[{i}] * W[{i-1}] - c2 * {lax}") 
    
        # Energy ordering 
        if my_print: print("\nEnergy ordering")
        for i, theta in enumerate(thetas):
            if i == 0:
                constraints += [0 == W[i]]
                if my_print: print(f"0 == W[{i}]")
            if i == M-1:
                constraints += [W[i] <= lax * psi * alpha_d]
                if my_print: print(f"W[{i}] <= {lax} * psi * alpha_d")
            else:
                constraints += [W[i] <= W[i+1]]
                if my_print: print(f"W[{i}] <= W[{i+1}]")
        obj = cp.Maximize(objective_func)
        prob = cp.Problem(obj, constraints)
        prob.solve(verbose=False)
        if prob.status != "optimal":
            raise Exception(f"Optimal contracts not found at lax {lax}")
        if my_print:
            print("G")
            print(G.value)
            print("\nW")
            print(W.value)
            print("-------\n")
        G_acc.append(G.value)
        W_acc.append(W.value)

    return np.array(G_acc).T, np.array(W_acc).T

def u_ev(g, w, theta, lax, c1 = 0.01, c2 = 0.1):
    return g - w*c1/theta - c2*lax 

def u_ev_typelax(g, w, theta, lax, c1 = 0.01, c2 = 0.1):
    return g - w*c1/theta - c2*lax/theta

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    laxes = [1,2,3,4,5]
    alpha_d = 11
    psi = 0.49
    G, W = time_var_contracts_tract(laxes = laxes)

    fig1 = plt.figure(figsize = (8, 6))
    ax1 = fig1.add_subplot(1,1,1)
    colors = ["red", "blue", "green", "orange", "purple"]
    ax1.set_ylabel("Payoff (g)")
    for i in range(len(laxes)):
        ax1.scatter(W[:,i],G[:,i], alpha = 0.5, color=colors[i], label = f"lax = {laxes[i]}")
        ax1.vlines(laxes[i]*alpha_d*psi, -0.1, 3.5, color=colors[i], ls = "--", alpha=0.5)
    ax1.set_xlim([-1, 30]) 
    ax1.set_ylim([-0.1, 3.5])
    ax1.set_xlabel("Energy (w)")
    ax1.legend()
    fig1.show()
    input()


