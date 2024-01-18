import numpy as np
from math import isclose
from icecream import ic # type: ignore
import cvxpy as cp
import os

def zipf(N, s):
    x = np.arange(N)
    # Warning divide by zero
    H = (1 / (x+1)**s).sum()
    return 1/H * 1/(x+1)**s

def priority(Y_tot, lower, upper, occ_spots, priority_vals):
    priority_list = np.argsort(priority_vals)
    y = lower.copy() 
    Y_temp = Y_tot - lower.sum()
    for i in priority_list:
        if i in occ_spots:
            y_i = np.min([Y_temp, upper[i] - lower[i]])
            y[i] += y_i
            Y_temp -= y_i
            if isclose(Y_temp, 0):
                break
    return y

def prioritySoft(Y_tot, lower, upper, occ_spots, priority_vals, s = 2):
    y = lower.copy()
    Y_temp = Y_tot - lower.sum()
    Range_y = upper - lower
    while (Y_temp - 0.0000001 > 0) and (Range_y.sum() > 0) and (occ_spots.sum() > 0):
        idx = y < upper
        priority_list = np.argsort(priority_vals[idx])
        prop = zipf(len(priority_list), s)[priority_list]
        y[idx] += prop*Y_temp 
        extra = np.maximum(y[idx]-upper[idx],0).sum()
        y[idx] = np.minimum(y[idx], upper[idx])
        idx = y < upper
        Y_temp = extra
    return y

def proportional(Y_tot, lower, upper, occ_spots):
    Range_y = upper-lower
    y = lower.copy()
    if Range_y.sum() > 0:
        range_y = (Range_y) / (Range_y).sum()
        Y_temp = Y_tot - lower.sum()
        y += range_y * Y_temp
    return y

def proportionalFairness(Y_tot, lower, upper, occ_spots):
    try:
        upper = upper.to_numpy(dtype = np.float64)
        lower = lower.to_numpy(dtype = np.float64)

        ev_ranges = (upper - lower)
        bool_ranges = np.array([not isclose(ev_range, 0, abs_tol = 1e-05) for ev_range in ev_ranges]) #
        to_disagg = bool_ranges & occ_spots
        n = sum(to_disagg)
        Y_surplus = (Y_tot - lower.sum())


        if n > 0:
            Y = cp.Variable(n)
            s_lower = lower[to_disagg] # Subset lower
            s_upper = upper[to_disagg] # Subset upper
            s_range = s_upper - s_lower

            if Y_surplus > s_range.sum():
                #print(f"Warning: surplus larger by: {Y_surplus - s_range.sum()}")
                if Y_surplus - s_range.sum() > 0.0001:
                    raise "Discrepancy too large"
                Y_surplus = s_range.sum()

            constraints = []
            constraints += [0 <= Y]
            constraints += [Y <= s_range]
            constraints += [cp.sum(Y) == Y_surplus]

            objective = cp.Maximize(cp.sum(cp.log(Y+1)))
            prob = cp.Problem(objective, constraints)

            # Custom error message
            try:
                prob.solve(cp.MOSEK, mosek_params = {'MSK_IPAR_NUM_THREADS': 8, 'MSK_IPAR_BI_MAX_ITERATIONS': 2_000_000, "MSK_IPAR_INTPNT_MAX_ITERATIONS": 800}, verbose=False)  
                if prob.status != 'optimal':
                    print(prob.status)
                    raise Exception("Optimal disaggregation not found")
            except Exception:
                print("----")
                print("low", [f"{x:.4f}" for x in lower])
                print("upp", [f"{x:.4f}" for x in upper])
                print(n)
                print("ran", [f"{x:.4f}" for x in (s_range)], s_range.sum())
                print(Y_surplus)
                print("Error in proportional fairness disaggregation")
                #raise SystemExit('Error in proportional fairness disaggregation.')

        y = lower.copy()
        j = 0
        for i in range(len(lower)):
            if to_disagg[i]:
                y[i] += Y.value[j]
                j += 1
    except Exception:
        print("WARNING! proportionalFairness failed")
        y = proportional(Y_tot, lower, upper, occ_spots)
        #Super hacky way to track errors
        os.system("echo 'err' >> err_disagg.txt")
    return y
