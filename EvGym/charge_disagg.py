import numpy as np
from math import isclose
from icecream import ic # type: ignore

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
