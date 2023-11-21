import numpy as np
from math import isclose

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

def proportional(Y_tot, lower, upper, occ_spots):
    Range_y = upper-lower
    y = lower.copy()
    if Range_y.sum() > 0:
        range_y = (Range_y) / (Range_y).sum()
        Y_temp = Y_tot - lower.sum()
        y += range_y * Y_temp
    return y
