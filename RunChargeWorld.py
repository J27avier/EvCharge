# Modules
import pandas as pd
import numpy as np
from tabulate import tabulate
import os

# User defined modules
from EvGym.charge_world import ChargeWorldEnv
from EvGym.charge_agent import agentRB
from EvGym import config

def main():
    lst = [[1,2,3], [2,3,4], [4,5,6]]
    os.system("clear")

    for i in range(5):
        printWorld(lst)
        lst[0][0] += 1

def printWorld(lst):
    print(tabulate(lst, tablefmt = "grid"))
    input()
    os.system("clear")

if __name__ == "__main__":
    main()
