# EvCharge
This is the accompanying repo for the paper "Efficient Trading of Aggregate Bidirectional EV Charging Flexibility with Reinforcement Learning", due to appear on ACM's e-Energy 2024 proceedings. You can find it in this repo as `AggregateFlex_eEnergy24.pdf`.

## Abstract
We study a virtual power plant (VPP) that trades the bidirectional charging flexibility of privately owned plug-in electric vehicles (EVs) in a real-time electricity market to maximize its profit. To incentivize EVs to allow bidirectional charging, we design incentive-compatible, variable-term contracts between the VPP and EVs. Through deliberate aggregation of the energy storage capacity of individual EVs, we learn a reinforcement learning (RL) policy to efficiently trade the flexibility, independent of the number of accepted contracts and connected EVs. The proposed aggregation method ensures the satisfaction of individual EV charging requirements by constraining the optimal action returned by the RL policy within certain bounds. We then develop a disaggregation scheme to allocate power to bidirectional chargers in a proportionally fair manner, given the total amount of energy traded in the market. Evaluation on a real-world dataset demonstrates robust performance of the proposed method despite uncertainties in electricity prices and shifts in the distribution of EV mobility.

## Overview
The code is divided into these parts
* `ContractDesign/`: The code and notebooks used to generate and analyze the incentive-compatible V2G contracts. 
* `ElectricityMarkets/`: The preprocessing and analysis for the electricity price dataset. 
* `EvGym/`: Here is the environment and the agents 
* `ExpLogs/`
* `ResultsAnalysis/`
* `data/`
* `scripts/`
* `time/`
* `utils/`
* `PreprocElaad.ipynb`
* `RunChargeWorld.py`
* `RunSACChargeWorld.py`

Additionally a `requirements.txt` file is provided.
Using a `virtualenv` is recommended.

