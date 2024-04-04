# EvCharge
This is the accompanying repo for the paper "Efficient Trading of Aggregate Bidirectional EV Charging Flexibility with Reinforcement Learning", due to appear on ACM's e-Energy 2024 proceedings. You can find it in this repo as `AggregateFlex_eEnergy24.pdf`.

## Abstract
We study a virtual power plant (VPP) that trades the bidirectional charging flexibility of privately owned plug-in electric vehicles (EVs) in a real-time electricity market to maximize its profit. To incentivize EVs to allow bidirectional charging, we design incentive-compatible, variable-term contracts between the VPP and EVs. Through deliberate aggregation of the energy storage capacity of individual EVs, we learn a reinforcement learning (RL) policy to efficiently trade the flexibility, independent of the number of accepted contracts and connected EVs. The proposed aggregation method ensures the satisfaction of individual EV charging requirements by constraining the optimal action returned by the RL policy within certain bounds. We then develop a disaggregation scheme to allocate power to bidirectional chargers in a proportionally fair manner, given the total amount of energy traded in the market. Evaluation on a real-world dataset demonstrates robust performance of the proposed method despite uncertainties in electricity prices and shifts in the distribution of EV mobility.

## Overview
The code is divided into these parts
* `ContractDesign/`: The code and notebooks used to generate and analyze the incentive-compatible V2G contracts. 
* `ElectricityMarkets/`: The analysis for the electricity price dataset at the proper time resolution. 
* `EvGym/`: The environment and the RL agents, more details in a later section.
* `ExpLogs/`: Records of the experiments used in the paper.
* `ResultsAnalysis/`: Notebooks used for analysis and visualization. 
* `data/`: Datasets used simulations.
* `scripts/`: The `Bash` scripts used for experiments, multiple simulation runs.
* `time/`: Results for time profiling of different agents.
* `PreprocElaad.ipynb`: Notebook for preprocessing Elaad, charging sessions dataset
* `RunChargeWorld.py`: Script for running simulations without RL.
* `RunSACChargeWorld.py`: Script for running simulation with RL

Additionally a `requirements.txt` file is provided.
Using a `virtualenv` is recommended.

## Notes for `RunSACChargeWorld.py`
This is of how we train our implementation of _Aggregate SAC_. 
First, we import some general modules. Then we import the user-defined modules, mainly the environment (`ChargeWorldEnv`), the actor (`agentSAC\_sagg`), and the critic (`SoftQNetwork`).

In the body of the program, we initialize `ChargeWorldEnv` with the dataset that contains the charging sessions (`df_sessions`), the dataset that contains the real-time prices (`df_prices`), the contract parameters (`contract_info`), and a random number generator (`rng`).

For the agent, we initialize the actor (`agentSAC_agg`) with the price dataset (`df_price`), arguments read from the command line (`args`), and the device  (`device`). This `device` is needed for certain PyTorch functionalities. The critic is composed of two Q networks (`SoftQNetwork`).
Additionally, soft actor-critic uses a replay buffer (`rb`). 

We can train the agent for many `episodes`, each with a predetermined number of `timesteps`.
Similar to Farama's Gym, the environment is initialized with a `world.reset()`. 
During training, the agent receives an observation from the environment, and it outputs an action with `agent.get_action()`. 
The environment receives the action and moves forward one timestep with \texttt{world.step()}.
The loop keeps going on until all the timesteps are completed for all the episodes.
At each iteration, the training of the agent is performed. 

The charging sessions dataset, real-time prices dataset and state are implemented in Pandas DataFrames.
Additionally, the environment also receives a Pandas DataFrame for the action.
Conversely, the agent works mainly with PyTorch Tensors.
To convert the state DataFrame into a PyTorch Tensor, we employ \texttt{agent.df\_to\_state()}.
Similarly, to convert the agent's action into the required Pandas format that the environment prefers, we use \texttt{agent.action\_to\_env()}.
