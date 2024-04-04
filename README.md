# :zap: :car: :zap: EvCharge 
This is the accompanying repo for the paper "Efficient Trading of Aggregate Bidirectional EV Charging Flexibility with Reinforcement Learning", due to appear on ACM's e-Energy 2024 proceedings. You can find it in this repo as `AggregateFlex_eEnergy24.pdf`.

## :cloud_with_lightning: Abstract 
We study a virtual power plant (VPP) that trades the bidirectional charging flexibility of privately owned plug-in electric vehicles (EVs) in a real-time electricity market to maximize its profit. To incentivize EVs to allow bidirectional charging, we design incentive-compatible, variable-term contracts between the VPP and EVs. Through deliberate aggregation of the energy storage capacity of individual EVs, we learn a reinforcement learning (RL) policy to efficiently trade the flexibility, independent of the number of accepted contracts and connected EVs. The proposed aggregation method ensures the satisfaction of individual EV charging requirements by constraining the optimal action returned by the RL policy within certain bounds. We then develop a disaggregation scheme to allocate power to bidirectional chargers in a proportionally fair manner, given the total amount of energy traded in the market. Evaluation on a real-world dataset demonstrates robust performance of the proposed method despite uncertainties in electricity prices and shifts in the distribution of EV mobility.

## :telescope: Overview 
The code is divided into these parts
* :scroll: `ContractDesign/`: The code and notebooks used to generate and analyze the V2G contracts. 
* :electric_plug: `ElectricityMarkets/`: The analysis for the electricity price dataset at the proper time resolution. 
* :weight_lifting: `EvGym/`: The environment and the RL agents, more details in a later section.
* :test_tube: `ExpLogs/`: Records of the experiments used in the paper.
* :microscope: `ResultsAnalysis/`: Notebooks used for analysis and visualization. 
* :books: `data/`: Datasets used simulations.
* :chess_pawn: `scripts/`: The `Bash` scripts used for experiments, multiple simulation runs.
* :hourglass: `time/`: Results for time profiling of different agents.
* :stew: `PreprocElaad.ipynb`: Notebook for preprocessing Elaad, charging sessions dataset
* :star: `RunChargeWorld.py`: Script for running simulations without RL.
* :star2: `RunSACChargeWorld.py`: Script for running simulation with RL

Additionally a `requirements.txt` file is provided.
Using a `virtualenv` is recommended.

## :gear: Parameters 
| Parameter                    | Value    | Description                                      |
|------------------------------|----------|--------------------------------------------------|
| `--agent`                    | SAC-sagg | Agent to use for real-time scheduling            |
| `--save-name`                | sac_a    | Name used for logs, results, etc.                |
| `--pred-noise`               | 0.00     | Nosie for price predictions in training          |
| `--seed`                     | 42       | Seed for random number generators                |
| `--years`                    | 200      | Number of episodes to train                      |
| `--batch-size`               | 512      | Batch size to sample from the replay buffer      |
| `--alpha`                    | 0.02     | Temperature parameter in SAC                     |
| `--policy-frequency`         | 4        | How often to update the policy (timesteps)       |
| `--target-network-frequency` | 2        | How often to update the second Q NN (timesteps)  |
| `--disagg`                   | PF       | (Proportional fairness) Disaggregation algorithm |
| `--buffer-size`              | 1e6      | Number of experiences to save in replay buffer   |
| `--save-agent`               | True     | Save the weights of the trained agent            |
| `--general`                  | True     | Run training mode (`False` is for deployment)    |

## Architecture :brain:

### Actor :person_fencing:
The architecture for the actor, the policy network. 

| Layer                          | In  | Out |
|--------------------------------|-----|-----|
| Linear (ReLU)                  | 59  | 256 |
| Linear (ReLU)                  | 256 | 256 |
| Head 1, Mean: Linear (Sigmoid) | 256 | 1   |
| Head 2, Logstd: Linear (Tanh)  | 256 | 1   |

### Critic :detective:
The architecture for the two critics, soft Q networks.

| Layer         | In  | Out |
|---------------|-----|-----|
| Linear (ReLU) | 60  | 256 |
| Linear (ReLU) | 256 | 256 |
| Linear        | 256 | 1   |

## Notes for `RunSACChargeWorld.py` :city_sunrise:	
This is of how we train our implementation of _Aggregate SAC_. 
First, we import some general modules. Then we import the user-defined modules, mainly the environment (`ChargeWorldEnv`), the actor (`agentSAC_sagg`), and the critic (`SoftQNetwork`).

In the body of the program, we initialize `ChargeWorldEnv` with the dataset that contains the charging sessions (`df_sessions`), the dataset that contains the real-time prices (`df_prices`), the contract parameters (`contract_info`), and a random number generator (`rng`).

For the agent, we initialize the actor (`agentSAC_agg`) with the price dataset (`df_price`), arguments read from the command line (`args`), and the device  (`device`). This `device` is needed for certain PyTorch functionalities. The critic is composed of two Q networks (`SoftQNetwork`).
Additionally, soft actor-critic uses a replay buffer (`rb`). 

We can train the agent for many `episodes`, each with a predetermined number of `timesteps`.
Similar to Farama's Gym, the environment is initialized with a `world.reset()`. 
During training, the agent receives an observation from the environment, and it outputs an action with `agent.get_action()`. 
The environment receives the action and moves forward one timestep with `world.step()`.
The loop keeps going on until all the timesteps are completed for all the episodes.
At each iteration, the training of the agent is performed. 

The charging sessions dataset, real-time prices dataset and state are implemented in Pandas DataFrames.
Additionally, the environment also receives a Pandas DataFrame for the action.
Conversely, the agent works mainly with PyTorch Tensors.
To convert the state DataFrame into a PyTorch Tensor, we employ `agent.df_to_state()`.
Similarly, to convert the agent's action into the required Pandas format that the environment prefers, we use `agent.action_to_env()`.
