# Ev ChargeEnv Documentation
This environment simulates an EV Charging Park (aka _park_) that contains many EV Charging Units. One unit can charge one car per _session_.

The term "charging station" is ambiguous and therefore is avoided.

This environment runs on **discrete time**. The timestep is measured in minutes, and a common setting is $\Delta t = 15 min$. [There is a reference that says that some countries regulations now focus on 15 min granularity instead of 1 hour granularity, need to check].


## State
The state takes inspiration from board games. In this case it's a single player game (PvE).

| Board Game | EvCharge      |
|------------|---------------|
| Board      | Charging park |
| Square     | Charging unit |
| Piece      | Car           |
| Turn       | Timestep      |
| Move       | _action_      |

### Charging park
Defined by a single constant

* `max_cars`: The maximum number of cars in the Park. This is equivalent to the number of board 

For now the size of the battery `C` will be the same for every car.

Therefore, the environment is a list of `max_cars` spots (units). Each spot can be occupied or vacant.

### Car
The input is a DataFrame containing Cars (also called sessions or transactions). 

The car iterator is $i \in \{0, 1, ..., maxcars \}$
Each row will have:


#### → Car constant attributes
* TransactionId (`idSess`): A unique identifier for that row (0 or -1 if spot is vacant)
* BatteryCapacity (`C`): The size of the battery (in Wh)
* TransactionStartTS (`t_arr`): The timestep in which the car arrives to the parking lot (in ts)
* `SOC_arr` (`soc_arr`): The State-Of-Charge with which the car arrives (from 0 to 1)
* TransactionStopTS (`t_dep`): Time of departure of the EV (in ts)

#### → Car variable attributes
This variables are the ones that will be simulated (i.e. recalculated at every timestep)

* `soc_t`: State of charge at timestep t

The update rule is:
$soc_{t}^i = (P_{t}^i \cdot \Delta t + soc_{t-1}^i)/C^i$

Although, by ignoring the constants, it can be simplified to: 
$soc_{t}^i = a_{t}^i + soc_{t-1}^i$

With $a_{t}^i$ being the "normalized" power. In which case, there is no use for battery size, `C`.

#### → Car auxiliary attributes
By simple arithmetic operations, the following features can be engineered. The value of them has not yet been measured.

* Sojurn time (`t_soj`): $t_{soj}^i = t_{dep}^i - t_{arr}^i$
* Energy at arrival (`E_arr`): $E_{arr}^i = soc_{arr}^i \cdot C^i$
* Energy at t (`E_t`): $E_{t}^i = soc_{t}^i \cdot C^i$
* Energy required (`E_req`): $E_{req}^i = C^i - E_{t}$
* Remaining time (`t_rem`): $t_{rem}^i = t_{dep}^i - t$

### Observation space
In the simplest case, ignoring `C` and auxiliary features, the space will be:

`max_cars` * ({`idSess`, `t_arr`, `soc_arr`, `t_dep`, `soc_t`} + {Additional attributes})

In it's simplest form in RlGym terms:

`Box([0, 0, 0, 0, 0], [max_idSess, t_max, 1, t_max, 1], (max_cars, 5))`

I.e. `max_cars` squares with each piece having 5 attributes.

Additionally, the timestep $t$ is a global variable.

## Agent
The agent is the Virtual Power Plant (VPP) or Virtual Operating Reserve (VOR) that sets the power for each charging session.

### Action space
The action will be $P_t = {p_t^1, p_t^2, ..., p_t^{maxcars}}$ in Watts.

Or $A_t = {a_t^1, a_t^2, ..., a_t^{maxcars}}$ in "normalized units".

In RlGym terms:

`Box(P_min, P_max, (max_cars, 1))`

## Notes
In RlGym: `Box(lower_lim, upper_lim, shape)`
