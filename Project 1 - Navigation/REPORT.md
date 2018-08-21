# Project report: Navigation

## Learning Algorithm

### Description of algorithm

The basic algorithm is built from the (Double Q-Network model)[url], with
a ReplayBuffer and Fixed Q-targets.

We start with randomly selecting actions using an $\epsilon$-greedy approach,
from which we observe a reward, $r_t$ the next state, $s_t$ and whether we are done, $d_t$.
Consequently, each experience is a tuple (%s_t%, %a_t%, %r_t%, %s_{t+1}%, %d_t%).



#### ReplayBuffer   

### Hyperparameters



## Plot of Rewards

## Ideas for Future Work
