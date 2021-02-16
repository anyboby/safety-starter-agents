# Safety Starter Agents

This fork provides several modifications to the safety starter agents repo. 

The branch <rllab> provides functionality for running experiments on rllab environments such as the circle and gather environments.

The branch <algos> provides modifications to cpo with experimental code for pathwise derivatives, maximum entropy objectives etc. in trust-region optimization. 
# Installation

To install this package, follow the common installation process for safety starter agents:

```
git clone https://github.com/anyboby/safety-starter-agents.git

cd safety-starter-agents

pip install -e .
```

**Warning:** Installing this package does **not** install Safety Gym. If you want to use the algorithms in this package to train agents on onstrained RL environments, make sure to install Safety Gym according to the instructions on the [Safety Gym repo](https://www.github.com/openai/safety-gym).

## Getting Started

**Example Script:** To run PPO-Lagrangian on the `Safexp-PointGoal1-v0` environment from Safety Gym, using neural networks of size (64,64):

```
from safe_rl import ppo_lagrangian
import gym, safety_gym

ppo_lagrangian(
	env_fn = lambda : gym.make('Safexp-PointGoal1-v0'),
	ac_kwargs = dict(hidden_sizes=(64,64))
	)

```


**Reproduce Experiments from Paper:** To reproduce an experiment from the paper, run:

```
cd /path/to/safety-starter-agents/scripts
python experiment.py --algo ALGO --task TASK --robot ROBOT --seed SEED 
	--exp_name EXP_NAME --cpu CPU
```

where 

* `ALGO` is in `['ppo', 'ppo_lagrangian', 'trpo', 'trpo_lagrangian', 'cpo']`.
* `TASK` is in `['goal1', 'goal2', 'button1', 'button2', 'push1', 'push2']` .
* `ROBOT` is in `['point', 'car', 'doggo']`.
* `SEED` is an integer. In the paper experiments, we used seeds of 0, 10, and 20, but results may not reproduce perfectly deterministically across machines.
* `CPU` is an integer for how many CPUs to parallelize across.

`EXP_NAME` is an optional argument for the name of the folder where results will be saved. The save folder will be placed in `/path/to/safety-starter-agents/data`. 


**Plot Results:** Plot results with:

```
cd /path/to/safety-starter-agents/scripts
python plot.py data/path/to/experiment
```

**Watch Trained Policies:** Test policies with:

```
cd /path/to/safety-starter-agents/scripts
python test_policy.py data/path/to/experiment
```


## Cite the Paper

If you use Safety Starter Agents code in your paper, please cite:

```
@article{Ray2019,
    author = {Ray, Alex and Achiam, Joshua and Amodei, Dario},
    title = {{Benchmarking Safe Exploration in Deep Reinforcement Learning}},
    year = {2019}
}
```
