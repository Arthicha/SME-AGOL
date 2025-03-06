# SME-AGOL: Sequential Motion Executor - Added Gradient-weighting Online Learning

The video is available at [https://youtu.be/XCI1opte-VA](https://youtu.be/XCI1opte-VA).

<p align="center">
    <img width="75%" src="/pictures/simulation.PNG" />
</p>

# Contents
- [Requirements](#Requirements)
- [Running](#Running)

# Requirements

* simulation software
	- CoppeliaSim 4.4.0 (at least)
	- Mujoco physic engine (come with Coppeliasim > 4.4.0)

* python 3.6.5
	- numpy 1.19.5 (at least)
	- pytorch 1.5.0+cu92 (at least)
 
# Running

1. Open the CoppeliaSim scene locating at `simulation/MORF_BasicLocomotionLearning`

2. In order to start the training, just run the following command:

```
python main.py
```

3. If you want to try different hyperparameter values, you can modify them according to the table below.

| Location | Parameter | Meaning  |
| ------------- | ------------- | ------------- |
| network.ini | W_TIME | transition speed/walking freqeuncy | 
| optimizer.ini | MINGRAD | gradient clipping (prevent exploding gradient) | 
|  | LR | learning rate | 
|  | SIGMA | starting exploration standard deviation (between 0.001-0.05)|
| main.py | NREPLAY | number of episodes/roll-outs used |
|  | NTIMESTEP | number of timesteps per episode | 
|  | NEPISODE | number of episode used for learning | 
|  | RESET | enable simulation/network reset | 
|  |  | (reset the simulation and the network after each episode ends) | 

4. Enjoy! With a proper set of hyperparameters, the robot should start walking within the first 40 episodes.



Reference:

Arthicha Srisuchinnawong and Poramate Manoonpong. "An Interpretable Neural Control Network with Adaptable Online Learning for Sample Efficient Robot Locomotion Learning." arXiv preprint arXiv:2501.10698 (2025).