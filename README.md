[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"


# Project 2: Continuous Control

### Introduction

For this project, we use the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of **+0.1** is provided for each step that the agent's hand is in the goal location. Thus, the goal of the trained agent is to maintain its position at the target location for as many time steps as possible.

The **observation space** consists of **33** variables corresponding to position, rotation, velocity, and angular velocities of the arm.

Each **action** is a vector with **4** numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number **between -1 and 1**.

### Distributed Training

For this project, we use a version of the the Unity environment that contains **20 identical agents**, each with its own copy of the environment.  

### Solving the Environment

To solve chosen version of the environment, the trained agents must get an average score of **+30** (**over 100 consecutive episodes**, and **over all agents**).
  
Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the **average score over 100 episodes is at least +30**. 

### Getting Started

1. Clone this repository, which contains all files to install dependencies, start the environment on Windows (64-bit) 
computer and (re)train the agents:
    - [Continuous_Control.ipynb](https://github.com/schambon77/DRLND-Continuous-Control/blob/master/Continuous_Control.ipynb): 
    Jupyter notebook to train the agent
    - [ddpg_agent.py](https://github.com/schambon77/DRLND-Continuous-Control/blob/master/ddpg_agent.py): modified DDPG code
    for 20 agents
    - [model.py](https://github.com/schambon77/DRLND-Continuous-Control/blob/master/model.py): neural network code for 
    DDPG actor and critic networks
    - [checkpoint_actor.pth](https://github.com/schambon77/DRLND-Continuous-Control/blob/master/checkpoint_actor.pth): 
    saved trained actor neural network coefficients
    - [checkpoint_critic.pth](https://github.com/schambon77/DRLND-Continuous-Control/blob/master/checkpoint_critic.pth): 
    saved trained critic neural network coefficients
    - [Reacher_Windows_x86_64_20agents](https://github.com/schambon77/DRLND-Continuous-Control/blob/master/Reacher_Windows_x86_64_20agents): 
    Reacher Unity enviroment with 20 agents for Windows 64-bit
    - [report.md](https://github.com/schambon77/DRLND-Continuous-Control/blob/master/report.md): report
 
2. If not using a Windows (64-bit) computer, download the environment from one of the links below. 
Refer to the comments in [Continuous_Control.ipynb](https://github.com/schambon77/DRLND-Continuous-Control/blob/master/Continuous_Control.ipynb)
to call the environment:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

3. Follow instructions in the [instructions in the DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) 
to set up your Python environment

### Instructions

Follow the instructions in [Continuous_Control.ipynb](https://github.com/schambon77/DRLND-Continuous-Control/blob/master/Continuous_Control.ipynb)
 to train the agent and see the result behaviour.

### Report

The [report](https://github.com/schambon77/DRLND-Continuous-Control/blob/master/report.md) provides a description of the implementation.
