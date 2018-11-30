[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# Project 2: Continuous Control

This repository contains solution to Project 2 of the Deep Reinforcement Learning course at Udacity.

## Project details

The objective of this project is to solve a "Reacher" environment. Essentially, we have a double-jointed arm which is required to move to a target position.
For each time step the agent the arm in the goal position, a reward of +.1 is given, whereas no reward is given at each time step the agent fails to 
keep the arm in the target position. Consequently, we want a agent which keeps the arm in the target position at all times.

### State and agent space

The state space has 33 dimensions, corresponding to position, rotation, velocity, and angular velocities of the arm. 

In each time step, the agent must decide four actions. These four numbers represent the torque applied to the two joints of the arm.
Each of these numbers should be between $[-1,1]$. 

The task is episodic, and the problem is considered solved when the agent received an average score of +30 over 100 consecutive episodes.

### Distributed training

One of the strength of humans is the ability to learn from each other. Here, we also attempt some kind of knowledge sharing. That is, we will have 20
parallel environments, each with its own agent and observation space. 

## Getting started

### Github

Please start by cloning the following repo [here](https://github.com/udacity/deep-reinforcement-learning)

### Setup Python + Jupyter

To use the code, please follow the excellent recipe given [here](https://github.com/udacity/deep-reinforcement-learning#dependencies).
However, note that if you are on a Windows OS, in Step 3 of the recipe, remove `torch==0.4.0` from `requirements.txt`, and run the following lines instead

* `pip3 install http://download.pytorch.org/whl/cu90/torch-0.4.1-cp36-cp36m-win_amd64.whl`
* `pip3 install torchvision`

### Download the environment

After successfully installing dependencies, we must build the environment. Please download the version suitable to your OS

* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

Then place the file in the `/p2_continuous_control` catalogue, and also overwrite `/p2_navigation/Continuous_control.ipynb`
from the `https://github.com/udacity/deep-reinforcement-learning` repo, with the file in this repo, and you are set!

## Instructions

To train the agent, please look at `Continuous_control.ipynb`. This file will take you though the process, from importing the necessary packages to training the model. Essentially, everything regarding the training of the model is given in Part 4 of the file.
The underlying algorithm is written in `ddpg_agent.py`, wheareas the neural networks used by the agent is written in `model.py`.
